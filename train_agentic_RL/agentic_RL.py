import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
sys.path.insert(0, str(BASE_DIR))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import os
import re
import gc
import json
import math
import random
import signal
import argparse
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate.data_loader import SkipBatchSampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import AgentRLDataset
from trainer_utils import Logger , lm_checkpoint, setup_seed, init_model, LMForRewardModel
from rollout_engine import create_rollout_engine, compute_per_token_logps

warnings.filterwarnings('ignore')

# ======== 模拟执行 ========
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-").replace("（", "(").replace("）", ")"), {"__builtins__": {}, "math": math}))},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * UNIT_DATA.get(f"{args.get('from_unit', '').lower()}_{args.get('to_unit', '').lower()}", 1), 4)},
    "get_current_weather": lambda args: (lambda w: {"city": args.get("location"), "temperature": w[0], "humidity": "65%", "condition": w[1]})(WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda args: {"datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency"), "to": args.get("to_currency"), "rate": EXCHANGE_DATA.get((args.get("from_currency"), args.get("to_currency")), 1.0)},
    "translate_text": lambda args: {"translated_text": TRANSLATE_DATA.get((args.get("text"), args.get("target_language")), args.get("text", ""))},
}

# ======== 工具调用解析与执行 ========
def parse_tool_calls(text):
    '''calls
    将tool_call里面的内容全部拿出来
    '''
    calls = []
    # 用正则 <tool_call>(.*?)</tool_call> 扫描模型生成文本，获取工具调用中的内容
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try:
            calls.append(json.loads(m.strip()))
        except:
            pass
    return calls

def execute_tool(name, args):
    '''

    name: rollout engine中tools
    args:

    '''
    fn = MOCK_RESULTS.get(name) # 到MOCK中获取名称
    if not fn: return None
    try:
        # 超时一秒使用传入的参数执行该tool函数
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(1)
        return fn(args)
    except:
        return None
    finally:
        try: signal.alarm(0)
        except: pass

def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    '''
    使用rollout_engine生成<tool call>token 根据tool call调用本地方法获取tool result，补充messages为
    原先对话prompt，和新生成的<tool call>和tool result组成的新信息做后续处理。
    '''
    all_outputs = []
    prompt_ids = None
    response_ids = []
    response_mask = []
    response_old_logps = []
    final_context = ""
    unfinished = False
    open_thinking = random.random() < thinking_ratio

    for turn in range(max_turns):
        '''
        第一步：
        一开始的messages格式如下
        messages = [
                  {"role": "system", "content": "# Tools ...", "tools": "[...]"},
                  ...长对话上下文
                  {"role": "user",   "content": "帮我算 256 * 37"}
                ]
        需要将messages进行tokenizer化之后，送入rollout采样答案
        '''
        # 所有上下文（包括前面多轮的问答，system中含有的tools，思考过程）打上聊天模板
        context = tokenizer.apply_chat_template(messages, tokenizer=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
        # 进行tokenizer
        inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)
        context_ids = inputs["input_ids"][0].tolist()
        # 把完整的上下文作为prompt
        if prompt_ids is None:
            prompt_ids = context_ids
        # 将上下文输入到rollout engine，重新输出rollout
        rollout_result = rollout_engine.rollout(
            prompt_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )

        # 获取输出文本的文字编号(ids)的和对数概率(logps)
        new_ids = rollout_result.completion_ids[0].tolist()
        new_logps = rollout_result.per_token_logps[0].tolist()
        if len(new_ids) != (new_logps):
            Logger(f"rollout token/logprob length mismatch: {len(new_ids)} vs {len(new_logps)}")
        pairs = [(t, lp) for t, lp in zip(new_ids, new_logps) if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
        new_ids = [t for t, _ in pairs]
        new_logps = [lp for _, lp in pairs]

        # 获取回答的文字，输出文字ids，mask以及logps
        new_text = rollout_result.completions[0]
        all_outputs.append(new_text)
        response_ids.extend(new_ids)
        response_mask.extend([1] * len(new_ids)) #对于response部分，这时候只有<tool_call>部分，打上掩码1，表示这部分后续需要进行loss的计算
        response_old_logps.extend(new_logps)

        final_context = context + new_text # 将上下文和回答进行拼接
        calls = parse_tool_calls(new_text) # 获取回答中的tool call内容
        if not calls:
            '''
            重点！如果rollout输出中没有包含tool call，直接结束处理！
            如果对话过程中模型觉得要进行多次tools call就会一直进入下一个turn，直接max turn达到或者返回输出不调用tools call

            '''
            break

        unfinished == turn == max_turns - 1

        messages.append({"role": "assistant", "content": new_text}) # 将rollout engine回答的结果添加到message
        '''     
        第二步：
        获取rollout中生成回答的<tool_call>(.*?)</tool_call>部分并将其加入到messages当中
        
        由于只会处理会调用tools call的信息，因此assistant本轮生成的内容必然是工具调用
        此时messages的格式如下：
        [
          {"role": "system", "content": "# Tools ..."},
          {"role": "user",   "content": "帮我算 256 * 37"},
          {"role": "assistant", "content": '<tool_call>{"name": "calculate_math", "arguments": {"expression": "256 * 37"}}</tool_call>'},  # 新增
        ]
        '''
        # 解析回答中的工具调用
        for call in calls:
            # 获取工具名称和参数字典
            name, raw = call.get("name", ""), call.get("arguments", {}) # 获取tool call里面工具名称和参数
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except:
                    raw = {}
            result = execute_tool(name, raw)
            if result:
                result_str = json.dump(result, ensure_ascii=False)[:2048] # 防止天文数字撑爆tokenizer
            else:
                result_str = '{"error": "tool not found"}'
            messages.append({"role": "tool", "content": result_str})
            ''' 
            第三步：
            获取tool_call中的函数以及对应参数，到MOCK_RESULT中得到（获取）工具执行的结果，并且将结果添加到messages当中
            
            此时messages的格式应该
            messages = [
                      {"role": "system",    "content": "# Tools ..."},
                      {"role": "user",      "content": "帮我算 256 * 37"},
                      {"role": "assistant", "content": '<tool_call>{"name": "calculate_math", "arguments": {"expression": "256 * 37"}}</tool_call>'},  ← 就是 new_text
                      {"role": "tool",      "content": '{"result": "9472"}'}      ← 工具执行结果
                      ]
                      
            第四步：
            对除去prompt和之前rollout engine生成的<tool_call>token外，新的tool result token部分打上0掩码（表示不需要loss计算部分）,因此这部分的logps也为0.0
            如果还没到max turn，且还需要继续调用工具，则将当前的message信息作为上下文，重复送入rollout采样的步骤，直到满足两者条件之一。
            
            这引出了agentc RL中的核心观点，这个阶段的训练是为了优化模型中“决定调用什么工具、参数怎么写、看到工具结果后怎么回答。”这部分的结果，关键在于工具调用和参数是否对上，是否成功调用工具。
            '''
        observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
        observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        current_len = len(prompt_ids) + len(response_ids)
        obs_delta = observe_ids[current_len:] # 截断后 obs_delta 只包含“工具结果渲染后新增的 token”。
        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta))
        response_old_logps.extend([0.0] * len(obs_delta))
        final_context = observe_context

    final_output = all_outputs[-1] if all_outputs else "" # 只会取最后一轮的结果，即tool call的输出结果，而不包含中间<tool call>的结果
    prompt_ids = prompt_ids or []
    return final_output, final_context, prompt_ids, response_ids, response_mask, response_old_logps, list(all_outputs), unfinished


def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, num_gen, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    '''
    对每一batch的messages多次采样生成含有tool call和tool result的完整messages组
    '''
    all_completions = []
    all_contexts = []
    all_prompt_ids = []
    all_response_ids = []
    all_response_masks = []
    all_response_old_logps = []
    all_turn_outputs = []
    all_unfinished = []

    for messages, tools in zip(messages_batch, tools_batch):
        # 多次采样，多次生成数据
        for _ in range(num_gen):
            msgs_copy = [dict(m) for m in messages]
            completion, context, prompt_ids, response_ids, response_mask, response_old_logps, turn_outputs, unfinished = rollout_single(
            rollout_engine, tokenizer, msgs_copy, tools, max_turns, max_new_tokens, thinking_ratio, device)
            all_completions.append(completion)
            all_contexts.append(context)
            all_prompt_ids.append(prompt_ids)
            all_response_ids.append(response_ids)
            all_response_masks.append(response_mask)
            all_response_old_logps.append(response_old_logps)
            all_turn_outputs.append(turn_outputs)
            all_unfinished.append(unfinished)
    return all_completions, all_contexts, all_prompt_ids, all_response_ids, all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished


def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen, reward_model=None, device="cuda", turn_outputs_batch=None, unfinished_batch=None):
    rewards = torch.zeros(len(completions), device=device)
    for idx, response in enumerate(completions):
        reward, answer = 0.0, response
        sample_idx = idx // num_gen
        tools = tools_batch[sample_idx]
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch is not None else [response]
        unfinished = unfinished_batch[idx] if unfinished_batch is not None else False
        turn_answers = [turn.split('</think>', 1)[-1].strip() if '</think>' in turn else turn.strip() for turn in
                        turn_outputs]
        answer = turn_answers[-1] if turn_answers else response.strip()
        valid_names = {t['function']['name'] for t in tools} if tools else set()
        tool_calls = []

def rl_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model=None, start_step=0, wandb=None, use_sglang=False):
    '''
    agentic rl训练主逻辑
    '''

    last_step = start_step
    for step, batch in enumerate(loader, start=start_step + 1):
        messages_batch = batch['messages']
        tools_batch = batch['tools']
        gt_batch = batch['gt']
        last_step = step

        with torch.no_grad():

            '''
            rollout采样回答空缺问题的完整messages
            
            输出参数含义如下:
            completions: 每条轨迹最后一次模型生成的文本,不包含中间<tool call>的结果,而是结合tool call回答最终结果的token
            contexts: 这条轨迹结束时的完整文本上下文（经过tokenizer模板化后）
            prompt_ids_batch：prompt的token id
            response_ids_batch: rollout生成回答token id，它不仅包括模型生成的 token，还包括工具结果以及多轮上下文新增的 token。
            response_masks_batch：对于rollout生成的token的mask
            response_old_logps_batch：rollout 时，旧策略模型生成每个 token 的 log probability。
            turn_outputs_batch：一条轨迹中每一轮模型实际生成的文本。包含中间模型的<tool call>部分
            unfinished_batch： 这条轨迹是否在最大轮数限制内没有正常结束。超过max turn后依然需要调用工具的标签
            '''
            completions, contexts, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch, turn_outputs_batch, unfinished_batch = rollout_batch(
                rollout_engine, tokenizer, messages_batch, tools_batch, args.num_generations, max_turns=3,
                max_new_tokens=args.max_gen_len, thinking_ratio=args.thinking_ratio, device=args.device)

            prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, tools=t) for m, t in zip(messages_batch, tools_batch)]
            packed_samples = []
            for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
                ids = p + r # prompt + <tool call>和response部分
                mask = [0] * len(p) + m # 前prompt为全0， <tool call>部分为1， tool_result部分为0
                old_logps = [0.0] * max(len(p) - 1, 0) + old_lp # 前prompt为全0.0， <tool call>部分为rollout的logps， tool_result部分为0.0

                if len(ids) > args.max_total_len:
                    ids = ids[-args.max_total_len:]
                    mask = mask[-args.max_total_len:]
                    old_logps = old_logps[-args.max_total_len:]

                prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask)) # 将mask中为1的部分作为prompt长度
                # prompt + <tool call>和response部分ids,
                packed_samples.append((ids, mask, prompt_len, old_logps))
            seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed_samples], device=args.device)
            max_len = seq_lens.max().item()
            # 补齐padding到最长长度
            input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _, _ in packed_samples], device=args.device)
            prompt_len = torch.tensor([prompt_len for _, _, prompt_len, _ in packed_samples], device=args.device)
            # response mask需要补齐到最长长度
            full_response_masks = torch.tensor([mask + [0] * (max_len - len(mask)) for _, _, _, old_logps in packed_samples], device=args.device, dtype=torch.float32)
            # logps需要补齐到最长长度 -1 ， 满足预测长度需求
            old_per_token_logps = torch.tensor(
                [old_logps + [0.0] * ((max_len - 1) - len(old_logps)) for _, _, _, old_logps in packed_samples],
                device=args.device, dtype=torch.float32)

            full_mask = (input_ids != tokenizer.pad_token_id).long()

            # 计算reward
            rewards = calculate_rewards(prompts, completions, gt_batch, tools_batch, args.num_generations, reward_model, device=args.device, turn_outputs_batch=turn_outputs_batch, unfinished_batch=unfinished_batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Agent RL")
    parser.add_argument("--save_dir", type=str, default=f"{BASE_DIR}/train_result", help="模型保存目录")
    parser.add_argument('--save_weight', default='agent_rl', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数") # macOS 多进程 DataLoader 可能增加内存和进程启动开销。
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=768, help="生成的最大长度")
    parser.add_argument("--max_total_len", type=int, default=2500, help="训练侧最终总长度上界")
    parser.add_argument("--data_path", type=str, default=f"{BASE_DIR}/dataset/agent_rl.jsonl", help="预训练数据路径")
    parser.add_argument("--num_generations", type=int, default=4, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Agent-RL", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.1, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # 1、初始化环境和随机种子
    setup_seed(42)
    # 如果有分布式训练可以继续追加分布式训练的环境配置

    # 2. 配置目录、模型参数、检查ckp
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir=CHECKPOINT_DIR) if args.from_resume == 1 else None

    # 3. 设置混合精度
    device_type = "mps" if "mps" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        torch.autocast(device_type=device_type, dtype=dtype, enabled=device_type == "mps")
        if device_type == "mps"
        else nullcontext()
    )

    # 4. 配wandb
    wandb = None
    if args.use_wandb:
        import swanlab as wandb

        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 5、初始化训练模型的分词器
    model, tokenizer = init_model(lm_config=lm_config, from_weight=args.from_weight,
                                  tokenizer_path=BASE_DIR / 'model', save_dir=args.save_dir,
                                  device=args.device)
    # 6、初始化参考模型
    base_model = args.from_weight
    ref_model, _ = init_model(lm_config=lm_config, from_weight=base_model, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    # 加载reward模型
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)

    # 7、初始化Rollout引擎
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )

    # 8、数据集和优化器
    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 以以下形式采样数据
    def collate_fn(batch):
        return {'message': [b['message'] for b in batch], 'tools': [b['tools'] for b in batch], 'gt': [b['gt'] for b in batch]}
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # 9、checkpoint恢复训练
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        # scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # 10、torch编译
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch compile enable')
        # 如果需要分布式训练需要将模型设置为分布式训练

    # 11、开始训练
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0

        base_sample = train_sampler or indices
        base_batch_sampler = SkipBatchSampler(
            BatchSampler(base_sample, batch_size=args.batch_size, drop_last=False),
            skip_batches=skip)
        loader = DataLoader(train_ds, batch_sampler=base_batch_sampler,
                            collate_fn=train_ds.collate_fn,
                            num_workers=args.num_workers, pin_memory=False)

        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            rl_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb,
                             use_sglang= (args.rollout_engine == 'sglang'))
        else:
            rl_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model,reward_model, 0, wandb,
                             use_sglang=(args.rollout_engine == 'sglang'))



