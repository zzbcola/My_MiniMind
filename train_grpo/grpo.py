import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
sys.path.insert(0, str(BASE_DIR))

import datasets
import argparse
import math
import re
import os
import time
import gc
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate.data_loader import SkipBatchSampler
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer_utils import get_lr, Logger, lm_checkpoint, setup_seed, init_model, LMForRewardModel
from rollout_engine import create_rollout_engine

def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    '''
    如何针对回答计算rewards
    '''

    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts) # prompts [Batch_size, seq_len]

        for i in range(batch_size):
            for j in range(args.num_generations):
                # 获取每个prompt对应的恢复id
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # rewards规则
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>" # tokenizer后对应的开始和结束模板
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches] # 获取上下文对话中的内容
                answer = response
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5 # 对于生成长度的加扣分机制

                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5 # 对于思考长度长度的加扣分机制
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25 # 只思考一次加分，否则扣分
                    answer = answer_content.strip()
                rewards[response_idx] -= rep_penalty(answer) # 对重复回答惩罚

                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score) # 将每个回答的得分都计入在内

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device) # 转化为张量 [batch_size * num_generation, 1]
        rewards += reward_model_scores



def GRPO_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None, use_sglang=False):
    '''
    训练流程大致如下：
    1. rollout 生成回答
    2. 计算 old_logp
    3. 当前 policy model forward，计算 current_logp
    4. reference model forward，计算 ref_logp
    5. 根据它们构造 loss
    6. loss.backward()
    7. optimizer.step() 更新 policy 权重
    8. 下一轮使用更新后的 policy
    '''

    start_time = time.time()
    last_step = start_step
    for step, batch in enumerate(loader, start=start_step + 1):
        last_step = step
        # loader为RLAIFDataset的输出结果
        prompts = batch['prompt']
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device) # 将prompt进行左侧填充到统一序列长度
        # 只取最后的max_seq_len长度
        if args.max_seq_len:
            prompt_inputs['input_ids'] = prompt_inputs['input_ids'][:, -args.max_seq_len]
            prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'][:, -args.max_seq_len]

        # 获取policy模型的输出结果
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations, # 多次生成答案的次数
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs = rollout_result.output_ids # prompt+回答的ids
        completion_ids = rollout_result.completion_ids # 回答部分ids
        completions = rollout_result.completions # 回答部分的文字
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach() # 生成token的对数概率 [Batch_size, seq_len, 1]
        prompt_lens = rollout_result.prompt_lens.to(args.device) # prompt长度
        full_mask = (outputs != tokenizer.pad_token_id).long()
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)

        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)  # 每个回答的得分计算rewards，形状为[B*num_gen]

        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model # 模型权重
        with autocast_ctx:
            # 等待后面loss.backward()之后就会更新参数，后续产生的per_token_logps为更新模型参数之后的输出token的对数概率
            res = model_unwrapped(outputs, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            # 计算当前模型policy model在回答中生成 token 上的 对数概率
            # 形状为[Batch_size * num_generations, completion_length]
            per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        # 参考模型中在回答中生成 token 上的 对数概率
        with torch.no_grad():
            ref_per_token_logps = F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        # 定期打印 prompt、每个生成的 completion 及其 reward，方便调试
        if args.debug_mode and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger('='*100)

        # 计算policy模型的回答优势，用于方便后续梯度传播
        grouped_rewards = rewards.view(-1, args.num_generations)  # [Batch_size, num_gen]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen] 对所有分数求平均
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # [B*num_gen] 对所有分数求标准差
        advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_gen] 优势计算公式，得到policy回答的整体优势

        # 对回答部分的eos后（结束部分后）padding的部分token不参与计算
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool() # 获取policy模型回答部分的mask bool值
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask  # [B*num_gen, completion_len]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        # 是一个01张量，用于记录completion中哪些应该被忽略。
        completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()  # [B*num_gen, completion_len]

        # 对参考模型和策略模型进行惩罚，防止多轮训练后模型回答严重偏离原sft模型
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, completion_len] KL散度惩罚
        # 当前策略生成某个token的概率与旧策略生成该 token 的概率之比
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*num_gen, completion_len]

        # 计算策略损失
        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon) # 限制所有的张量元素在1 - args.epsilon 到 1 + args.epsilon，防止策略过于激进
            per_token_loss1 = ratio * advantages.unsqueeze(1) # 放大优势 减少劣势，可以理解为调整当前策略对token的偏好，使得根据prompt生成回答时候对于这些token生成的概率更高
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1) # 截断过大优势，防止策略过于激进
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        # 最后的策略loss必须要乘completion_mask这个01张量，放置对不必要的部分也计算loss
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
        loss.backward()


        # 日志报告loss情况
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            policy_loss_val = loss.item() * args.accumulation_steps
            avg_reward_val = rewards.mean().item() # 多个回答的平均reward，反应总体的reward情况
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(completion_mask.sum().item(), 1)
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            current_lr = optimizer.param_groups[0]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}, '
                   f"epoch_time: {eta_min}")
        if wandb:
            wandb.log({
                "reward": avg_reward_val,
                "kl_ref": kl_ref_val,
                "advantages_std": advantages_std_val,
                "advantages_mean": advantages_mean_val,
                "policy_loss": policy_loss_val,
                "avg_response_len": avg_len_val,
                "learning_rate": current_lr,
                "epoch_time": eta_min
            })

        # 模型到一定轮数进行保存
        if (step % args.save_interval == 0 or step == iters):
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, epoch=epoch, step=step, wandb=wandb, save_dir=CHECKPOINT_DIR)
            model.train()
            del state_dict

        # 让rollout中的policy模型更新为参数更新后的policy模型
        if step % args.save_interval == 0 or step == iters:
            rollout_engine.update_policy(model)

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask, completion_pad_mask, prompt_lens, logp_pos

    # 尾部梯度更新
    # 剩余数据不足够进行一次梯度更新的时候，将最后一组数据的结果作为最后的更新结果。
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        # scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # scaler.step(optimizer)
        # scaler.update()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO")
    parser.add_argument("--save_dir", type=str, default=f"{BASE_DIR}/train_result", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数") # macOS 多进程 DataLoader 可能增加内存和进程启动开销。
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default=f"{BASE_DIR}/dataset/rlaif.jsonl", help="预训练数据路径")
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default=f"{BASE_DIR}/internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")

    # # 加速训练配置
    # parser.add_argument("--packing", default=1, type=int, choices=[0, 1], help="是否启用文档打包（0=不打包，使用原数据集）")
    # parser.add_argument("--length_grouped", default=0, type=int, choices=[0, 1], help="是否启用长度分桶（packing=0 时生效）")
    # parser.add_argument("--chunk_batches", type=int, default=64, help="长度分桶时每组排序的 batch 数，越大越省 padding 但越打乱全局次序")
    # parser.add_argument("--rebuild_cache", default=0, type=int, choices=[0, 1], help="是否强制重建 token 缓存（改了 max_seq_len 或数据后需要）")

    args = parser.parse_args()
    # 1、初始化环境和随机种子
    setup_seed(42)
    # 如果有分布式训练可以继续追加分布式训练的环境配置

    # 2. 配置目录、模型参数、检查ckp
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=CHECKPOINT_DIR) if args.from_resume==1 else None

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

    # 5. 定义模型、数据、优化器
    base_model = args.from_weight
    model, tokenizer = init_model(lm_config=lm_config, from_weight=args.from_weight,
                                  tokenizer_path=BASE_DIR / 'model', save_dir=args.save_dir,
                                  device=args.device)
    # 6、初始化参考模型
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

    # 8、数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
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
            GRPO_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, start_step, wandb,
                             use_sglang= (args.rollout_engine == 'sglang'))
        else:
            GRPO_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, 0, wandb,
                             use_sglang=(args.rollout_engine == 'sglang'))

