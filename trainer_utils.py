import math
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from fontTools.misc.timeTools import epoch_diff
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, AutoModel

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def get_model_params(model, config):
    '''
    获取模型参数量和moe的细节参数量
    '''
    total = sum(p.numel() for p in model.parameters()) / 1e6 # 获取模型总参数量
    # moe相关
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0)) # 路由总专家数
    n_active = getattr(config, 'num_experts_per_tok', 0) # 激活的专家数
    n_shared = getattr(config, 'n_shared_experts', 0) # 共享专家数
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6 # 单个专家参数量
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6 # 共享专家数量
    base = total - (expert * n_routed) - (shared_expert * n_shared) # 除去moe后的基础参数两
    active = base + (expert * n_active) + (shared_expert * n_shared) # 激活后的参数量
    # 报告参数情况
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')

def setup_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_lr(current_step:int, total_steps:int, lr:float):
    '''
    通过当前步数和总步数计算逐渐变化的学习率
    '''
    return lr * (0.1 + 0.45 * math.cos(math.pi * (current_step / total_steps)))

def empty_device_cache(device:str):
    '''
    针对cuda和mps设备清空闲置内存
    '''
    if device == 'mps':
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    elif device == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def Logger(content):
    print(content)

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None,
                  epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    # 节点保存设置
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    # 保存模式
    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model) # 获取模型中的_orig_mod属性
        state_dict = raw_model.state_dict() # 获取其中的字典
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()} # 将模型的v转化为半精并且存储在cpu中
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path) # 将新的模型替换原来的checkpoints
        wandb_id = None
        # 获取wandb的id，针对新旧方法, wandb是一种类似于线上的tensorboard，用于监管训练
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': 1, # mps训练的情况下无法多卡训练
            'wandb_id': wandb_id
        }

        # 检查其他参数中是否有state_dict方法，有的话需要将其也存入到resume_data里面,否则直接将值直接存入resume_data中
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + 'tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        empty_device_cache('mps')
    # 加载模式
    else:
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            # cuda中多卡步骤设计gpu数量变化才需要以下判断
            # saved_ws = ckp_data.get('world_size', 1)
            # current_ws = 1
            # if saved_ws != current_ws:
            #     ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
            #     Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None

def init_model(lm_config, from_weight='pretrain', tokenizer_path='./model',
               save_dir='./out', device='mps'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
    # 获取模型参数量，moe数量
    get_model_params(model, lm_config)

    # 训练参数量
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer

class LMForRewardModel:
    '''
    调用打分模型并且根据上下文和回答返回打分
    '''
    def __init__(self, model_path, device='mps', dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

        @torch.no_grad()
        def get_score(self, messages, response):
            # 获取数据集的完整回答上下文，以及最后一个问题
            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in message[:-1]])
            last_query = message[-1]['content'] if message else ""
            message_context = f"{history_text}\n以上是历史对话。我的新问题是:\n{last_query}" if history_text else last_query
            # 将上下文和新的回答组织成一段问答信息
            eval_message = [
                {"role": "user", "content": message_context},
                {"role": "assistant", "content": response}
            ]
            # 给评价模型进行打分
            score = self.model.get_score(self.tokenizer, eval_message)
            return max(min(score, 3.0), -3.0)