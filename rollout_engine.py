'''
使用sglang启动大模型

'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import requests
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer

# 计算每个token的对数概率
def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    # n_keep：回答长度
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    # 将输出的logits丢弃最后一个位置，使得logits与input_ids对齐
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach(), clone() if ids_row.is_inference() else ids_row
        # 对每个预测token的概率进行softmax之后再执行log，而选取词的索引用ids_row的最后一维来选取
        per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1))
    return torch.stack(per_token_logps)


with torch.no_grad():
    ref_per_token_logps = F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1).gather(
        2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
# ===== Rollout 结果 =====
@dataclass
class RolloutResult:
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: List[str]
    prompt_lens: Tensor
    completion_mask: Tensor


# ===== Rollout 引擎抽象基类 =====
class RolloutEngine(ABC):
    tokenizer = None

    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int,
                temperature: float = 0.8) -> RolloutResult:
        pass

    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        pass


# 使用pytorch原生进行推理
class TorchRolloutEngine(RolloutEngine):
    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()

        with torch.no_grad(), ctx():
            # 1、根据prompt生成回答的ids, 按照num_generations次数多次生成（同题多答）
            output_ids = model.generate(
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                max_new_tokens=max_new_tokens,
                do_sample=True, # 随机采样搜索，保证生成的答案不一样
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            ).clone()  # [Batch_size*num_gen, P+R] 经过generation方法已经自动将所有序列补齐到最长的实际生成长度(P + R)。

            # 2、 分离prompt_ids和completion_ids， 只获取除去prompt后模型自己生成的那部分
            prompt_len = prompt_ids.size(1)
            completion_ids = output_ids[:, prompt_len:] #

            full_mask = (output_ids != self.tokenizer.pad_token_id).long()
            # 3、获取每个token的预测对数概率
            per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1), attention_mask=full_mask)
        # 4、直接将结果解码成为文字（skip_special_tokens）输出
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions,
                             prompt_ids.new_full((output_ids.size(0),), prompt_len),
                             attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)))

    def update_policy(self, model: torch.nn.Module):
        # 更新权重后直接使用
        self.policy_model = model


# 使用SGLang HTTP API 推理引擎
class SGLangRolloutEngine(RolloutEngine):
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.http = requests

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        '''
        将prompt送到sglang的模型（policy model）请求，返回回答的ids和logprobs，补齐统一长度后输出RolloutResult类
        '''

        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):
            # 只获取没有被掩码的部分，前num_generations作为最终的input_ids
            valid_ids = ids[mask.bool()].tolist()
            input_ids_list.append(valid_ids)
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)] # 只取前num_generations作为

        payload = {
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            },
            "return_logprob": True,
        }
        # 向sglang服务器发送服务请求
        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status() # 请求状态
        results = resp.json()
        if not isinstance(results, list):
            results = [results]

        all_output_ids, all_completion_ids, all_logprobs = [], [], []
        completions = []
        for i, result in enumerate(results):
            # 获取reward模型返回的completion_ids和token对数概率
            meta = result.get("meta_info", {})
            completion_ids = meta.get("output_ids", result.get("output_ids", [])) # 只有回答的部分
            raw_logprobs = meta.get("output_token_logprobs", [])

            # 输出的item提取出对数概率logprobs,添加到列表当中
            logprobs = []
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])
                elif isinstance(item, (int, float)):
                    logprobs.append(item)

            # 如果返回每个token的对数概率长度与回答token的ids长度不一致的时候，做以下处理
            if len(logprobs) < len(completion_ids):
                logprobs = [0.0] * (len(completion_ids) - len(logprobs)) + logprobs # 补齐长度到completion_ids的长度
            elif len(logprobs) > len(completion_ids):
                logprobs = logprobs[-len(completion_ids):] if completion_ids else [] # 只取最后completion_ids的长度

            full_output = prompt + completion_ids
            all_output_ids.append(full_output) # prompt + 回答部分的ids
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs) # 回答部分的token对数概率
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True)) # 将回答部分的token转化为文字添加到completions

        # 获取回答的最长长度
        device = prompt_ids.device
        max_comp_len = max(1, max(len(ids) for ids in all_completion_ids))
        max_out_len = max(len(ids) for ids in all_input_ids) + max_comp_len

        # 将所有序列,右侧补 pad 到统一长度，按照所有回答的最长长度进行padding
        def pad_to_tensor(seqs, max_len, pad_val=0):
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)

        pad_id = self.tokenizer.pad_token_id
        # 输出RolloutResult格式，保证所有的回答都已经补齐到统一长度
        return RolloutResult(
            output_ids=pad_to_tensor(all_output_ids, max_out_len, pad_val=pad_id),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len, pad_val=pad_id),
            per_token_logps=pad_to_tensor(all_logprobs, max_comp_len, pad_val=0.0),
            completions=completions,
            prompt_lens=torch.tensor([len(ids) for ids in all_input_ids], device=device),
            completion_mask=torch.tensor(
                [[1] * len(ids) + [0] * (max_comp_len - len(ids)) for ids in all_completion_ids], device=device),
        )

    def update_policy(self, model: torch.nn.Module):
        '''
        把更新完的policy权重同步给加载在sgalang中的policy模型
        '''
        ok = True
        # 获取模型权重
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)
        # 是训练进程和 SGLang 服务器都能访问的地址
        abs_path = os.path.abspath(self.shared_ckpt_path)

        # 转换权重格式到cpu然后保存
        state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
        unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
        self.tokenizer.save_pretrained(abs_path)

        # 重新加载新的权重
        resp = self.http.post(f"{self.base_url}/update_weights_from_disk", json={"model_path": abs_path},
                              timeout=self.timeout)
        if resp.status_code != 200: print(
            f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")
        ok = resp.status_code == 200

        # # 多GPU时候加载权重处理
        # if dist.is_initialized():
        #     ok_t = torch.tensor(int(ok), device=next(model.parameters()).device)
        #     dist.broadcast(ok_t, src=0);
        #     dist.barrier();
        #     ok = bool(ok_t.item())
        if not ok: raise RuntimeError("SGLang update_policy failed")
        return ok

    def flush_cache(self) -> bool:
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200

    def health(self) -> bool:
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False

def create_rollout_engine(
    engine_type: str = "torch",
    policy_model: torch.nn.Module = None,
    tokenizer = None,
    device: str = "cuda",
    autocast_ctx = None,
    sglang_base_url: str = None,
    sglang_model_path: str = None,
    sglang_shared_path: str = None,
) -> RolloutEngine:
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")
