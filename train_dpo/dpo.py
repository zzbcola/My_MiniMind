'''
训练dpo的主要流程
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.distributions.utils import logits_to_probs
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import DPODataset
from accelerate.data_loader import SkipBatchSampler
from trainer_utils import get_lr, Logger, lm_checkpoint, setup_seed, init_model


def logits_to_log_probs(logits, labels):
    '''
    对整句话每个token预测出来的概率矩阵，获取其labels真实标签对应的那个词的对数概率
    logist: [batch_size, seq_len, vocab_size] 模型对于整句话每个token预测出来的概率矩阵
    labels: [batch_size, seq_len] 真正的groundtruth token序列
    return: [batch_size, seq_len]

    举例：
    对于[1, 2, 3]的logits，真实标签为[[ 1,  2 ]]
    # 时间步 t=0 的打分        # 时间步 t=1 的打分
    [[[ 1.5,  0.5,  0.2 ],   [ 0.1,  2.0,  0.8 ]]]

    执行过程如下：
    先计算softmax
    [[[-0.52, -1.51, -1.66], [-2.30, -0.38, -1.51]]]
    再选取真实目标的token
    [[-1.51, -1.51]]
    '''
    log_probs = F.log_softmax(logits, dim=2) # 只对预测概率维度进行softmax
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    '''
    对整句话每个token预测出来的概率矩阵，获取其labels真实标签对应的那个词的对数概率
    ref_log_probs: [batch_size, seq_len]
    policy_log_probs: [batch_size, seq_len]
    return:

    '''
    # 首先对序列乘以mask，获取真正回答部分的概率预测
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 将chosen和rejected部分的数据分开
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    rejected_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    rejected_policy_log_probs = policy_log_probs[batch_size // 2:]

    # 计算对数比率差
    pi_logrations = chosen_policy_log_probs - rejected_policy_log_probs
    ref_logrations = chosen_ref_log_probs - rejected_ref_log_probs
    logits = pi_logrations - ref_logrations # 目的是为了策略模型更加偏好选择chosen答案（pi_logrations上升）
    loss = -F.logsigmoid(beta * logits) # 最后通过取结果的负数让梯度更新偏向pi_logrations上升 -> chosen_policy_log_probs上升
    return loss.mean()

def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step = 0, wandb = None, beta=0.1):
    '''
    模型dpo训练主逻辑
    '''
    start_time = time.time()
    last_step = start_step

    for step, batch in enumerate(loader, start=start_step + 1):
        last_step = step

        # 将dpo所准备好的数据加载到设备当中
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)

        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # 学习率动态调整
        lr = get_lr(current_step = epoch * iters + step, total_steps = args.epochs * iters,
                    lr = args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 向前传播loss计算
        with autocast_ctx:
            with torch.no_grad(): # 参考模型不需要更新参数，因此需要冻结
                ref_outputs = ref_model(x) # 用参考模型将数据跑一遍结果
                ref_logits = ref_outputs.logits
            # 获取模型生成下一个token的对数概率
            ref_log_probs = logits_to_log_probs(ref_logits, y) # [batch_size, seq_len]

            # 对策略模型也跑一遍结果
            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y) # [batch_size, seq_len]

            #计算策略模型和参考模型的log概率差值作为loss
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = dpo_loss_val + outputs.aux_loss
            loss = loss / args.accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()  # 使用 GradScaler 放大 loss 后进行反向传播，防止半精度（FP16）下的梯度下溢出。
        # 梯度累加控制
        # 当累加到一定的步数的时候，对优化器进行一个真正的参数更新
        if epoch % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_chip)  # 梯度裁剪放置梯度爆炸

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)  # 更新完参数后梯度重新设置为零。

        # 日志报告loss情况
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.aux_loss.item() if dpo_loss_val.aux_loss is not None else 0.0
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log(
                {"loss": current_loss, "current_dpo_loss": current_dpo_loss, "aux_loss": current_aux_loss,
                 "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters):
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss

        # 尾部梯度更新
        # 剩余数据不足够进行一次梯度更新的时候，将最后一组数据的结果作为最后的更新结果。
        if last_step > start_step and last_step % args.accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DPO")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--beta", default="0.15", type=float, help="DPO的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()


    # 1、初始化环境和随机种子
    setup_seed(42)
    # 如果有分布式训练可以继续追加分布式训练的环境配置

    # 2. 配置目录、模型参数、检查ckp
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # 3. 设置混合精度
    device_type = "mps" if "mps" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        # mps训练下不启用autocast
    autocast_ctx = nullcontext() if device_type in ("mps", "cpu") else torch.cuda.amp.autocast(dtype=dtype)

    # 4. 配wandb
    wandb = None
    if args.use_wandb:
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 5.初始化策略模型和优化器
    model, tokenizer = init_model(lm_config=lm_config, from_weight=args.from_weight, device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')

    # 初始化参考模型
    ref_model, _ = init_model(lm_config=lm_config, from_weight=args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')

    # 数据集和优化器设置
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 6、checkpoint恢复训练
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # 7、torch编译
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch compile enable')
        # 如果需要分布式训练需要将模型设置为分布式训练

    # 8、开始训练
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randerm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip) # 将采样的数据合成一个个batch方便后续训练
        loader = DataLoader(train_ds, batch_sampler = batch_sampler, num_workers=args.num_workers, pin_memory = True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)