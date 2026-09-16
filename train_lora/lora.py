'''
后训练的lora具体训练过程
'''
import argparse
import torch
import time
import os
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from dataset.lm_dataset import SFTDataset
from trainer_utils import get_lr, Logger, lm_checkpoint, setup_seed, init_model
from model.model_minimind import MiniMindConfig, MiniMindModel
from model.model_lora import apply_lora, save_lora
from accelerate.data_loader import SkipBatchSampler


def train_epoch(epoch, loader, iters, start_step = 0, wandb = None):
    '''
    模型lora训练主逻辑
    '''
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start = start_step + 1):
        # 将训练数据以及标签放到计算设备上
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        # 学习率动态调整
        lr = get_lr(current_step = epoch * iters + step, total_steps = args.epochs * iters,
                    lr = args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 向前传播loss计算
        with autocast_ctx:
            res = model(input_ids, labels=labels) # 这里得到的是MiniMindForCausalLM类的forward结果
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        # 反向传播
        scaler.scale(loss).backward() # 使用 GradScaler 放大 loss 后进行反向传播，防止半精度（FP16）下的梯度下溢出。
        # 梯度累加控制
            # 当累加到一定的步数的时候，对优化器进行一个真正的参数更新
        if epoch % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_chip) # 梯度裁剪放置梯度爆炸

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True) # 更新完参数后梯度重新设置为零。

        # 日志报告loss情况
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters):
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}{moe_suffix}.pth'
            save_lora(model, lora_save_path)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
        del input_ids, labels, res, loss

        # 尾部梯度更新
        # 剩余数据不足够进行一次梯度更新的时候，将最后一组数据的结果作为最后的更新结果。
        if last_step > start_step and last_step % args.accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind SFT-LoRA")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='lora', type=str, help="保存权重的前缀名")
    parser.add_argument("--lora_name", type=str, default="lora_medical",help="LoRA权重名称(如lora_identity/lora_medical等)")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/lora_medical.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb项目名")
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
        wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 5.初始化模型和lora相关应用
    model, tokenizer = init_model(lm_config=lm_config, from_weight=args.from_weight, device=args.device)
    apply_lora(model)

        # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")

        # 冻结非lora层
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    # 6.定义数据集和优化器
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None # 如果是分布式训练需要分布式采样训练数据
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr = args.learning_rate)

    # 7、checkpoint恢复训练
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # 8、torch编译
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch compile enable')
        # 如果需要分布式训练需要将模型设置为分布式训练

    # 9、开始训练
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

    # 9、清理训练进程
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
