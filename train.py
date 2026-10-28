"""VLM 训练入口:冻结视觉编码器与 LLM 基座,仅微调 projector 与文本解码器 LoRA。

流程:
  1. 读配置("vlm" 建模型,"training" 配训练),自动创建输出目录
  2. 建数据(按 val_ratio 划分训练/验证集)与模型(冻结基座)
  3. 两阶段:前 stage1_epochs 轮冻结 LoRA 只训 projector(视觉冷启动),
     之后解冻 LoRA 联合微调;两阶段各有独立的 warmup+余弦调度
  4. 逐轮:训练 → 验证 → val 下降则保存最优参数 → 保存训练现场(供续训)
  5. val 连续 patience 轮不下降则早停(切换阶段时计数重置);
     projector.pt / lora.pt 始终为最优参数

用法: python train.py [--config configs/test.json] [--device cuda]
"""
import argparse
import logging
import math
import os
import random
import sys
import time

import numpy as np
import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader, Subset

# 数据/模型模块内部使用相对导入,必须以 vlm_train 包路径导入 → 父目录入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_train.models.vlm import VLM, VlmConfig
from vlm_train.data.dataset import MyDataset, VisionBatch
from vlm_train.utils import json_load, json_dump

CONFIG_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "test.json")


# ================================== 训练配置 ==================================
class TrainingConfig(BaseModel):
    """configs 中 "training" 字段"""
    # 数据
    data_path: str               # 样本 JSON(MyDataset 直接读取)
    val_ratio: float = 0.1       # 验证集占比(随机划分,只训不测)
    num_workers: int = 4
    seed: int = 42
    # 优化
    num_epochs: int = 5
    batch_size: int = 2
    lr: float = 2e-4             # 阶段二(projector + LoRA 联合)学习率
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03   # warmup 步数 / 所处阶段总步数
    grad_clip: float = 1.0
    amp: bool = True             # fp16 混合精度(V100 无 bf16 张量核心)
    # 两阶段:阶段一冻结 LoRA 只训 projector(视觉冷启动),阶段二解冻联合微调
    stage1_epochs: int = 5       # 阶段一轮数(计入 num_epochs),0 = 不分阶段
    stage1_lr: float = 1e-4      # 阶段一学习率
    # 早停
    patience: int = 3            # val 连续 N 轮不下降则早停,0 关闭
    # 输出(output_dir 下:train.log、checkpoint.pt(续训现场)、projector.pt + lora.pt(最优参数))
    output_dir: str
    log_steps: int = 10
    num_val_generate: int = 2    # 每轮验证后打印的生成样例数,0 关闭
    # 续训
    resume: bool = False         # 从 output_dir/checkpoint.pt 恢复现场继续训练


# ================================== 基础工具 ==================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir: str) -> logging.Logger:
    """日志同时写控制台与 output_dir/train.log"""
    logger = logging.getLogger("vlm-train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(),
                    logging.FileHandler(os.path.join(output_dir, "train.log"), encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="VLM 微调训练(projector + LoRA)")
    parser.add_argument("--config", default=CONFIG_DEFAULT, help="配置 JSON 路径")
    parser.add_argument("--device", default="cuda", help="如 cuda / cuda:0 / cpu")
    return parser.parse_args()


# ================================== 组件构建 ==================================
def build_model(vlm_cfg: dict, device: torch.device, logger) -> VLM:
    """构建 VLM;冻结视觉编码器与 LLM 基座,可训练参数仅剩 projector 与 LoRA。"""
    model = VLM(VlmConfig(**vlm_cfg)).to(device)
    model.freeze("vision_encoder")
    # LLM 基座参数已由 PEFT 冻结,但词表 resize 可能重置 requires_grad,统一再冻结一次非 LoRA 参数
    for name, p in model.text_decoder.named_parameters():
        if "lora_" not in name:
            p.requires_grad = False

    # 参数统计:确认可训练的只有 projector 与 text_decoder 中的 LoRA
    stats = {}
    for name, p in model.named_parameters():
        top = name.split(".", 1)[0]
        total, trainable = stats.get(top, (0, 0))
        if p.requires_grad:
            assert top == "projector" or "lora_" in name, f"意外可训练参数: {name}"
            trainable += p.numel()
        stats[top] = (total + p.numel(), trainable)
    for top, (total, trainable) in stats.items():
        logger.info(f"参数统计 {top:<15} 总数 {total / 1e6:8.2f}M | 可训练 {trainable / 1e6:8.2f}M")
    assert stats["vision_encoder"][1] == 0, "视觉编码器存在可训练参数"
    return model


def make_collate():
    """构造 collate:变长 tokens/coords 右侧零填充对齐为 batch,并给出有效位 mask"""
    def collate(batch):
        tokens_list, coords_list, prompts, targets = zip(*batch)
        B, n_max = len(tokens_list), max(t.shape[0] for t in tokens_list)
        tokens = torch.zeros(B, n_max, tokens_list[0].shape[1])
        coords = torch.zeros(B, n_max, 3, dtype=torch.long)
        mask = torch.zeros(B, n_max, dtype=torch.bool)
        for i, (t, c) in enumerate(zip(tokens_list, coords_list)):
            tokens[i, :t.shape[0]] = t
            coords[i, :c.shape[0]] = c
            mask[i, :t.shape[0]] = True
        return VisionBatch(tokens, coords, mask), list(prompts), list(targets)

    return collate


def to_device(img, device):
    """视觉 batch(tokens/coords/mask)整体搬至 device"""
    return VisionBatch(*(t.to(device, non_blocking=True) for t in img))


def make_loaders(cfg: TrainingConfig, logger):
    """同一份样本按 val_ratio 随机划分训练/验证集"""
    dataset = MyDataset(cfg.data_path)
    n_val = int(len(dataset) * cfg.val_ratio)
    perm = torch.randperm(len(dataset),
                          generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    train_set, val_set = Subset(dataset, perm[n_val:]), Subset(dataset, perm[:n_val])
    collate = make_collate()
    loader_kwargs = dict(num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=cfg.num_workers > 0, collate_fn=collate)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(cfg.seed),
                              **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, **loader_kwargs)
    logger.info(f"数据 {cfg.data_path} | 训练 {len(train_set)} | 验证 {len(val_set)} | "
                f"变长视觉 token(背景已去除)")
    return train_loader, val_loader


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """warmup + 余弦衰减"""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def stage_of(epoch: int, cfg: TrainingConfig) -> int:
    """epoch 所处训练阶段:1 = 仅 projector,2 = projector + LoRA"""
    return 1 if epoch <= cfg.stage1_epochs else 2


def enter_stage(model, stage, cfg: TrainingConfig, steps_per_epoch: int, logger):
    """进入指定阶段:阶段一冻结 LoRA(只训 projector),阶段二解冻联合;
    按该阶段轮数与学习率重建优化器/调度器。返回 (optimizer, scheduler, trainable_params)"""
    for name, p in model.text_decoder.named_parameters():
        if "lora_" in name:
            p.requires_grad = stage == 2
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    lr = cfg.stage1_lr if stage == 1 else cfg.lr
    n_epochs = cfg.stage1_epochs if stage == 1 else cfg.num_epochs - cfg.stage1_epochs
    total_steps = steps_per_epoch * n_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
    n_train = sum(p.numel() for p in trainable_params) / 1e6
    logger.info(f"阶段 {stage}({'仅 projector' if stage == 1 else 'projector + LoRA 联合'}) | "
                f"lr {lr:.2e} | 可训练 {n_train:.2f}M | "
                f"总步数 {total_steps}(warmup {warmup_steps})")
    return optimizer, scheduler, trainable_params


# ================================== 存档 / 续训 ==================================
def lora_state(model: VLM) -> dict:
    """文本解码器中已注入的 LoRA 参数"""
    return {k: v for k, v in model.text_decoder.model.state_dict().items() if "lora_" in k}


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, global_step, best, stage):
    """训练现场 = 可训练参数(projector + LoRA)+ 优化器/调度器/AMP/RNG/早停状态"""
    torch.save({
        "epoch": epoch, "global_step": global_step, "stage": stage, "best": best,
        "projector": model.projector.state_dict(), "lora": lora_state(model),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": (torch.get_rng_state(), random.getstate(), np.random.get_state()),
    }, path)


def load_checkpoint(path, model, scaler):
    """恢复模型权重/AMP/RNG/早停状态,返回 (start_epoch, global_step, best, ckpt)。
    优化器/调度器与阶段绑定,由 main 进入对应阶段时再决定是否载入。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.projector.load_state_dict(ckpt["projector"])
    model.text_decoder.model.load_state_dict(ckpt["lora"], strict=False)
    scaler.load_state_dict(ckpt["scaler"])
    torch_rng, py_rng, np_rng = ckpt["rng"]
    torch.set_rng_state(torch_rng)
    random.setstate(py_rng)
    np.random.set_state(np_rng)
    best = ckpt.get("best", {"best_val": float("inf"), "best_epoch": 0, "no_improve": 0})
    return ckpt["epoch"] + 1, ckpt["global_step"], best, ckpt


# ================================== 训练 / 验证 ==================================
def train_one_epoch(model, loader, optimizer, scheduler, scaler, trainable_params,
                    epoch, cfg, device, use_amp, logger, global_step):
    model.train()
    loss_sum, n_steps, t0 = 0.0, 0, time.time()
    for i, (img, prompt, target) in enumerate(loader):
        img = to_device(img, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            _, loss = model(img, prompt=prompt, target=target)
        if not torch.isfinite(loss):
            logger.warning(f"epoch {epoch} step {i}: loss 非有限值,跳过该 batch")
            continue
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1
        loss_sum, n_steps = loss_sum + loss.item(), n_steps + 1
        if global_step % cfg.log_steps == 0:
            speed = (i + 1) / (time.time() - t0)
            logger.info(f"epoch {epoch}/{cfg.num_epochs} | step {i + 1}/{len(loader)} | "
                        f"loss {loss_sum / n_steps:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                        f"{speed:.2f} it/s")
    return loss_sum / max(1, n_steps), global_step


@torch.no_grad()
def validate(model, loader, device, use_amp, cfg, logger) -> float:
    """验证集平均 loss,并生成少量样例观察效果"""
    model.eval()
    loss_sum, n_samples = 0.0, 0
    for img, prompt, target in loader:
        img = to_device(img, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            _, loss = model(img, prompt=prompt, target=target)
        if torch.isfinite(loss):
            loss_sum += loss.item() * img.size(0)
            n_samples += img.size(0)
    val_loss = loss_sum / n_samples if n_samples else float("inf")  # 全部非有限时按最差处理
    logger.info(f"验证 | loss {val_loss:.4f} | {n_samples} 样本")

    # 生成样例:取第一个验证 batch 的前几个样本
    k = min(cfg.num_val_generate, len(loader.dataset))
    if k > 0:
        img, prompt, target = next(iter(loader))
        sub = VisionBatch(img.tokens[:k], img.coords[:k], img.mask[:k])  # 前 k 个样本
        gen = model.generate(to_device(sub, device), prompt=prompt[:k], max_new_tokens=96)
        for pr, g, t in zip(prompt[:k], gen, target[:k]):
            logger.info(f"  [问] {pr}")
            logger.info(f"  [生成] {g[:200]}")
            logger.info(f"  [目标] {t[:200]}")
    return val_loss


# ================================== 主流程 ==================================
def main():
    # ---------- 1. 配置与输出目录 ----------
    args = parse_args()
    cfg_dict = json_load(args.config)
    cfg = TrainingConfig(**cfg_dict["training"])
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = setup_logger(cfg.output_dir)
    device = torch.device(args.device)
    use_amp = cfg.amp and device.type == "cuda"
    set_seed(cfg.seed)
    json_dump(cfg_dict, os.path.join(cfg.output_dir, "config_used.json"))  # 配置留档

    logger.info(f"配置 {args.config} | device {device} | AMP {use_amp} | 输出目录 {cfg.output_dir}")
    logger.info(f"训练设置: epochs {cfg.num_epochs} | patience {cfg.patience} | "
                f"batch {cfg.batch_size} | lr {cfg.lr} | "
                f"wd {cfg.weight_decay} | warmup {cfg.warmup_ratio} | clip {cfg.grad_clip}")

    # ---------- 2. 模型与数据 ----------
    model = build_model(cfg_dict["vlm"], device, logger)
    train_loader, val_loader = make_loaders(cfg, logger)

    # ---------- 3. AMP 与续训恢复(先于优化器:当前阶段由起始轮决定) ----------
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ckpt_path = os.path.join(cfg.output_dir, "checkpoint.pt")
    proj_path = os.path.join(cfg.output_dir, "projector.pt")
    lora_path = os.path.join(cfg.output_dir, "lora.pt")
    start_epoch, global_step, ckpt = 1, 0, None
    best = {"best_val": float("inf"), "best_epoch": 0, "no_improve": 0}
    if cfg.resume:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"resume=true 但不存在 {ckpt_path};全新训练请置 resume=false")
        start_epoch, global_step, best, ckpt = load_checkpoint(ckpt_path, model, scaler)
        logger.info(f"已从 checkpoint 恢复现场: 从 epoch {start_epoch} / step {global_step} 继续 | "
                    f"历史最优 val {best['best_val']:.4f}(epoch {best['best_epoch']})")
    elif os.path.exists(ckpt_path):
        logger.info("注意: 目录下已有 checkpoint.pt,本次全新训练将覆盖它")

    # ---------- 4. 逐轮:训练 → 验证 → 存档(两阶段) ----------
    optimizer = scheduler = trainable_params = cur_stage = None
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        stage = stage_of(epoch, cfg)
        if stage != cur_stage:  # 首轮或阶段切换:冻结/解冻 LoRA 并重建优化器
            if epoch > start_epoch:  # 运行途中切换(非本次起始轮)
                best["no_improve"] = 0  # 新阶段新起点,早停计数重新起算
                logger.info(f"=== 切换到阶段 {stage},早停计数重置 ===")
            optimizer, scheduler, trainable_params = enter_stage(
                model, stage, cfg, len(train_loader), logger)
            if ckpt is not None and ckpt.get("stage", 1) == stage:
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                logger.info("已载入 checkpoint 中本阶段的优化器/调度器状态")
            cur_stage, ckpt = stage, None

        t0 = time.time()
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, trainable_params,
            epoch, cfg, device, use_amp, logger, global_step)
        val_loss = validate(model, val_loader, device, use_amp, cfg, logger)

        # val 下降 → 保存最优参数;否则累计未下降轮数
        if val_loss < best["best_val"]:
            best = {"best_val": val_loss, "best_epoch": epoch, "no_improve": 0}
            torch.save(model.projector.state_dict(), proj_path)
            torch.save(lora_state(model), lora_path)
            logger.info(f"val 下降,最优参数已保存 | val {val_loss:.4f}")
        else:
            best["no_improve"] += 1

        save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler,
                        epoch, global_step, best, stage)
        logger.info(f"epoch {epoch} 完成 | train {train_loss:.4f} | val {val_loss:.4f} | "
                    f"最优 val {best['best_val']:.4f}(epoch {best['best_epoch']}) | "
                    f"耗时 {(time.time() - t0) / 60:.1f} min | 现场已保存")
        if 0 < cfg.patience <= best["no_improve"]:
            logger.info(f"val 连续 {best['no_improve']} 轮未下降,早停(patience={cfg.patience})")
            break

    # ---------- 5. 结束 ----------
    logger.info(f"训练结束 | 最优 val {best['best_val']:.4f}(epoch {best['best_epoch']})| "
                f"参数 {proj_path} 与 {lora_path}")


if __name__ == "__main__":
    main()
