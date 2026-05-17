# -*- coding: utf-8 -*-
"""
train.py
深度学习主导版
目标：
- 深度学习负责主要风格生成
- 传统法只是辅助构造训练 target
- 更适合论文写作：基于深度学习的图像风格转换
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

SEED = 42
DATA_ROOT = "dataset/train"
SAMPLE_DIR = "training_samples"
BEST_MODEL_PATH = "sketch_model_best.pth"
FINAL_MODEL_PATH = "sketch_model_final.pth"

ENABLE_ANOMALY_DETECTION = False


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

if ENABLE_ANOMALY_DETECTION:
    torch.autograd.set_detect_anomaly(True)


class SketchDataset(Dataset):
    def __init__(self, root_dir, mode="train", image_size=512, split_ratio=0.9):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, "input")
        self.target_dir = os.path.join(root_dir, "target")
        self.image_size = image_size
        self.mode = mode

        self.file_list = sorted([
            f for f in os.listdir(self.input_dir)
            if os.path.isfile(os.path.join(self.input_dir, f))
        ])

        if len(self.file_list) == 0:
            raise RuntimeError(f"未在 {self.input_dir} 中找到训练图片")

        split_idx = int(split_ratio * len(self.file_list))
        split_idx = max(1, min(split_idx, len(self.file_list) - 1))

        if mode == "train":
            self.file_list = self.file_list[:split_idx]
        else:
            self.file_list = self.file_list[split_idx:]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]

        input_path = os.path.join(self.input_dir, name)
        target_path = os.path.join(self.target_dir, name)

        input_img = Image.open(input_path).convert("RGB")
        target_img = Image.open(target_path).convert("L")

        input_img = TF.resize(input_img, [self.image_size, self.image_size])
        target_img = TF.resize(target_img, [self.image_size, self.image_size])

        if self.mode == "train":
            if random.random() < 0.5:
                input_img = TF.hflip(input_img)
                target_img = TF.hflip(target_img)

            if random.random() < 0.15:
                input_img = TF.vflip(input_img)
                target_img = TF.vflip(target_img)

            if random.random() < 0.15:
                angle = random.choice([90, 180, 270])
                input_img = TF.rotate(input_img, angle)
                target_img = TF.rotate(target_img, angle)

            brightness = random.uniform(0.92, 1.10)
            contrast = random.uniform(0.92, 1.10)
            saturation = random.uniform(0.95, 1.06)

            input_img = TF.adjust_brightness(input_img, brightness)
            input_img = TF.adjust_contrast(input_img, contrast)
            input_img = TF.adjust_saturation(input_img, saturation)

        input_tensor = TF.to_tensor(input_img)
        target_tensor = TF.to_tensor(target_img)

        return input_tensor, target_tensor, name


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

        self.out_relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.out_relu(out)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        mid = max(ch // 8, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(ch, mid, 1)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(mid, ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.conv1(w)
        w = self.relu(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        return x * w


class SketchUNetPlus(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            ResBlock(64),
            nn.ReLU(inplace=False),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            ResBlock(128),
            nn.ReLU(inplace=False),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            ResBlock(256),
            nn.ReLU(inplace=False),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            ResBlock(512),
            nn.ReLU(inplace=False),
        )

        self.attn3 = AttentionBlock(256)
        self.attn4 = AttentionBlock(512)

        self.dec4_up = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4_res = ResBlock(256)

        self.dec3_up = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3_res = ResBlock(128)

        self.dec2_up = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2_res = ResBlock(64)

        self.out = nn.Conv2d(64, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        d4 = self.dec4_up(self.attn4(x4))
        d4 = d4 + self.attn3(x3)
        d4 = self.dec4_res(d4)

        d3 = self.dec3_up(d4)
        d3 = d3 + x2
        d3 = self.dec3_res(d3)

        d2 = self.dec2_up(d3)
        d2 = d2 + x1
        d2 = self.dec2_res(d2)

        out = self.out(d2)
        out = self.sigmoid(out)
        return out


def edge_loss(pred, target, device):
    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=torch.float32,
        device=device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
        dtype=torch.float32,
        device=device
    ).view(1, 1, 3, 3)

    pred_ex = F.conv2d(pred, sobel_x, padding=1)
    pred_ey = F.conv2d(pred, sobel_y, padding=1)
    tar_ex = F.conv2d(target, sobel_x, padding=1)
    tar_ey = F.conv2d(target, sobel_y, padding=1)

    return F.l1_loss(pred_ex, tar_ex) + F.l1_loss(pred_ey, tar_ey)


def tone_loss(pred, target):
    pred_mean = pred.mean(dim=[2, 3], keepdim=True)
    tar_mean = target.mean(dim=[2, 3], keepdim=True)
    return F.l1_loss(pred_mean, tar_mean)


def charbonnier_loss(pred, target, eps=1e-3):
    diff = pred - target
    loss = torch.sqrt(diff * diff + eps * eps)
    return loss.mean()


def save_sample_images(model, loader, device, epoch, save_dir=SAMPLE_DIR, use_amp=True):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        batch = next(iter(loader))
        inputs, targets, _ = batch
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            outputs = model(inputs)

        cols = min(4, inputs.size(0))
        fig, axes = plt.subplots(3, cols, figsize=(4 * cols, 9))

        if cols == 1:
            axes = np.array(axes).reshape(3, 1)

        for i in range(cols):
            axes[0, i].imshow(inputs[i].detach().cpu().permute(1, 2, 0).numpy())
            axes[0, i].set_title(f"Input {i + 1}")
            axes[0, i].axis("off")

            axes[1, i].imshow(outputs[i, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[1, i].set_title(f"Output {i + 1}")
            axes[1, i].axis("off")

            axes[2, i].imshow(targets[i, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[2, i].set_title(f"Target {i + 1}")
            axes[2, i].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch:03d}.png"), dpi=120, bbox_inches="tight")
        plt.close()

    model.train()


def get_device_info():
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        return torch.device("cuda"), name, total_mem_gb
    return torch.device("cpu"), "CPU", 0.0


def suggest_num_workers():
    cpu_count = os.cpu_count() or 4
    return min(8, max(2, cpu_count // 2))


def train(
    num_epochs=40,
    batch_size=8,
    image_size=512,
    learning_rate=1e-4,
    weight_decay=1e-5,
    grad_clip=1.0,
    save_every=5,
):
    device, device_name, total_mem_gb = get_device_info()

    print("=" * 70)
    print("开始训练 SketchUNetPlus")
    print(f"设备: {device}")
    print(f"设备名称: {device_name}")
    if device.type == "cuda":
        print(f"显存总量: {total_mem_gb:.2f} GB")
    print("=" * 70)

    train_dataset = SketchDataset(DATA_ROOT, mode="train", image_size=image_size, split_ratio=0.9)
    val_dataset = SketchDataset(DATA_ROOT, mode="val", image_size=image_size, split_ratio=0.9)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("数据集为空，请先运行 gen_dataset.py 生成训练数据")

    use_amp = (device.type == "cuda")
    workers = 0 if os.name == "nt" else suggest_num_workers()
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=(workers > 0),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=(workers > 0),
        drop_last=False,
    )

    print(f"训练集: {len(train_dataset)} 张")
    print(f"验证集: {len(val_dataset)} 张")
    print(f"batch_size: {batch_size}")
    print(f"image_size: {image_size}")
    print(f"num_workers: {workers}")
    print(f"AMP: {'开启' if use_amp else '关闭'}")
    print("-" * 70)

    model = SketchUNetPlus().to(device)

    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0

        for batch_idx, (img, target, _) in enumerate(train_loader):
            img = img.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(img)

                loss_mse = mse(pred, target)
                loss_l1 = l1(pred, target)
                loss_edge = edge_loss(pred, target, device)
                loss_tone = tone_loss(pred, target)
                loss_char = charbonnier_loss(pred, target)

                # 深度学习主导版 loss
                loss = (
                    0.15 * loss_mse +
                    0.30 * loss_l1 +
                    0.20 * loss_edge +
                    0.15 * loss_tone +
                    0.20 * loss_char
                )

            scaler.scale(loss).backward()

            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()

            if batch_idx % 20 == 0:
                print(
                    f"Epoch {epoch + 1:03d} | "
                    f"Batch {batch_idx:03d}/{len(train_loader):03d} | "
                    f"Loss {loss.item():.4f} | "
                    f"MSE {loss_mse.item():.4f} | "
                    f"L1 {loss_l1.item():.4f} | "
                    f"Edge {loss_edge.item():.4f}"
                )

        avg_train_loss = train_loss_sum / max(len(train_loader), 1)

        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for img, target, _ in val_loader:
                img = img.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(img)

                    loss = (
                        0.15 * mse(pred, target) +
                        0.30 * l1(pred, target) +
                        0.20 * edge_loss(pred, target, device) +
                        0.15 * tone_loss(pred, target) +
                        0.20 * charbonnier_loss(pred, target)
                    )

                val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / max(len(val_loader), 1)
        scheduler.step()

        if (epoch + 1) % save_every == 0:
            save_sample_images(model, val_loader, device, epoch + 1, use_amp=use_amp)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"✅ 保存最佳模型: {BEST_MODEL_PATH}")

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train {avg_train_loss:.4f} | "
            f"Val {avg_val_loss:.4f} | "
            f"LR {scheduler.get_last_lr()[0]:.6f}"
        )
        print("-" * 70)

    torch.save(model.state_dict(), FINAL_MODEL_PATH)

    print("\n🎉 训练完成")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳模型: {BEST_MODEL_PATH}")
    print(f"最终模型: {FINAL_MODEL_PATH}")
    print(f"样例图目录: {SAMPLE_DIR}")


if __name__ == "__main__":
    train(
        num_epochs=40,
        batch_size=8,
        image_size=512,
        learning_rate=1e-4,
        weight_decay=1e-5,
        grad_clip=1.0,
        save_every=5,
    )