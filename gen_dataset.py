# -*- coding: utf-8 -*-


import os
import cv2
import numpy as np
from tqdm import tqdm

INPUT_DIR = "dataset/train/input"
TARGET_DIR = "dataset/train/target"

IMG_SIZE = 512
NUM_SAMPLES = 2000


def ensure_dirs():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(TARGET_DIR, exist_ok=True)


def generate_gradient_background(size=512):
    h, w = size, size
    bg = np.zeros((h, w, 3), dtype=np.uint8)

    direction = np.random.choice(["horizontal", "vertical", "diag"])
    c1 = np.random.randint(30, 160, size=3)
    c2 = np.random.randint(100, 240, size=3)

    if direction == "horizontal":
        for i in range(w):
            alpha = i / max(w - 1, 1)
            bg[:, i] = (1 - alpha) * c1 + alpha * c2
    elif direction == "vertical":
        for i in range(h):
            alpha = i / max(h - 1, 1)
            bg[i, :] = (1 - alpha) * c1 + alpha * c2
    else:
        for y in range(h):
            for x in range(w):
                alpha = (x + y) / max(h + w - 2, 1)
                bg[y, x] = (1 - alpha) * c1 + alpha * c2

    return bg.astype(np.uint8)


def add_soft_blobs(img, num_blobs=20):
    h, w = img.shape[:2]
    overlay = img.copy()

    for _ in range(num_blobs):
        center = (np.random.randint(0, w), np.random.randint(0, h))
        axes = (np.random.randint(20, 120), np.random.randint(20, 120))
        angle = np.random.randint(0, 180)
        color = tuple(int(x) for x in np.random.randint(40, 230, size=3))
        cv2.ellipse(overlay, center, axes, angle, 0, 360, color, -1)

    alpha = np.random.uniform(0.15, 0.35)
    out = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    return out


def add_lines_and_edges(img, num_lines=40):
    h, w = img.shape[:2]

    for _ in range(num_lines):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        color = tuple(int(x) for x in np.random.randint(20, 220, size=3))
        thickness = np.random.randint(1, 4)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return img


def add_polygons_and_circles(img, num_shapes=25):
    h, w = img.shape[:2]

    for _ in range(num_shapes):
        mode = np.random.choice(["circle", "rect", "poly"])
        color = tuple(int(x) for x in np.random.randint(30, 220, size=3))

        if mode == "circle":
            center = (np.random.randint(0, w), np.random.randint(0, h))
            radius = np.random.randint(8, 60)
            cv2.circle(img, center, radius, color, -1)

        elif mode == "rect":
            x1, y1 = np.random.randint(0, w - 10), np.random.randint(0, h - 10)
            x2 = min(w - 1, x1 + np.random.randint(10, 120))
            y2 = min(h - 1, y1 + np.random.randint(10, 120))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        else:
            n_pts = np.random.randint(3, 7)
            pts = np.zeros((n_pts, 1, 2), dtype=np.int32)
            pts[:, 0, 0] = np.random.randint(0, w, size=n_pts)
            pts[:, 0, 1] = np.random.randint(0, h, size=n_pts)
            cv2.fillPoly(img, [pts], color)

    return img


def add_texture(img):
    h, w = img.shape[:2]

    noise = np.random.normal(0, 8, img.shape).astype(np.float32)
    base = np.clip(img.astype(np.float32) + noise, 0, 255)

    texture = np.random.normal(128, 25, (h, w)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), 8)
    texture_3c = cv2.merge([texture, texture, texture])

    out = cv2.addWeighted(base, 0.88, texture_3c, 0.12, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_pseudo_natural_image(size=512):
    img = generate_gradient_background(size=size)
    img = add_soft_blobs(img, num_blobs=np.random.randint(10, 30))
    img = add_polygons_and_circles(img, num_shapes=np.random.randint(10, 35))
    img = add_lines_and_edges(img, num_lines=np.random.randint(20, 60))
    img = add_texture(img)

    if np.random.rand() < 0.8:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    if np.random.rand() < 0.6:
        blur = cv2.GaussianBlur(img, (0, 0), 2)
        img = cv2.addWeighted(img, 1.3, blur, -0.3, 0)

    return np.clip(img, 0, 255).astype(np.uint8)


def soft_white_background(gray):
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    p5, p95 = np.percentile(gray, 5), np.percentile(gray, 95)

    if p95 > p5:
        gray = np.clip((gray - p5) * 255.0 / (p95 - p5), 0, 255).astype(np.uint8)

    gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
    _, bright = cv2.threshold(gray, 205, 255, cv2.THRESH_TOZERO)
    result = cv2.addWeighted(gray, 0.88, bright, 0.12, 10)
    return np.clip(result, 0, 255).astype(np.uint8)


def generate_paper_texture(shape):
    noise = np.random.normal(245, 4, shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.8)
    return np.clip(noise, 235, 255).astype(np.uint8)


def make_sketch_target(bgr_img):

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)

    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (0, 0), sigmaX=10, sigmaY=10)
    dodge = cv2.divide(gray, 255 - blur, scale=256)

    edges = cv2.Canny(gray, 50, 130)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    line_layer = 255 - edges

    # 关键：降低传统硬边权重，提升柔和素描基底
    sketch = cv2.addWeighted(dodge, 0.90, line_layer, 0.10, 0)

    sketch = cv2.normalize(sketch, None, 0, 255, cv2.NORM_MINMAX)
    sketch = cv2.GaussianBlur(sketch, (0, 0), 0.6)
    sketch = soft_white_background(sketch)

    texture = generate_paper_texture(sketch.shape)
    sketch = cv2.multiply(
        sketch.astype(np.float32) / 255.0,
        texture.astype(np.float32) / 255.0,
        scale=255.0
    )

    sketch = np.clip(sketch, 0, 255).astype(np.uint8)
    sketch = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(sketch)
    sketch = cv2.addWeighted(sketch, 0.95, np.full_like(sketch, 255), 0.05, 0)

    return np.clip(sketch, 0, 255).astype(np.uint8)


def augment_image(img):
    out = img.copy()

    if np.random.rand() < 0.5:
        out = cv2.flip(out, 1)

    if np.random.rand() < 0.4:
        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.uniform(-12, 12)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if np.random.rand() < 0.4:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    return out


def make_dataset(num_samples=NUM_SAMPLES, size=IMG_SIZE):
    ensure_dirs()

    print(f"正在生成训练数据集，共 {num_samples} 对 ...")

    for i in tqdm(range(num_samples)):
        img = make_pseudo_natural_image(size=size)
        img = augment_image(img)
        sketch = make_sketch_target(img)

        cv2.imwrite(os.path.join(INPUT_DIR, f"{i:06d}.jpg"), img)
        cv2.imwrite(os.path.join(TARGET_DIR, f"{i:06d}.jpg"), sketch)

    print("数据集生成完成")
    print(f"输入目录: {INPUT_DIR}")
    print(f"目标目录: {TARGET_DIR}")


if __name__ == "__main__":
    make_dataset()
