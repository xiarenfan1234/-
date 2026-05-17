#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from train import SketchUNetPlus

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
RESULT_FOLDER = os.path.join(STATIC_DIR, 'results')

MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, 'sketch_model_best.pth'),
    os.path.join(BASE_DIR, 'sketch_model_final.pth'),
    os.path.join(BASE_DIR, 'sketch_model.pth'),
]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
loaded_model_path = None


def load_model():
    global model, loaded_model_path

    for model_path in MODEL_CANDIDATES:
        if os.path.exists(model_path):
            try:
                net = SketchUNetPlus().to(device)
                state = torch.load(model_path, map_location=device)

                if isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']

                clean_state = {}
                model_state = net.state_dict()

                for k, v in state.items():
                    nk = k.replace('module.', '')
                    if nk in model_state and model_state[nk].shape == v.shape:
                        clean_state[nk] = v

                model_state.update(clean_state)
                net.load_state_dict(model_state, strict=False)
                net.eval()

                model = net
                loaded_model_path = model_path
                print(f" 模型加载成功: {model_path}")
                return
            except Exception as e:
                print(f" 模型加载失败 {model_path}: {e}")

    model = None
    loaded_model_path = None
    print(" 未加载到可用模型，将仅使用传统铅笔画算法")


load_model()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path, target_size=(512, 512)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图片文件")

    original_size = (img.shape[1], img.shape[0])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)

    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    return img, img_tensor.to(device), original_size


def postprocess_output(output_tensor, original_size):
    output = output_tensor.squeeze().detach().cpu().numpy()

    if output.ndim == 3:
        output = output[0]

    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)
    output = cv2.resize(output, original_size, interpolation=cv2.INTER_CUBIC)
    return output


def soft_white_background(gray):
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    p5, p95 = np.percentile(gray, 5), np.percentile(gray, 95)

    if p95 > p5:
        gray = np.clip((gray - p5) * 255.0 / (p95 - p5), 0, 255).astype(np.uint8)

    gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
    _, bright = cv2.threshold(gray, 210, 255, cv2.THRESH_TOZERO)
    result = cv2.addWeighted(gray, 0.90, bright, 0.10, 8)
    return np.clip(result, 0, 255).astype(np.uint8)


def generate_paper_texture(shape):
    noise = np.random.normal(246, 3, shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.8)
    return np.clip(noise, 238, 255).astype(np.uint8)


def classical_pencil_sketch(bgr_img):

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)

    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (0, 0), sigmaX=10, sigmaY=10)
    dodge = cv2.divide(gray, 255 - blur, scale=256)

    edges = cv2.Canny(gray, 50, 130)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    line_layer = 255 - edges

    # 辅助版传统结果：更柔和一些
    sketch = cv2.addWeighted(dodge, 0.88, line_layer, 0.12, 0)
    sketch = cv2.normalize(sketch, None, 0, 255, cv2.NORM_MINMAX)
    sketch = soft_white_background(sketch)

    texture = generate_paper_texture(sketch.shape)
    sketch = cv2.multiply(
        sketch.astype(np.float32) / 255.0,
        texture.astype(np.float32) / 255.0,
        scale=255.0
    )

    return np.clip(sketch, 0, 255).astype(np.uint8)


def enhance_final_sketch(gray, source_bgr=None):

    gray = cv2.GaussianBlur(gray, (0, 0), 0.4)
    gray = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)
    gray = soft_white_background(gray)

    if source_bgr is not None:
        src_gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
        src_edges = cv2.Canny(src_gray, 70, 170)
        src_edges = cv2.GaussianBlur(255 - src_edges, (3, 3), 0)
        gray = cv2.addWeighted(gray, 0.94, src_edges, 0.06, 0)

    return np.clip(gray, 0, 255).astype(np.uint8)


def hybrid_pencil_sketch(image_path):

    bgr_img, input_tensor, original_size = preprocess_image(image_path)

    classical = classical_pencil_sketch(bgr_img)

    if model is None:
        final_sketch = classical
    else:
        with torch.no_grad():
            output_tensor = model(input_tensor)

        deep_out = postprocess_output(output_tensor, original_size)
        deep_out = cv2.GaussianBlur(deep_out, (0, 0), 0.5)
        deep_out = cv2.normalize(deep_out, None, 0, 255, cv2.NORM_MINMAX)

        # 关键：深度学习70%，传统法30%
        merged = cv2.addWeighted(classical, 0.30, deep_out, 0.70, 0)
        final_sketch = enhance_final_sketch(merged, source_bgr=bgr_img)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"sketch_{timestamp}.jpg"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

    cv2.imwrite(result_path, final_sketch)
    print(f" 铅笔画生成完成: {result_filename}")

    return result_filename


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('没有选择文件')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('没有选择文件')
        return redirect(url_for('index'))

    if not (file and allowed_file(file.filename)):
        flash('文件格式不支持，请上传 png/jpg/jpeg/gif/bmp/webp 格式')
        return redirect(url_for('index'))

    try:
        original_filename = secure_filename(file.filename)
        filename_base, filename_ext = os.path.splitext(original_filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{filename_base}_{timestamp}{filename_ext}"

        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(upload_path)
        print(f" 文件已上传: {unique_filename}")

        result_filename = hybrid_pencil_sketch(upload_path)

        return render_template(
            'result.html',
            upload_image=unique_filename,
            result_image=result_filename,
            model_status='已加载' if model is not None else '未加载（当前使用传统算法）'
        )

    except Exception as e:
        error_msg = f"处理图片时出错: {str(e)}"
        print(f" {error_msg}")
        flash(error_msg)
        return redirect(url_for('index'))


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.errorhandler(413)
def too_large(e):
    flash('文件太大，请上传小于16MB的图片')
    return redirect(url_for('index'))


@app.errorhandler(404)
def not_found(e):
    return "404 Not Found", 404


@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': loaded_model_path,
        'device': str(device),
        'classical_ratio': 0.30,
        'deep_ratio': 0.70,
    }


if __name__ == '__main__':
    print("=" * 60)
    print(" 深度学习主导的混合式铅笔画风格转换系统")
    print(f" 上传目录: {app.config['UPLOAD_FOLDER']}")
    print(f" 结果目录: {app.config['RESULT_FOLDER']}")
    print(f"️ 设备: {device}")
    print(f" 模型状态: {'已加载' if model is not None else '未加载，仅传统算法'}")
    print(" 融合比例: 传统法 30% / 深度学习 70%")
    print(" 服务地址: http://127.0.0.1:6006")
    print("=" * 60)

    app.run(host='0.0.0.0', port=6006, debug=True)
