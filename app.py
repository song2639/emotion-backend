"""
情绪识别后端 —— 接入你的真实 CNN 模型
==========================================
模型：SimpleCNN，训练自 FER2013Train 数据（8类）
权重：best_model.pth（已从微信文件夹复制到本目录）

模型输入：48x48 灰度图，归一化 mean=0.5, std=0.5
输出：8类情绪概率（对应 FER+ 标签）

运行：
  C:\\Users\\26390\\.workbuddy\\binaries\\python\\envs\\emotion\\Scripts\\python.exe app.py

接口：POST /predict
  - multipart/form-data，字段名 image
  - 返回：{ emotion, confidence, all_scores, face_detected }
"""

import os
import io
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

from models import SimpleCNN, MediumCNN

app = Flask(__name__)
CORS(app)

# ── 路径配置 ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODEL_SIMPLE_PATH = os.path.join(BASE_DIR, 'best_model.pth')
MODEL_MEDIUM_PATH = os.path.join(BASE_DIR, 'best_model_medium.pth')
HAAR_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# ── FER+ 8类标签映射 ───────────────────────────────────────────
CLASS_LABELS = ['neutral', 'happy', 'surprise', 'fear', 'disgust', 'anger', 'contempt', 'sad']

# ── 双模型加载 ──────────────────────────────────────────────────
models = {}

print("[*] Loading SimpleCNN...")
models['simple'] = SimpleCNN(num_classes=8)
models['simple'].load_state_dict(torch.load(MODEL_SIMPLE_PATH, map_location='cpu', weights_only=True))
models['simple'].eval()
print("[OK] SimpleCNN loaded:", MODEL_SIMPLE_PATH)

print("[*] Loading MediumCNN...")
models['medium'] = MediumCNN(num_classes=8)
models['medium'].load_state_dict(torch.load(MODEL_MEDIUM_PATH, map_location='cpu', weights_only=True))
models['medium'].eval()
print("[OK] MediumCNN loaded:", MODEL_MEDIUM_PATH)

# ── 人脸检测器 ──────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)
print("[OK] Face detector ready")


def preprocess_image(cv_img):
    """
    人脸检测 → 裁剪 → 缩放 48x48 → 灰度 → 归一化
    返回 (tensor, face_detected_bool)
    针对训练集小图（48x48灰度）做了特殊兼容
    """
    h, w = cv_img.shape[:2]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # 如果图片本身就很小且接近正方形，大概率是训练集样本（48x48），直接缩放
    # 条件：宽高均 ≤ 200px，且长宽比在 0.5~2.0 之间
    # 处理方式与训练时完全一致：直接缩放到 48x48 → 归一化
    aspect_ratio = w / h if h > 0 else 1
    is_small_square = (h <= 200 and w <= 200 and 0.5 <= aspect_ratio <= 2.0)
    if is_small_square:
        print(f"[Debug] 小图直通模式: {w}x{h}, 与训练预处理一致")
        # 直接缩放到 48x48，不做任何增强，保持与训练时完全一致
        resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor, True  # 训练集样本默认包含人脸

    # 正常大图走 Haar 人脸检测
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )

    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        pad = int(max(fw, fh) * 0.15)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + fw + pad)
        y2 = min(gray.shape[0], y + fh + pad)
        face = gray[y1:y2, x1:x2]
        face_detected = True
    else:
        # 没检测到人脸：尝试降低阈值再检一次
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30)
        )
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            pad = int(max(fw, fh) * 0.15)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(gray.shape[1], x + fw + pad)
            y2 = min(gray.shape[0], y + fh + pad)
            face = gray[y1:y2, x1:x2]
            face_detected = True
        else:
            # 仍然没检测到，取图像中心区域作为 fallback
            center_crop = min(h, w)
            y1 = (h - center_crop) // 2
            x1 = (w - center_crop) // 2
            face = gray[y1:y1+center_crop, x1:x1+center_crop]
            face_detected = False

    resized = cv2.resize(face, (48, 48))
    # 调试：打印裁剪后灰度图的统计
    print(f"[Debug] face crop stats: min={resized.min()}, max={resized.max()}, mean={resized.mean():.1f}, std={resized.std():.1f}")
    tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5  # Normalize(mean=0.5, std=0.5)
    print(f"[Debug] tensor stats: mean={tensor.mean():.3f}, std={tensor.std():.3f}")
    return tensor, face_detected


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image field named "image"'}), 400

    # 读取模型选择参数，默认用 SimpleCNN
    model_name = request.form.get('model', 'simple')
    if model_name not in models:
        model_name = 'simple'
    selected_model = models[model_name]

    try:
        raw_bytes = request.files['image'].read()
        print(f"[Debug] 收到图片大小: {len(raw_bytes)/1024:.1f} KB, 模型: {model_name}")
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img = pil_img.convert('RGB')
        orig_w, orig_h = pil_img.size
        max_side = 800
        if max(pil_img.size) > max_side:
            ratio = max_side / max(pil_img.size)
            new_w = int(pil_img.width * ratio)
            new_h = int(pil_img.height * ratio)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            print(f"[Debug] 图片缩放: {orig_w}x{orig_h} -> {new_w}x{new_h}")
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({'error': f'Image read error: {str(e)}'}), 400

    tensor, face_detected = preprocess_image(cv_img)

    with torch.no_grad():
        logits = selected_model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()

    all_scores = {CLASS_LABELS[i]: round(float(probs[i]), 4) for i in range(8)}
    best_label = CLASS_LABELS[int(np.argmax(probs))]
    confidence = float(max(probs))

    print(f"[Predict] [{model_name}] {best_label} ({confidence:.2%}) | face={face_detected}")

    return jsonify({
        'emotion': best_label,
        'confidence': round(confidence, 4),
        'all_scores': all_scores,
        'face_detected': face_detected,
        'model': model_name,
        'is_mock': False
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'SimpleCNN-FER8', 'mock': False})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[*] Server running at 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
