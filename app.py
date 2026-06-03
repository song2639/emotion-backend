"""
情绪识别后端 v2 —— 三模型切换 + 情绪日记
==========================================
模型：SimpleCNN / MediumCNN / AdvancedCNN
日记：SQLite 本地存储，REST API

接口：
  POST /predict        — 情绪识别（支持 model 参数：simple|medium|advanced）
  POST /diary/save     — 保存日记 { user_id, emotion, confidence, all_scores, note }
  GET  /diary/list     — 查询记录 ?user_id=xxx&year=2026&month=6
  GET  /diary/weekly   — 本周报告 ?user_id=xxx
  GET  /health         — 健康检查
"""

import os, io, sqlite3, json, warnings, logging
from datetime import datetime, timedelta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import cv2, torch
from PIL import Image
from flask import Flask, request, jsonify, g
from flask_cors import CORS

from models import SimpleCNN, MediumCNN, AdvancedCNN

app = Flask(__name__)
CORS(app)

# ── 路径配置 ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, 'emotion_diary.db')
HAAR_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# ── 标签映射 ────────────────────────────────────────────────────
CLASS_LABELS = ['neutral', 'happy', 'surprise', 'fear', 'disgust', 'anger', 'contempt', 'sad']

# ── 三模型加载 ──────────────────────────────────────────────────
models = {}

for arch, cls, fname in [
    ('simple', SimpleCNN, 'best_model.pth'),
    ('medium', MediumCNN, 'best_model_medium.pth'),
    ('advanced', AdvancedCNN, 'best_model_advanced.pth')
]:
    path = os.path.join(BASE_DIR, fname)
    print(f"[*] Loading {arch}...")
    models[arch] = cls(num_classes=8)
    models[arch].load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    models[arch].eval()
    print(f"[OK] {arch} loaded")

face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)
print("[OK] Face detector ready")

# ── 数据库初始化 ─────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS diary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        date TEXT NOT NULL,
        emotion TEXT NOT NULL,
        confidence REAL,
        all_scores TEXT,
        note TEXT DEFAULT '',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, date)
    )''')
    conn.commit()
    conn.close()

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db: db.close()

init_db()
print("[OK] SQLite diary database ready")

# ── 图像预处理 ──────────────────────────────────────────────────
def preprocess_image(cv_img):
    h, w = cv_img.shape[:2]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    aspect_ratio = w / h if h > 0 else 1
    is_small_square = (h <= 200 and w <= 200 and 0.5 <= aspect_ratio <= 2.0)
    if is_small_square:
        resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor, True
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        pad = int(max(fw, fh) * 0.15)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + fw + pad), min(h, y + fh + pad)
        face = gray[y1:y2, x1:x2]
        face_detected = True
    else:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            pad = int(max(fw, fh) * 0.15)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + fw + pad), min(h, y + fh + pad)
            face = gray[y1:y2, x1:x2]
            face_detected = True
        else:
            center_crop = min(h, w)
            y1, x1 = (h - center_crop) // 2, (w - center_crop) // 2
            face = gray[y1:y1+center_crop, x1:x1+center_crop]
            face_detected = False
    resized = cv2.resize(face, (48, 48))
    tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor, face_detected

# ── 情绪识别 ────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image field named "image"'}), 400
    model_name = request.form.get('model', 'simple')
    if model_name not in models:
        model_name = 'simple'
    selected_model = models[model_name]
    try:
        raw_bytes = request.files['image'].read()
        print(f"[Debug] image: {len(raw_bytes)/1024:.1f} KB, model: {model_name}")
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
        if max(pil_img.size) > 800:
            ratio = 800 / max(pil_img.size)
            pil_img = pil_img.resize((int(pil_img.width*ratio), int(pil_img.height*ratio)), Image.LANCZOS)
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({'error': f'Image read error: {str(e)}'}), 400
    tensor, face_detected = preprocess_image(cv_img)
    with torch.no_grad():
        probs = torch.softmax(selected_model(tensor), dim=1).squeeze().tolist()
    all_scores = {CLASS_LABELS[i]: round(float(probs[i]), 4) for i in range(8)}
    best_label = CLASS_LABELS[int(np.argmax(probs))]
    confidence = float(max(probs))
    print(f"[Predict] [{model_name}] {best_label} ({confidence:.2%})")
    return jsonify({
        'emotion': best_label, 'confidence': round(confidence, 4),
        'all_scores': all_scores, 'face_detected': face_detected,
        'model': model_name, 'is_mock': False
    })

# ── 日记保存 ────────────────────────────────────────────────────
@app.route('/diary/save', methods=['POST'])
def diary_save():
    data = request.get_json()
    user_id = data.get('user_id', 'default')
    emotion = data.get('emotion', '')
    confidence = data.get('confidence', 0)
    all_scores = json.dumps(data.get('all_scores', {}))
    note = data.get('note', '')
    # 支持补签：前端可传指定日期，否则默认今天
    date_str = data.get('date', datetime.now().strftime('%Y-%m-%d'))
    if not emotion:
        return jsonify({'error': 'emotion is required'}), 400
    db = get_db()
    db.execute('''INSERT OR REPLACE INTO diary (user_id, date, emotion, confidence, all_scores, note)
                  VALUES (?, ?, ?, ?, ?, ?)''',
               (user_id, date_str, emotion, confidence, all_scores, note))
    db.commit()
    return jsonify({'status': 'ok', 'date': date_str, 'emotion': emotion})

# ── 日记列表（某月）──────────────────────────────────────────────
@app.route('/diary/list', methods=['GET'])
def diary_list():
    user_id = request.args.get('user_id', 'default')
    year = request.args.get('year', str(datetime.now().year))
    month = request.args.get('month', str(datetime.now().month))
    db = get_db()
    rows = db.execute(
        'SELECT date, emotion, confidence, note FROM diary WHERE user_id=? AND strftime("%Y", date)=? AND strftime("%m", date)=? ORDER BY date',
        (user_id, year.zfill(4), month.zfill(2))
    ).fetchall()
    records = [dict(r) for r in rows]
    for r in records:
        r['confidence'] = round(r['confidence'], 4)
    return jsonify({'records': records, 'count': len(records)})

# ── 每周报告 ────────────────────────────────────────────────────
@app.route('/diary/weekly', methods=['GET'])
def diary_weekly():
    user_id = request.args.get('user_id', 'default')
    # 支持指定周的起始日期，默认本周一
    week_start_str = request.args.get('week_start', '')
    if week_start_str:
        monday = datetime.strptime(week_start_str, '%Y-%m-%d')
    else:
        today = datetime.now()
        monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6)
    db = get_db()
    rows = db.execute(
        'SELECT date, emotion, confidence, all_scores FROM diary WHERE user_id=? AND date BETWEEN ? AND ? ORDER BY date',
        (user_id, monday.strftime('%Y-%m-%d'), sunday.strftime('%Y-%m-%d'))
    ).fetchall()
    if not rows:
        return jsonify({'message': '本周暂无记录', 'days': 0, 'emotions': {}, 'week_start': monday.strftime('%Y-%m-%d')})
    emotion_count = {}
    total_confidence = 0
    for r in rows:
        e = r['emotion']
        emotion_count[e] = emotion_count.get(e, 0) + 1
        total_confidence += r['confidence']
    trend = 'stable'
    if len(rows) >= 2:
        first_half = [r['emotion'] for r in rows[:len(rows)//2]]
        second_half = [r['emotion'] for r in rows[len(rows)//2:]]
        positive = {'happy', 'surprise'}
        negative = {'anger', 'sad', 'fear', 'disgust'}
        def score(emotions):
            return sum(1 for e in emotions if e in positive) - sum(1 for e in emotions if e in negative)
        s1, s2 = score(first_half), score(second_half)
        if s2 > s1: trend = 'up'
        elif s2 < s1: trend = 'down'
    return jsonify({
        'days': len(rows),
        'emotions': {k: {'count': v, 'pct': round(v/len(rows)*100, 1)} for k, v in sorted(emotion_count.items(), key=lambda x: -x[1])},
        'dominant': max(emotion_count, key=emotion_count.get),
        'avg_confidence': round(total_confidence / len(rows), 4),
        'trend': trend,
        'streak': len(rows)
    })

# ── 健康检查 ────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models': list(models.keys())})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[*] Server running at 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
