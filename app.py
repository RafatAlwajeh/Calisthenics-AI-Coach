# -*- coding: utf-8 -*-
"""
====================================================================
Gym Enterprise AI — v3.6 | النسخة الشاملة (تتبع الجسم + تتبع الأصابع واليدين)
Prepared by Eng : Rafat Al-Wajeh (Rafat Hazza Qasim)
====================================================================
"""

import os
import time
import traceback
import urllib.request
import streamlit as st
import cv2
import numpy as np
import tempfile

st.set_page_config(page_title="Gym Enterprise AI — منصة المدرب الذكي",
                   page_icon="💪", layout="wide")

OUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# معالجة النصوص العربية بدقة
# ------------------------------------------------------------------
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def ar(text):
        return get_display(arabic_reshaper.reshape(text))
except Exception:
    def ar(text):
        return text

FONT_FILE = "Cairo-Regular.ttf"
FONT_URL = ("https://github.com/google/fonts/raw/main/ofl/cairo/"
            "Cairo%5Bslnt%2Cwght%5D.ttf")
_font_cache = {}

def ensure_font():
    if os.path.exists(FONT_FILE):
        return FONT_FILE
    try:
        urllib.request.urlretrieve(FONT_URL, FONT_FILE)
        return FONT_FILE
    except Exception:
        return None

def put_ar(frame, text, x, y, size=22, color=(255, 255, 255)):
    from PIL import Image, ImageDraw, ImageFont
    s = int(size)
    if s not in _font_cache:
        path = ensure_font()
        try:
            _font_cache[s] = ImageFont.truetype(path, s) if path else ImageFont.loadDefault()
        except Exception:
            _font_cache[s] = ImageFont.loadDefault()
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img).text((x, y), ar(text), font=_font_cache[s], fill=color[::-1])
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ------------------------------------------------------------------
# المحركات: YOLOv8 (للجسم) / MediaPipe Hands (للأصابع واليدين)
# ------------------------------------------------------------------
@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n-pose.pt")

@st.cache_resource
def load_mp_hands():
    import mediapipe as mp
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def process_frame(frame, mode_type):
    """معالجة الإطار بناءً على اختيار المستخدم (جسم أو أصابع يد)"""
    h, w = frame.shape[:2]
    annotated = frame.copy()
    
    if mode_type == "hands":
        import mediapipe as mp
        mp_hands = load_mp_hands()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb)
        
        hands_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # رسم الهيكل العظمي والنقاط الحمراء على الأصابع (مطابق للصور المرجعية)
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
                # استخراج إحداثيات المعصم وبداية الأصابع
                lm = hand_landmarks.landmark
                hands_data.append({"wrist": [lm[0].x * w, lm[0].y * h], "index_tip": [lm[8].x * w, lm[8].y * h]})
        return annotated, hands_data

    else:
        # محرك YOLOv8 للجسم
        res = load_yolo().track(frame, persist=True, verbose=False)[0]
        if res.keypoints is None or res.boxes is None:
            return annotated, []
        
        kps_all = res.keypoints.xy.cpu().numpy() if res.keypoints.xy is not None else []
        confs_all = res.keypoints.conf
        confs_all = confs_all.cpu().numpy() if confs_all is not None else np.ones((len(kps_all), 17))
        
        annotated = res.plot(boxes=False, labels=False)
        persons = []
        for i, kp in enumerate(kps_all):
            if len(kp) < 10:
                continue
            persons.append({"kps": kp, "conf": float(np.nanmean(confs_all[i])) if i < len(confs_all) else 0.8})
        return annotated, persons

def calculate_angle(a, b, c):
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# ------------------------------------------------------------------
# أدوات الرسم والنظام
# ------------------------------------------------------------------
class Painter:
    WHITE = (255, 255, 255)
    GRAY = (200, 200, 200)
    DARK = (15, 23, 42)
    GREEN = (46, 204, 113)
    RED = (56, 67, 235)
    AMBER = (58, 178, 247)

    @staticmethod
    def blend(frame, x1, y1, x2, y2, alpha=0.75):
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), Painter.DARK, -1)
        return cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0)

def draw_watermark(frame):
    h, w = frame.shape[:2]
    cv2.putText(frame, "WF.DIETNCHEAT.COM", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, Painter.GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, "Eng: Rafat Al-Wajeh", (w - 220, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Painter.GRAY, 1, cv2.LINE_AA)
    return frame

# ------------------------------------------------------------------
# واجهة Streamlit
# ------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600&display=swap');
  h1,h2,h3,h4,p,label,.stMarkdown {direction:rtl;text-align:right;font-family:'Cairo',Tahoma,sans-serif;}
  section[data-testid="stSidebar"] {direction:rtl;background:linear-gradient(180deg,#0f172a,#1e293b);}
  section[data-testid="stSidebar"] * {text-align:right;color:#e2e8f0 !important;}
  .stButton>button {width:100%;border-radius:12px;font-weight:600;height:3em;
    background:linear-gradient(135deg,#3b82f6,#1d4ed8);color:#fff !important;border:none;}
  .hero {border:1px solid #334155;border-radius:14px;padding:16px 20px;margin-bottom:14px;
    background:linear-gradient(135deg,#1e293b,#0f172a);color:#fff;direction:rtl;text-align:right;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🏋️‍♂️ مركز القيادة والتحكم الذكي")
st.sidebar.markdown("---")
mode_ai = st.sidebar.radio("اختر وضع الذكاء الاصطناعي:", [
    "✋ تقطاع الأصابع واليدين (MediaPipe Hands)",
    "🏃‍♂️ تحليل تمارين الجسم (YOLOv8 Pose)"
])
engine_type = "hands" if "MediaPipe" in mode_ai else "body"

mode_action = st.sidebar.radio("وضع التشغيل:", ["📤 رفع فيديو", "📸 لقطة كاميرا (معالجة فورية)"])

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Eng : Rafat Al-Wajeh**  \n[WF.DIETNCHEAT.COM](https://wf.dietncheat.com)")

st.markdown("""
<div class="hero"><h2>🛡️ Gym Enterprise AI — وضع المفاصل والأصابع المتقدم</h2>
<p>دعم كامل لرصد مفاصل الأصابع واليدين أونلاين مثل النماذج القياسية أو تحليل الجسم.</p></div>
""", unsafe_allow_html=True)

if mode_action.startswith("📤"):
    st.subheader("📤 معالجة فيديو")
    up = st.file_uploader("ارفع الفيديو:", type=["mp4", "avi", "mov"])
    if up is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(up.read())
        tfile.close()
        st.video(tfile.name)
        if st.button("🚀 بدء المعالجة"):
            cap = cv2.VideoCapture(tfile.name)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            out_path = os.path.join(OUT_DIR, f"gym_out_{int(time.time())}.mp4")
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            
            live = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, _ = process_frame(frame, engine_type)
                annotated = draw_watermark(annotated)
                out.write(annotated)
                live.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            cap.release()
            out.release()
            st.success("✅ تمت المعالجة بنجاح!")
            with open(out_path, "rb") as f:
                st.download_button("📥 تنزيل الفيديو المعالج", data=f.read(), file_name="processed.mp4", mime="video/mp4")
else:
    st.subheader("📸 تحليل لقطة فورية (معالجة مباشرة لليد أو الجسم)")
    shot = st.camera_input("التقط صورة:")
    if shot is not None:
        data = np.frombuffer(shot.getvalue(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        annotated, results = process_frame(frame, engine_type)
        annotated = draw_watermark(annotated)
        
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        if results:
            st.success("✅ تم رصد المفاصل والعناصر بنجاح تام مثل الصور المرجعية!")
        else:
            st.warning("⚠️ لم يُكتشف أي عنصر واضح — تأكد من وضع يدك أو جسمك بابتعاد مناسب أمام الكاميرا.")
