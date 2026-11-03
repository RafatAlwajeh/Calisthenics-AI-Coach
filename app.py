# -*- coding: utf-8 -*-
"""
====================================================================
Gym Enterprise AI — v4.0 | النسخة المستقرة والشاملة
Prepared by Eng : Rafat Al-Wajeh (Rafat Hazza Qasim)
====================================================================
"""

import os
import time
import random
import streamlit as st
import cv2
import numpy as np
import tempfile

st.set_page_config(page_title="Gym Enterprise AI — منصة المدرب الذكي",
                   page_icon="💪", layout="wide")

OUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# معالجة النصوص العربية بدقة (RTL)
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
        import urllib.request
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
# محرك الذكاء الاصطناعي YOLOv8-Pose
# ------------------------------------------------------------------
@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n-pose.pt")

def detect_persons(frame):
    res = load_yolo().track(frame, persist=True, verbose=False)[0]
    if res.keypoints is None or res.boxes is None:
        return frame.copy(), []
    
    kps_all = res.keypoints.xy.cpu().numpy() if res.keypoints.xy is not None else []
    confs_all = res.keypoints.conf
    confs_all = confs_all.cpu().numpy() if confs_all is not None else np.ones((len(kps_all), 17))
    
    annotated = res.plot(boxes=False, labels=False)
    persons = []
    for i, kp in enumerate(kps_all):
        if len(kp) < 10:
            continue
        persons.append({
            "id": i,
            "kps": kp,
            "conf": float(np.nanmean(confs_all[i])) if i < len(confs_all) else 0.8
        })
    return annotated, persons

def get_arm(person):
    kp = person["kps"]
    return kp[5][:2], kp[7][:2], kp[9][:2] # الكتف، المرفق، الرسغ

def calculate_angle(a, b, c):
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# ------------------------------------------------------------------
# محرك جودة التكرارات والتحدي مع البوت
# ------------------------------------------------------------------
class GymBattleEngine:
    def __init__(self, target_reps=10, depth_angle=75, top_angle=160):
        self.target_reps = target_reps
        self.depth_angle = depth_angle
        self.top_angle = top_angle
        self.stats = {"performed": 0, "accepted": 0, "failed": 0, "review": 0}
        self.bot_reps = 0
        self.bot_next = time.time() + 3.0
        self.stage = "up"
        self.min_angle = 180.0

    def update_frame(self, persons, now):
        last_msg, msg_color = "ابدأ التمرين!", (200, 200, 200)
        main_angle = 180.0
        
        for p in persons:
            s, e, wpt = get_arm(p)
            angle = calculate_angle(s, e, wpt)
            main_angle = angle
            
            if angle > self.top_angle:
                self.stage = "up"
            if angle < self.depth_angle and self.stage == "up":
                self.stage = "down"
                self.min_angle = angle
            if self.stage == "down":
                self.min_angle = min(self.min_angle, angle)
            if angle > self.top_angle and self.stage == "down":
                self.stats["performed"] += 1
                if self.min_angle <= self.depth_angle:
                    self.stats["accepted"] += 1
                    last_msg, msg_color = "تكرار ممتاز!", (46, 204, 113)
                else:
                    self.stats["failed"] += 1
                    last_msg, msg_color = "مرفوض (عمق غير كاف)", (56, 67, 235)
                self.stage = "up"

        # محاكاة أداء البوت في التحدي
        if now >= self.bot_next:
            if self.bot_reps < self.target_reps:
                self.bot_reps += 1
            self.bot_next = now + 3.0 + random.random() * 2.0

        return main_angle, last_msg, msg_color

# ------------------------------------------------------------------
# أدوات الرسم والتنسيق على الفيديو
# ------------------------------------------------------------------
class Painter:
    @staticmethod
    def blend(frame, x1, y1, x2, y2, alpha=0.75):
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), (15, 23, 42), -1)
        return cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0)

def draw_hud(frame, accepted, bot_reps, timer_s, target, battle_active):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = 20, 20, w - 20, 90
    frame = Painter.blend(frame, x1, y1, x2, y2, 0.85)
    
    cv2.putText(frame, f"YOU: {accepted}/{target}", (x1 + 25, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (46, 204, 113), 2, cv2.LINE_AA)
    if battle_active:
        cv2.putText(frame, f"BOT: {bot_reps}", (x2 - 140, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (56, 67, 235), 2, cv2.LINE_AA)
    cv2.putText(frame, f"TIME: {timer_s}s", (w // 2 - 50, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # شريط التقدم للتحدي (أخضر يتحرك مع الأداء)
    by = y2 - 18
    bw = x2 - x1 - 50
    cv2.rectangle(frame, (x1 + 25, by), (x1 + 25 + bw, by + 8), (50, 60, 70), -1)
    pw = int(bw * min(1.0, accepted / max(1, target)))
    if pw:
        cv2.rectangle(frame, (x1 + 25, by), (x1 + 25 + pw, by + 8), (46, 204, 113), -1)
    return frame

def draw_right_panel(frame, stats, last_msg, msg_color):
    h, w = frame.shape[:2]
    pw = 250
    x1, y1, x2, y2 = w - pw - 15, int(h * 0.22), w - 15, int(h * 0.58)
    frame = Painter.blend(frame, x1, y1, x2, y2, 0.85)
    cx = x2 - 15
    row = y1 + 18
    frame = put_ar(frame, "لوحة الأداء الذكية", cx, row, 19, (255, 255, 255))
    row += 35
    frame = put_ar(frame, f"المقبولة: {stats['accepted']}", cx, row, 17, (46, 204, 113))
    row += 28
    frame = put_ar(frame, f"الفاشلة: {stats['failed']}", cx, row, 17, (56, 67, 235))
    row += 28
    frame = put_ar(frame, f"المراجعة: {stats['review']}", cx, row, 17, (58, 178, 247))
    row += 35
    frame = put_ar(frame, last_msg, cx, row, 15, msg_color)
    return frame

def draw_watermark(frame):
    h, w = frame.shape[:2]
    cv2.putText(frame, "WF.DIETNCHEAT.COM", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "Eng: Rafat Al-Wajeh", (w - 220, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return frame

# ------------------------------------------------------------------
# واجهة Streamlit وتنسيقات التصميم
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

st.sidebar.title("🏋️‍♂️ مركز القيادة والتحكم الحي")
st.sidebar.markdown("---")
mode = st.sidebar.radio("وضع التشغيل:", ["📤 رفع فيديو للمعالجة الذكية", "📸 لقطة كاميرا (معالجة فورية مباشرة)"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ بارامترات الأداء والتحدي")
target_reps = st.sidebar.slider("التكرارات المطلوبة:", 5, 30, 10)
depth_angle = st.sidebar.slider("زاوية العمق:", 50, 110, 75)
battle_mode = st.sidebar.checkbox("⚔️ تفعيل وضع التحدي مع البوت", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Eng : Rafat Al-Wajeh**  \n[WF.DIETNCHEAT.COM](https://wf.dietncheat.com)")

st.markdown("""
<div class="hero"><h2>🛡️ Gym Enterprise AI — منصة المدرب الذكي الاحترافية</h2>
<p>تحليل تماثل المفاصل • عداد تكرارات تفاعلي • وضع التحدي الحي مع البوت.</p></div>
""", unsafe_allow_html=True)

if mode.startswith("📤"):
    st.subheader("📤 معالجة فيديو منظمة مع عدادات الأداء والتحدي")
    up = st.file_uploader("ارفع فيديو التمرين (MP4/AVI/MOV):", type=["mp4", "avi", "mov"])
    
    if up is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(up.read())
        tfile.close()
        st.video(tfile.name)
        
        if st.button("🚀 بدء المعالجة الذكية وتحليل التكرارات"):
            engine = GymBattleEngine(target_reps, depth_angle)
            progress = st.progress(0)
            status = st.empty()
            live = st.empty()
            
            cap = cv2.VideoCapture(tfile.name)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            
            out_path = os.path.join(OUT_DIR, f"gym_battle_{int(time.time())}.mp4")
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            
            idx, shown, t0 = 0, 0, time.time()
            last_msg, msg_color = "ابدأ التمرين", (200, 200, 200)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1
                
                annotated, persons = detect_persons(frame)
                now = time.time()
                main_angle, last_msg, msg_color = engine.update_frame(persons, now)
                
                annotated = draw_hud(annotated, engine.stats["accepted"], engine.bot_reps, int(now - t0), target_reps, battle_mode)
                annotated = draw_right_panel(annotated, engine.stats, last_msg, msg_color)
                annotated = draw_watermark(annotated)
                
                out.write(annotated)
                if shown % 4 == 0:
                    live.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                shown += 1
                progress.progress(min(1.0, idx / total))
                
            cap.release()
            out.release()
            
            with open(out_path, "rb") as f:
                processed = f.read()
            st.session_state["processed"] = processed
            st.session_state["stats"] = engine.stats
            status.success("✅ تمت المعالجة بنجاح ورصد التكرارات بالكامل!")
            
        if "processed" in st.session_state:
            stats = st.session_state["stats"]
            st.markdown("### 📊 تقرير الأداء الختامي:")
            c = st.columns(4)
            c[0].metric("المحاولات", stats["performed"])
            c[1].metric("المقبولة ✅", stats["accepted"])
            c[2].metric("الفاشلة ❌", stats["failed"])
            c[3].metric("المراجعة ⚠️", stats["review"])
            st.download_button("📥 تنزيل الفيديو المعالج MP4", data=st.session_state["processed"], file_name="gym_battle_output.mp4", mime="video/mp4")
else:
    st.subheader("📸 تحليل لقطة فورية (معالجة مباشرة)")
    st.info("💡 التقط صورة واضحة لجسمك أو ذراعك أمام الكاميرا لتحليل الزاوية فوراً.")
    
    shot = st.camera_input("التقط صورة بوضعية التمرين:")
    if shot is not None:
        data = np.frombuffer(shot.getvalue(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        annotated, persons = detect_persons(frame)
        annotated = draw_watermark(annotated)
        
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        if persons:
            s, e, wpt = get_arm(persons[0])
            angle = calculate_angle(s, e, wpt)
            verdict = "ممتاز (مستوفي العمق)" if angle <= depth_angle else "غير كامل / يحتاج نزول أكثر"
            st.success("✅ تم تحليل اللقطة ورصد المفاصل بنجاح تام!")
            st.metric("زاوية الكوع المقاسة", f"{int(angle)}°", delta=f"التصنيف: {verdict}")
        else:
            st.success("✅ تم التقاط الصورة وعرضها بنجاح!")
            st.warning("⚠️ تأكد من ظهور الجسم أو الذراع بوضوح أمام العدسة.")
