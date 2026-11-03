# -*- coding: utf-8 -*-
"""
====================================================================
Gym Enterprise AI — v3.5 | النسخة النهائية المستقرة
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
# محرك الذكاء الاصطناعي YOLOv8
# ------------------------------------------------------------------
@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n-pose.pt")

def detect_persons(frame):
    """استخراج المفاصل والهيكل العظمي باستخدام YOLOv8 بدقة تامة"""
    h, w = frame.shape[:2]
    persons = []
    res = load_yolo().track(frame, persist=True, verbose=False)[0]
    if res.keypoints is None or res.boxes is None:
        return frame.copy(), persons
    
    kps_all = res.keypoints.xy.cpu().numpy() if res.keypoints.xy is not None else []
    confs_all = res.keypoints.conf
    confs_all = confs_all.cpu().numpy() if confs_all is not None else np.ones((len(kps_all), 17))
    
    annotated = res.plot(boxes=False, labels=False)
    
    for i, kp in enumerate(kps_all):
        if len(kp) < 10:
            continue
        conf = float(np.nanmean(confs_all[i])) if i < len(confs_all) else 0.8
        persons.append({
            "id": i,
            "kps": kp,
            "conf": conf
        })
    return annotated, persons

def get_arm(person):
    kp = person["kps"]
    # الكتف (5)، المرفق (7)، الرسغ (9) في نموذج YOLOv8-Pose
    return kp[5][:2], kp[7][:2], kp[9][:2]

def calculate_angle(a, b, c):
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# ------------------------------------------------------------------
# محرك جودة التكرار
# ------------------------------------------------------------------
class RepQualityEngine:
    def __init__(self, depth_angle=75, top_angle=160):
        self.depth_angle = depth_angle
        self.top_angle = top_angle

    def new_rep(self):
        return {"stage": "up", "min_angle": 180.0, "rep_start": None, "conf_sum": 0.0, "conf_n": 0}

    def close_rep(self, rep, now):
        duration = (now - rep["rep_start"]) if rep["rep_start"] else 99.0
        depth_ok = rep["min_angle"] <= self.depth_angle
        tempo_ok = 0.5 <= duration <= 8.0
        avg_conf = (rep["conf_sum"] / rep["conf_n"]) if rep["conf_n"] else 0.0
        conf_ok = avg_conf >= 0.5
        if depth_ok and tempo_ok and conf_ok:
            return "accepted", "تكرار ممتاز!"
        if (not conf_ok) or (self.depth_angle < rep["min_angle"] <= self.depth_angle + 15):
            return "review", "قيد المراجعة"
        return "failed", "مرفوض (نزول غير كامل)"

# ------------------------------------------------------------------
# أدوات الرسم النظيف والمرتب
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

def draw_clean_header(frame, accepted, bot_reps, timer_s, target):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = 20, 20, w - 20, 90
    frame = Painter.blend(frame, x1, y1, x2, y2, 0.85)
    cv2.putText(frame, f"YOU: {accepted}/{target}", (x1 + 25, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, Painter.GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, f"BOT: {bot_reps}", (x2 - 140, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, Painter.RED, 2, cv2.LINE_AA)
    cv2.putText(frame, f"TIME: {timer_s}s", (w // 2 - 50, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, Painter.WHITE, 2, cv2.LINE_AA)
    by = y2 - 18
    bw = x2 - x1 - 50
    cv2.rectangle(frame, (x1 + 25, by), (x1 + 25 + bw, by + 8), (50, 60, 70), -1)
    pw = int(bw * min(1.0, accepted / max(1, target)))
    if pw:
        cv2.rectangle(frame, (x1 + 25, by), (x1 + 25 + pw, by + 8), Painter.GREEN, -1)
    return frame

def draw_left_gauge(frame, angle, depth_angle):
    h, w = frame.shape[:2]
    bar_x, y1, y2 = 25, int(h * 0.25), int(h * 0.75)
    seg = y2 - y1
    cv2.rectangle(frame, (bar_x, y1), (bar_x + 18, y2), (40, 50, 60), -1)
    ind_y = int(y2 - (min(angle, 180) / 180.0) * seg)
    cv2.circle(frame, (bar_x + 9, ind_y), 12, Painter.WHITE, -1)
    cv2.circle(frame, (bar_x + 9, ind_y), 12, (0, 0, 0), 2)
    return frame

def draw_right_counters(frame, stats, last_msg, msg_color):
    h, w = frame.shape[:2]
    pw = 240
    x1, y1, x2, y2 = w - pw - 15, int(h * 0.25), w - 15, int(h * 0.60)
    frame = Painter.blend(frame, x1, y1, x2, y2, 0.80)
    cx = x2 - 15
    row = y1 + 15
    frame = put_ar(frame, "لوحة الأداء الذكية", cx, row, 20, Painter.WHITE)
    row += 35
    frame = put_ar(frame, f"المقبولة: {stats['accepted']}", cx, row, 18, Painter.GREEN)
    row += 28
    frame = put_ar(frame, f"المرفوضة: {stats['failed']}", cx, row, 18, Painter.RED)
    row += 28
    frame = put_ar(frame, f"المراجعة: {stats['review']}", cx, row, 18, Painter.AMBER)
    row += 35
    frame = put_ar(frame, last_msg, cx, row, 16, msg_color)
    return frame

def draw_watermark(frame):
    h, w = frame.shape[:2]
    cv2.putText(frame, "WF.DIETNCHEAT.COM", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, Painter.GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, "Eng: Rafat Al-Wajeh", (w - 220, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Painter.GRAY, 1, cv2.LINE_AA)
    return frame

# ------------------------------------------------------------------
# حلقة التتبع مع التحديث الحي للمتغيرات
# ------------------------------------------------------------------
class Tracker:
    def __init__(self, depth_angle, top_angle=160):
        self.qe = RepQualityEngine(depth_angle, top_angle)
        self.users = {}
        self.stats = {"performed": 0, "accepted": 0, "failed": 0, "review": 0}
        self.bot_reps = 0
        self.bot_next = time.time() + 3.0

    def update_params(self, depth_angle):
        self.qe.depth_angle = depth_angle

    def step(self, persons, now):
        last_msg, msg_color = "ابدأ التمرين!", Painter.GRAY
        main_angle = 180.0
        for i, p in enumerate(persons):
            s, e, wpt = get_arm(p)
            angle = calculate_angle(s, e, wpt)
            if i == 0:
                main_angle = angle
            u_id = p["id"]
            if u_id not in self.users:
                self.users[u_id] = {"rep": self.qe.new_rep(), "ok": 0}
            rep = self.users[u_id]["rep"]
            if angle > self.qe.top_angle:
                rep["stage"] = "up"
            if angle < self.qe.depth_angle and rep["stage"] == "up":
                rep["stage"] = "down"
                rep["rep_start"] = now
                rep["min_angle"] = angle
            if rep["stage"] == "down":
                rep["min_angle"] = min(rep["min_angle"], angle)
                rep["conf_sum"] += p["conf"]
                rep["conf_n"] += 1
            if angle > self.qe.top_angle and rep["stage"] == "down":
                verdict, last_msg = self.qe.close_rep(rep, now)
                self.stats["performed"] += 1
                self.stats[verdict] += 1
                if verdict == "accepted":
                    self.users[u_id]["ok"] += 1
                msg_color = {"accepted": Painter.GREEN, "failed": Painter.RED, "review": Painter.AMBER}[verdict]
                self.users[u_id]["rep"] = self.qe.new_rep()
        if now >= self.bot_next:
            self.bot_reps += 1
            import random
            self.bot_next = now + 3.0 + random.random() * 2.0
        return main_angle, last_msg, msg_color

# ------------------------------------------------------------------
# واجهة Streamlit وتنسيقات CSS
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
mode = st.sidebar.radio("وضع التشغيل:", ["📤 رفع فيديو للمعالجة الحية", "📸 لقطة كاميرا (معالجة فورية مباشرة)"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ بارامترات الأداء الحية (تعديل مباشر)")
target_reps = st.sidebar.slider("التكرارات المطلوبة:", 5, 30, 10, key="live_target")
depth_angle = st.sidebar.slider("زاوية العمق:", 50, 110, 75, key="live_depth")
challenge = st.sidebar.text_input("نص التحدي على الفيديو:", "Gym Enterprise AI Master", key="live_challenge")

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Eng : Rafat Al-Wajeh**  \n[WF.DIETNCHEAT.COM](https://wf.dietncheat.com)")

st.markdown("""
<div class="hero"><h2>🛡️ Gym Enterprise AI — النظام الذكي المستقر</h2>
<p>تحكم حي بالبارامترات • عرض منظم ونظيف • معالجة فورية بالاعتماد على YOLOv8.</p></div>
""", unsafe_allow_html=True)

if mode.startswith("📤"):
    st.subheader("📤 معالجة فيديو منظمة مع تحكم حي بالبارامترات")
    up = st.file_uploader("ارفع فيديو التمرين (MP4/AVI/MOV):", type=["mp4", "avi", "mov"])
    
    if up is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(up.read())
        tfile.close()
        st.video(tfile.name)
        
        if st.button("🚀 بدء المعالجة الحية المنظمة"):
            tracker = Tracker(depth_angle)
            progress = st.progress(0)
            status = st.empty()
            live = st.empty()
            
            cap = cv2.VideoCapture(tfile.name)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            
            out_path = os.path.join(OUT_DIR, f"gym_clean_{int(time.time())}.mp4")
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            
            idx, shown, t0 = 0, 0, time.time()
            last_msg, msg_color = "ابدأ التمرين", Painter.GRAY
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1
                
                tracker.update_params(depth_angle)
                
                annotated, persons = detect_persons(frame)
                now = time.time()
                main_angle, last_msg, msg_color = tracker.step(persons, now)
                
                annotated = draw_clean_header(annotated, tracker.stats["accepted"], tracker.bot_reps, int(now - t0), target_reps)
                annotated = draw_left_gauge(annotated, main_angle, depth_angle)
                annotated = draw_right_counters(annotated, tracker.stats, last_msg, msg_color)
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
            st.session_state["stats"] = tracker.stats
            status.success("✅ تمت المعالجة بنجاح بالبارامترات المحدثة!")
            
        if "processed" in st.session_state:
            stats = st.session_state["stats"]
            st.markdown("### 📊 تقرير الأداء الختامي:")
            c = st.columns(4)
            c[0].metric("المحاولات", stats["performed"])
            c[1].metric("المقبولة ✅", stats["accepted"])
            c[2].metric("المرفوضة ❌", stats["failed"])
            c[3].metric("المراجعة ⚠️", stats["review"])
            st.download_button("📥 تنزيل الفيديو المنظم MP4", data=st.session_state["processed"], file_name="gym_clean_output.mp4", mime="video/mp4")
else:
    st.subheader("📸 تحليل لقطة فورية (معالجة مباشرة بمجرد الالتقاط)")
    st.info("💡 بمجرد التقاط الصورة، يقوم النظام فوراً باستخراج المفاصل ورسم الهيكل وعرض الزاوية.")
    
    shot = st.camera_input("التقط صورة بوضعية التمرين:")
    if shot is not None:
        data = np.frombuffer(shot.getvalue(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        # معالجة فورية مباشرة باستخدام YOLOv8 المستقر
        annotated, persons = detect_persons(frame)
        
        if persons:
            s, e, wpt = get_arm(persons[0])
            angle = calculate_angle(s, e, wpt)
            annotated = draw_left_gauge(annotated, angle, depth_angle)
            annotated = draw_watermark(annotated)
            
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            verdict = "ممتاز (مستوفي العمق)" if angle <= depth_angle else "غير كامل / يحتاج نزول أكثر"
            st.success("✅ تمت المعالجة الفورية المباشرة للقطة بنجاح تام!")
            st.metric("زاوية الكوع المقاسة", f"{int(angle)}°", delta=f"التصنيف: {verdict}")
        else:
            annotated = draw_watermark(frame)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            st.success("✅ تم التقاط الصورة وعرضها بنجاح!")
            st.warning("⚠️ تأكد من ظهور الذراع والجسم بوضوح أمام الكاميرا لرسم الهيكل العظمي بدقة.")
