# -*- coding: utf-8 -*-
"""
====================================================================
Gym Enterprise AI — v3.0 | المنصة الذكية الكاملة
Prepared by Eng : Rafat Al-Wajeh (Rafat Hazza Qasim)
====================================================================
• أوضاع التشغيل: رفع فيديو | كاميرا مباشرة (webrtc) | لقطة كاميرا
• محركان: YOLOv8-Pose (متعدد الأشخاص + تتبع) | MediaPipe (أسرع، شخص واحد)
• رسم دقيق على الفيديو: شريط معركة، نص تحدي، مؤشر زاوية، عدادات عربية
• صرامة كاملة: تحقق من المدخلات، كشف الأعطال الصامتة، تنزيل مضمون
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
# دعم العربية على الفيديو
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


def put_ar(frame, text, x, y, size=26, color=(255, 255, 255)):
    from PIL import Image, ImageDraw, ImageFont
    s = int(size)
    if s not in _font_cache:
        path = ensure_font()
        try:
            _font_cache[s] = ImageFont.truetype(path, s) if path \
                else ImageFont.loadDefault()
        except Exception:
            _font_cache[s] = ImageFont.loadDefault()
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img).text((x, y), ar(text), font=_font_cache[s],
                             fill=color[::-1])
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------------
# المحركات: YOLOv8 (تتبع متعدد) / MediaPipe (شخص واحد سريع)
# ------------------------------------------------------------------
@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n-pose.pt")


@st.cache_resource
def load_mediapipe():
    import mediapipe as mp
    return mp.solutions.pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)


def detect_persons(frame, engine):
    """يعيد قائمة موحّدة: {id, shoulder, elbow, wrist, conf, draw_fn}"""
    h, w = frame.shape[:2]
    persons = []
    if engine == "yolo":
        res = load_yolo().track(frame, persist=True, verbose=False)[0]
        if res.keypoints is None or res.boxes.id is None:
            return res.plot(boxes=False, labels=False), persons
        ids = res.boxes.id.cpu().numpy()
        kps = res.keypoints.xy.cpu().numpy()
        confs = res.keypoints.conf
        confs = confs.cpu().numpy() if confs is not None \
            else np.ones((len(kps), 17))
        annotated = res.plot(boxes=False, labels=False)
        for i, u_id in enumerate(ids):
            kp = kps[i]
            if len(kp) < 10:
                continue
            persons.append({
                "id": int(u_id), "kps": kp,
                "conf": float(np.nanmean(confs[i])), "engine": "yolo"})
        return annotated, persons

    # MediaPipe
    import mediapipe as mp
    pose = load_mediapipe()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = pose.process(rgb)
    annotated = frame.copy()
    if r.pose_landmarks:
        lm = r.pose_landmarks.landmark
        def pt(i):
            return np.array([lm[i].x * w, lm[i].y * h])
        conf = float(np.mean([lm[i].visibility for i in (11, 13, 15)]))
        persons.append({"id": 0, "kps": {"s": pt(11), "e": pt(13),
                                         "w": pt(15)}, "conf": conf,
                        "engine": "mp"})
        # هيكل عظمي أخضر سميك (مطابق للّقطات)
        for a, b in mp.solutions.pose.POSE_CONNECTIONS:
            pa = (int(lm[a].x * w), int(lm[a].y * h))
            pb = (int(lm[b].x * w), int(lm[b].y * h))
            cv2.line(annotated, pa, pb, (0, 255, 60), 4, cv2.LINE_AA)
        for i in (11, 13, 15):
            cv2.circle(annotated, (int(lm[i].x * w), int(lm[i].y * h)),
                       6, (0, 200, 255), -1)
    return annotated, persons


def get_arm(person):
    """نقاط الذراع: YOLO keypoints 5/7/9 — MediaPipe s/e/w"""
    if person["engine"] == "yolo":
        kp = person["kps"]
        return kp[5][:2], kp[7][:2], kp[9][:2]
    return person["kps"]["s"], person["kps"]["e"], person["kps"]["w"]


def calculate_angle(a, b, c):
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
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
        return {"stage": "up", "min_angle": 180.0,
                "rep_start": None, "conf_sum": 0.0, "conf_n": 0}

    def close_rep(self, rep, now):
        duration = (now - rep["rep_start"]) if rep["rep_start"] else 99.0
        depth_ok = rep["min_angle"] <= self.depth_angle
        tempo_ok = 0.5 <= duration <= 8.0
        avg_conf = (rep["conf_sum"] / rep["conf_n"]) if rep["conf_n"] else 0.0
        conf_ok = avg_conf >= 0.5
        if depth_ok and tempo_ok and conf_ok:
            return "accepted", "تكرار ممتاز!"
        if (not conf_ok) or \
           (self.depth_angle < rep["min_angle"] <= self.depth_angle + 15):
            return "review", "قيد المراجعة (رؤية ضعيفة)"
        return "failed", "مرفوض (نزول غير كامل)"


# ------------------------------------------------------------------
# الرسم الدقيق على الفيديو (مطابق لأسلوب اللقطات)
# ------------------------------------------------------------------
class Painter:
    WHITE = (255, 255, 255)
    GRAY = (200, 200, 200)
    DARK = (18, 24, 32)
    GREEN = (46, 204, 113)
    RED = (56, 67, 235)
    ORANGE = (38, 130, 250)
    AMBER = (58, 178, 247)
    BLUE = (245, 160, 30)

    @staticmethod
    def blend(frame, x1, y1, x2, y2, alpha=0.70):
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), Painter.DARK, -1)
        return cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0)

    @staticmethod
    def avatar(frame, cx, cy, r, name, color):
        cv2.circle(frame, (cx, cy), r, color, 3)
        cv2.circle(frame, (cx, cy), r - 3, (40, 48, 58), -1)
        initial = name.strip("@")[:1].upper() if name.strip("@") else "P"
        cv2.putText(frame, initial, (cx - 12, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, Painter.WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, name, (cx - 60, cy + r + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, Painter.WHITE, 2, cv2.LINE_AA)


def draw_battle_header(frame, player_name, bot_name, accepted, bot_reps,
                       timer_s, target):
    h, w = frame.shape[:2]
    m = max(16, int(w * 0.035))
    x1, y1, x2, y2 = m, m, w - m, m + int(h * 0.11)
    frame = Painter.blend(frame, x1, y1, x2, y2, 0.78)
    cy = y1 + int((y2 - y1) * 0.42)
    # الصور الرمزية والأسماء
    Painter.avatar(frame, x1 + 70, cy - 10, 34, player_name, Painter.GREEN)
    Painter.avatar(frame, x2 - 70, cy - 10, 34, bot_name, Painter.RED)
    # النتائج الكبيرة
    cv2.putText(frame, str(accepted), (x1 + 150, cy + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, Painter.GREEN, 4, cv2.LINE_AA)
    cv2.putText(frame, str(bot_reps), (x2 - 210, cy + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, Painter.RED, 4, cv2.LINE_AA)
    # الوقت بالوسط
    cv2.putText(frame, "TIME", (w // 2 - 45, y1 + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, Painter.GRAY, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{timer_s}s", (w // 2 - 62, y1 + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, Painter.WHITE, 3, cv2.LINE_AA)
    # شريط التنافس الأخضر/الأحمر
    by = y2 - 42
    bw = x2 - x1 - 40
    cv2.rectangle(frame, (x1 + 20, by), (x1 + 20 + bw, by + 18),
                  (70, 78, 88), -1)
    pw = int(bw * min(1.0, accepted / max(1, target)))
    bwid = int(bw * min(1.0, bot_reps / max(1, target)))
    if pw:
        cv2.rectangle(frame, (x1 + 20, by), (x1 + 20 + pw, by + 18),
                      Painter.GREEN, -1)
    if bwid:
        cv2.rectangle(frame, (x2 - 20 - bwid, by), (x2 - 20, by + 18),
                      Painter.RED, -1)
    return frame


def draw_challenge_text(frame, text):
    h, w = frame.shape[:2]
    cv2.putText(frame, text, (w // 2 - int(len(text) * 15), int(h * 0.185)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, Painter.WHITE, 5, cv2.LINE_AA)
    cv2.putText(frame, text, (w // 2 - int(len(text) * 15), int(h * 0.185)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
    return frame


def draw_left_gauge(frame, angle, depth_angle):
    h, w = frame.shape[:2]
    bar_x = max(20, int(w * 0.055))
    y1, y2 = int(h * 0.26), int(h * 0.82)
    seg = y2 - y1
    r_end = y1 + int(seg * 0.40)
    b_end = r_end + int(seg * 0.25)
    cv2.rectangle(frame, (bar_x, y1), (bar_x + 26, r_end), Painter.RED, -1)
    cv2.rectangle(frame, (bar_x, r_end), (bar_x + 26, b_end), Painter.BLUE, -1)
    cv2.rectangle(frame, (bar_x, b_end), (bar_x + 26, y2), Painter.GREEN, -1)
    cv2.rectangle(frame, (bar_x, y1), (bar_x + 26, y2), (0, 0, 0), 1)
    ind_y = int(y2 - (min(angle, 180) / 180.0) * seg)
    cv2.circle(frame, (bar_x + 13, ind_y), 16, Painter.WHITE, -1)
    cv2.circle(frame, (bar_x + 13, ind_y), 16, (0, 0, 0), 2)
    return frame


def draw_right_counters(frame, stats, target, last_msg, msg_color):
    h, w = frame.shape[:2]
    pw = min(330, int(w * 0.33))
    x1, y1, x2, y2 = w - pw - 14, int(h * 0.26), w - 14, int(h * 0.78)
    frame = Painter.blend(frame, x1, y1, x2, y2, 0.72)
    cx = x2 - 16
    row = y1 + 12
    frame = put_ar(frame, "لوحة الأداء الذكية", cx, row, 24,
                   Painter.WHITE)
    row += 40
    frame = put_ar(frame, f"المطلوبة: {target}", cx, row, 20, Painter.GRAY)
    row += 30
    frame = put_ar(frame, f"المنفَّذة: {stats['performed']}", cx, row, 30,
                   Painter.WHITE)
    row += 44
    for label, val, color in [("المقبولة", stats["accepted"], Painter.GREEN),
                              ("المرفوضة", stats["failed"], Painter.RED),
                              ("قيد المراجعة", stats["review"], Painter.AMBER)]:
        cv2.circle(frame, (cx - 14, row + 11), 6, color, -1)
        frame = put_ar(frame, f"{label}: {val}", cx - 32, row, 20,
                       Painter.WHITE)
        row += 34
    row += 8
    frame = put_ar(frame, last_msg, cx, row, 19, msg_color)
    return frame


def draw_watermark(frame):
    h, w = frame.shape[:2]
    cv2.putText(frame, "WF.DIETNCHEAT.COM", (w // 2 - 300, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, Painter.WHITE, 4, cv2.LINE_AA)
    cv2.putText(frame, "WF.DIETNCHEAT.COM", (w // 2 - 300, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "Prepared by Eng : Rafat Al-Wajeh", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, Painter.GRAY, 1, cv2.LINE_AA)
    return frame


# ------------------------------------------------------------------
# حلقة التتبع والعدّ المشتركة
# ------------------------------------------------------------------
class Tracker:
    def __init__(self, depth_angle, top_angle=160):
        self.qe = RepQualityEngine(depth_angle, top_angle)
        self.users = {}
        self.stats = {"performed": 0, "accepted": 0,
                      "failed": 0, "review": 0}
        self.bot_reps = 0
        self.bot_next = time.time() + 3.0

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
                msg_color = {"accepted": Painter.GREEN, "failed": Painter.RED,
                             "review": Painter.AMBER}[verdict]
                self.users[u_id]["rep"] = self.qe.new_rep()
        # محاكاة البوت
        if now >= self.bot_next:
            self.bot_reps += 1
            import random
            self.bot_next = now + 2.5 + random.random() * 2.0
        return main_angle, last_msg, msg_color


# ------------------------------------------------------------------
# واجهة المستخدم — CSS
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

st.sidebar.title("🏋️‍♂️ مركز القيادة")
st.sidebar.markdown("---")
mode = st.sidebar.radio("وضع التشغيل:",
                        ["📤 رفع فيديو للمعالجة", "🎥 كاميرا مباشرة",
                         "📸 لقطة كاميرا (صورة)"])
engine = st.sidebar.radio("محرك التحليل:",
                          ["YOLOv8 (معركة متعددة الأشخاص)",
                           "MediaPipe (شخص واحد — أسرع)"])
engine_key = "yolo" if engine.startswith("YOLO") else "mp"

st.sidebar.markdown("---")
target_reps = st.sidebar.slider("التكرارات المطلوبة:", 5, 30, 10)
depth_angle = st.sidebar.slider("زاوية العمق:", 50, 110, 75)
challenge = st.sidebar.text_input("نص التحدي على الفيديو:",
                                  "Bro thinks this is easy")
player_name = st.sidebar.text_input("اسم اللاعب:", "@you")
bot_name = st.sidebar.text_input("اسم الخصم (البوت):", "AI-Bot")
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Eng : Rafat Al-Wajeh**  \n[WF.DIETNCHEAT.COM](https://wf.dietncheat.com)")

st.markdown("""
<div class="hero"><h2>🛡️ Gym Enterprise AI — النظام الذكي لتحليل الأداء</h2>
<p>معالجة بالذكاء الاصطناعي • تصنيف مقبول/مرفوض/مراجعة • فيديو معالج بتعليقات احترافية • تنزيل مضمون.</p></div>
""", unsafe_allow_html=True)


def report_and_download(stats, target, processed_bytes=None, path=None):
    st.markdown("### 📊 تقرير الأداء الختامي:")
    c = st.columns(5)
    c[0].metric("الهدف", target)
    c[1].metric("المحاولات", stats["performed"])
    c[2].metric("المقبولة ✅", stats["accepted"])
    c[3].metric("المرفوضة ❌", stats["failed"])
    c[4].metric("المراجعة ⚠️", stats["review"])
    rate = int(100 * stats["accepted"] / max(1, stats["performed"]))
    st.progress(rate / 100)
    st.markdown(f"**نسبة النجاح: {rate}%**")
    if processed_bytes:
        st.markdown("### 📥 تنزيل الفيديو المعالج:")
        st.download_button("📥 تنزيل MP4", data=processed_bytes,
                           file_name="gym_processed.mp4", mime="video/mp4")
        st.caption(f"الحجم: {len(processed_bytes)/1024:.0f} KB — جاهز للتشغيل في أي مشغّل")


# ==================================================================
# الوضع 1: رفع فيديو
# ==================================================================
if mode.startswith("📤"):
    st.subheader("📤 معالجة فيديو مسجّل")
    up = st.file_uploader("ارفع فيديو (MP4/AVI/MOV — بحد أقصى 200MB):",
                          type=["mp4", "avi", "mov"])

    if up is not None:
        data = up.read()
        if len(data) > 200 * 1024 * 1024:
            st.error("⛔ الفيديو أكبر من 200MB — ارفع ملفاً أصغر. (معالجة بحزم)")
            st.stop()
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(data)
        tfile.close()
        st.video(tfile.name)

        if st.button("🚀 بدء المعالجة الصارمة"):
            tracker = Tracker(depth_angle)
            progress = st.progress(0)
            status = st.empty()
            live = st.empty()
            try:
                cap = cv2.VideoCapture(tfile.name)
                if not cap.isOpened():
                    st.error("⛔ تعذر فتح الفيديو — الملف تالف أو غير مدعوم.")
                    st.stop()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

                out_path = os.path.join(OUT_DIR,
                                        f"gym_{int(time.time())}.mp4")
                out = cv2.VideoWriter(out_path,
                                      cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, (w, h))
                if not out.isOpened():
                    st.error("⛔ فشل إنشاء ملف الإخراج — تحقق من مساحة القرص.")
                    st.stop()

                status.text("⚙️ معالجة إطاراً بإطار…")
                idx, shown, t0 = 0, 0, time.time()
                last_msg, msg_color = "ابدأ!", Painter.GRAY
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    idx += 1
                    annotated, persons = detect_persons(frame, engine_key)
                    now = time.time()
                    main_angle, last_msg, msg_color = tracker.step(persons, now)
                    annotated = draw_battle_header(annotated, player_name,
                                                   bot_name,
                                                   tracker.stats["accepted"],
                                                   tracker.bot_reps,
                                                   int(now - t0), target_reps)
                    annotated = draw_challenge_text(annotated, challenge)
                    annotated = draw_left_gauge(annotated, main_angle,
                                                depth_angle)
                    annotated = draw_right_counters(annotated, tracker.stats,
                                                    target_reps, last_msg,
                                                    msg_color)
                    annotated = draw_watermark(annotated)
                    out.write(annotated)
                    if shown % 4 == 0:
                        live.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                   channels="RGB", use_container_width=True)
                    shown += 1
                    progress.progress(min(1.0, idx / total))

                cap.release()
                out.release()

                # ✦ التنزيل المضمون: قراءة البايتات والتحقق من الحجم
                if not (os.path.exists(out_path)
                        and os.path.getsize(out_path) > 5000):
                    st.error("⛔ فشلت كتابة الفيديو الناتج (ملف فارغ). "
                             "جرّب ترميز 'avc1' أو تحقق من الأذونات.")
                    st.stop()
                with open(out_path, "rb") as f:
                    processed = f.read()
                st.session_state["processed"] = processed
                st.session_state["stats"] = tracker.stats
                status.success("✅ اكتملت المعالجة بنجاح!")
            except Exception:
                st.error("⛔ عطل أثناء المعالجة — التفاصيل:")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

        if "processed" in st.session_state:
            report_and_download(st.session_state["stats"], target_reps,
                                processed_bytes=st.session_state["processed"])

# ==================================================================
# الوضع 2: كاميرا مباشرة (webrtc)
# ==================================================================
elif mode.startswith("🎥"):
    st.subheader("🎥 كاميرا مباشرة عبر المتصفح")
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, \
            RTCConfiguration
        import av
    except ImportError:
        st.warning("⚠️ لم يتم تثبيت مكتبة الكاميرا المباشرة. نفّذ مرة واحدة:")
        st.code("pip install streamlit-webrtc")
        st.stop()

    class LiveProcessor(VideoProcessorBase):
        def __init__(self):
            self.tracker = Tracker(depth_angle)
            self.t0 = time.time()

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            annotated, persons = detect_persons(img, engine_key)
            now = time.time()
            main_angle, last_msg, msg_color = self.tracker.step(persons, now)
            annotated = draw_battle_header(annotated, player_name, bot_name,
                                           self.tracker.stats["accepted"],
                                           self.tracker.bot_reps,
                                           int(now - self.t0), target_reps)
            annotated = draw_challenge_text(annotated, challenge)
            annotated = draw_left_gauge(annotated, main_angle, depth_angle)
            annotated = draw_right_counters(annotated, self.tracker.stats,
                                            target_reps, last_msg, msg_color)
            annotated = draw_watermark(annotated)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    ctx = webrtc_streamer(
        key="gym-live",
        video_processor_factory=LiveProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True)
    st.caption("اضغط START لبدء البث — المعالجة تتم داخل المتصفح بثبات.")

# ==================================================================
# الوضع 3: لقطة كاميرا (صورة)
# ==================================================================
else:
    st.subheader("📸 تحليل لقطة فورية")
    shot = st.camera_input("التقط صورة بوضعية التمرين:")
    if shot is not None:
        data = np.frombuffer(shot.getvalue(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        annotated, persons = detect_persons(frame, engine_key)
        if persons:
            s, e, wpt = get_arm(persons[0])
            angle = calculate_angle(s, e, wpt)
            annotated = draw_left_gauge(annotated, angle, depth_angle)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     channels="RGB", use_container_width=True)
            verdict = "ممتاز" if angle <= depth_angle else \
                ("مقارب" if angle <= depth_angle + 30 else "غير كامل")
            st.metric("زاوية الكوع المقاسة", f"{int(angle)}°",
                      delta=f"التصنيف: {verdict}")
        else:
            st.warning("لم يُكتشف أي شخص — أعد التقاط اللقطة بوضوح أعلى.")
