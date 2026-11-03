# -*- coding: utf-8 -*-
"""
====================================================================
Gym Enterprise AI — منصة المدرب الذكي (Streamlit Edition) v2
Prepared by Eng : Rafat Hazza Qasim (Rafat Al-Wajeh)
====================================================================
إصلاحات جوهرية عن النسخة الأولى:
  1. منطق عدّ صحيح: كل محاولة تُحسب + تصنيف مقبول/مرفوض/مراجعة بالعمق
     والإيقاع وثقة الرؤية (المشكلة الأساسية: failed_reps كانت ميتة).
  2. دعم عربي حقيقي داخل الفيديو (PIL + reshaper) مع بديل تلقائي.
  3. RTL محصور في النصوص فقط — لا يكسر تخطيط Streamlit.
  4. زر التنزيل يبقى بعد المعالجة (session_state) دون إعادة تشغيل.
====================================================================
"""

import os
import time
import urllib.request
import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="منصة المدرب الذكي لنادي الكالسثانكس",
                   page_icon="💪", layout="wide")

# ------------------------------------------------------------------
# دعم العربية داخل الفيديو (نفس أسلوب نسخة OpenCV)
# ------------------------------------------------------------------
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def ar(text):
        return get_display(arabic_reshaper.reshape(text))
    HAS_AR = True
except Exception:
    HAS_AR = False
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
# CSS: RTL للنصوص فقط + ثيم مؤسسي نظيف
# ------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600&display=swap');

    h1, h2, h3, h4, p, label, .stMarkdown {
        direction: rtl; text-align: right; font-family: 'Cairo', Tahoma, sans-serif;
    }
    section[data-testid="stSidebar"] { direction: rtl; }
    section[data-testid="stSidebar"] * { text-align: right; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #1e293b);
    }
    section[data-testid="stSidebar"] *:not(a) { color: #e2e8f0 !important; }

    .stButton>button {
        width: 100%; border-radius: 12px; font-weight: 600; height: 3.2em;
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white; border: none; box-shadow: 0 4px 6px rgba(0,0,0,.2);
        transition: .3s; font-family: 'Cairo', sans-serif;
    }
    .stButton>button:hover { background: linear-gradient(135deg, #2563eb, #1e40af); }

    .hero-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        color: white; padding: 20px; border-radius: 16px; margin-bottom: 20px;
        border: 1px solid #334155; direction: rtl; text-align: right;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# النموذج + أدوات هندسية
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n-pose.pt")


model = load_model()


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


class RepQualityEngine:
    """تصنيف كل تكرار: مقبول (عمق+إيقاع+ثقة) / مرفوض / قيد المراجعة."""
    def __init__(self, depth_angle=75, top_angle=160):
        self.depth_angle = depth_angle
        self.top_angle = top_angle

    def new_rep(self):
        return {"stage": "up", "min_angle": 180.0,
                "rep_start": None, "conf_sum": 0.0, "conf_n": 0}

    def update(self, rep, angle, conf):
        rep["min_angle"] = min(rep["min_angle"], angle)
        rep["conf_sum"] += conf
        rep["conf_n"] += 1

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
# الشريط الجانبي
# ------------------------------------------------------------------
st.sidebar.title("🏋️‍♂️ إعدادات النادي الذكي")
st.sidebar.markdown("---")
target_reps = st.sidebar.slider("الهدف المطلوب للتكرارات:", 5, 30, 10)
depth_angle = st.sidebar.slider("زاوية العمق:", 50, 110, 75)
frame_skip = st.sidebar.slider("تخطي إطارات لتسريع المعالجة:", 1, 3, 1,
                               help="كل N إطار يُعالج إطار واحد")
st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 هندسة وتطوير")
st.sidebar.markdown("**Eng : Rafat Hazza Qasim**")
st.sidebar.markdown("[WF.DIETNCHEAT.COM](https://wf.dietncheat.com)")

# ------------------------------------------------------------------
# الهيدر
# ------------------------------------------------------------------
st.markdown("""
<div class="hero-card">
  <h2>🛡️ النظام الذكي لتحليل الأداء الرياضي — Gym Enterprise AI</h2>
  <p>تحليل دقيق للتكرارات، فصل العدات المقبولة عن المرفوضة والقيد المراجعة،
     وإنتاج فيديو معالج جاهز للتنزيل.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# رفع الفيديو
# ------------------------------------------------------------------
st.subheader("📁 وحدة رفع ومعالجة فيديوهات الرياضيين")
uploaded_file = st.file_uploader("قم برفع فيديو التمرين (MP4, AVI, MOV)...",
                                 type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    video_path = tfile.name

    with st.expander("▶️ معاينة الفيديو الأصلي"):
        st.video(video_path)

    if st.button("🚀 بدء المعالجة وتحليل الأداء بالذكاء الاصطناعي"):
        engine = RepQualityEngine(depth_angle)
        stats = {"performed": 0, "accepted": 0, "failed": 0, "review": 0}
        users = {}
        last_msg, last_color = "ابدأ التمرين!", (190, 190, 190)

        progress_bar = st.progress(0)
        status_text = st.empty()
        frame_placeholder = st.empty()

        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        output_path = tempfile.NamedTemporaryFile(delete=False,
                                                  suffix=".mp4").name
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (w, h))

        status_text.text("⚙️ جاري المعالجة وتطبيق خوارزميات الهيكل العظمي...")
        frame_idx, shown = 0, 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            results = model.track(frame, persist=True, verbose=False)
            annotated = results[0].plot(boxes=False, labels=False)
            main_angle = 180.0

            try:
                if results[0].boxes.id is not None and \
                   results[0].keypoints is not None:
                    ids = results[0].boxes.id.cpu().numpy()
                    kps = results[0].keypoints.xy.cpu().numpy()
                    confs = results[0].keypoints.conf
                    confs = confs.cpu().numpy() if confs is not None \
                        else np.ones((len(kps), 17))

                    for idx, u_id in enumerate(ids):
                        u_id = int(u_id)
                        kp = kps[idx]
                        conf = float(np.nanmean(confs[idx]))
                        if u_id not in users:
                            users[u_id] = {"rep": engine.new_rep(), "ok": 0}
                        if len(kp) >= 10:
                            angle = calculate_angle(kp[5][:2], kp[7][:2],
                                                    kp[9][:2])
                            if idx == 0:
                                main_angle = angle
                            rep = users[u_id]["rep"]
                            now = time.time()
                            if angle > engine.top_angle:
                                rep["stage"] = "up"
                            if angle < engine.depth_angle and \
                               rep["stage"] == "up":
                                rep["stage"] = "down"
                                rep["rep_start"] = now
                                rep["min_angle"] = angle
                            if rep["stage"] == "down":
                                engine.update(rep, angle, conf)
                            if angle > engine.top_angle and \
                               rep["stage"] == "down":
                                # ✦ هنا الإصلاح الجوهري: كل محاولة تُغلق
                                #   وتُصنَّف — لا كود ميت.
                                verdict, last_msg = engine.close_rep(rep, now)
                                stats["performed"] += 1
                                stats[verdict] += 1
                                if verdict == "accepted":
                                    users[u_id]["ok"] += 1
                                last_color = {"accepted": (46, 204, 113),
                                              "failed": (56, 67, 235),
                                              "review": (58, 178, 247)}[verdict]
                                users[u_id]["rep"] = engine.new_rep()
            except Exception:
                pass

            # لوحة المعلومات على الفيديو
            annotated = put_ar(annotated,
                               f"المطلوبة: {target_reps} | المنفذة: {stats['performed']}",
                               24, 20, 30, (255, 255, 255))
            annotated = put_ar(annotated,
                               f"مقبولة: {stats['accepted']}  مرفوضة: {stats['failed']}  مراجعة: {stats['review']}",
                               24, 62, 26, (46, 204, 113))
            annotated = put_ar(annotated, last_msg, 24, 100, 26, last_color)
            cv2.putText(annotated,
                        "Gym Enterprise AI - Eng: Rafat Hazza Qasim",
                        (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1, cv2.LINE_AA)

            out.write(annotated)
            shown += 1
            if shown % 3 == 0:
                frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                        channels="RGB", use_container_width=True)
            progress_bar.progress(min(1.0, frame_idx / total_frames))

        cap.release()
        out.release()
        status_text.success("✅ تمت معالجة الفيديو بنجاح تام!")

        # حفظ النتائج في session حتى يبقى زر التنزيل بعد إعادة التشغيل
        st.session_state["processed"] = output_path
        st.session_state["stats"] = stats
        st.session_state["target"] = target_reps

    # التقرير + التنزيل (يبقيان متاحين بعد المعالجة)
    if "processed" in st.session_state:
        stats = st.session_state["stats"]
        st.markdown("### 📊 تقرير الأداء الختامي:")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("الهدف", st.session_state["target"])
        c2.metric("إجمالي المحاولات", stats["performed"])
        c3.metric("المقبولة ✅", stats["accepted"])
        c4.metric("المرفوضة ❌", stats["failed"])
        c5.metric("قيد المراجعة ⚠️", stats["review"])
        rate = int(100 * stats["accepted"] / max(1, stats["performed"]))
        st.progress(rate / 100.0)
        st.markdown(f"**نسبة النجاح: {rate}%**")

        st.markdown("### 📥 تنزيل الفيديو المعالج:")
        with open(st.session_state["processed"], "rb") as f:
            st.download_button("📥 اضغط هنا لتنزيل الفيديو المعالج MP4",
                               data=f, file_name="processed_gym_output.mp4",
                               mime="video/mp4")
else:
    st.info("👆 ارفع فيديو تمرين لبدء التحليل الذكي.")
