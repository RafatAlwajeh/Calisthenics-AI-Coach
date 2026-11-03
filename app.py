import streamlit as st
import cv2
import numpy as np
import time
import tempfile
from ultralytics import YOLO
from core.angle_calculator import calculate_angle
from core.rep_counter import PushUpCounter
from core.battle_engine import AIBotRival

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="Gym Enterprise AI - Calisthenics Battle Coach",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تخصيص واجهة CSS الداعمة للغة العربية (RTL) والألوان الاحترافية
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap');
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        text-align: right;
        background-color: #0d1117;
        color: #ffffff;
    }
    .stApp { background-color: #0d1117; }
    </style>
""", unsafe_allow_html=True)

# التهيئة في الجلسة
if 'counter_obj' not in st.session_state:
    st.session_state.counter_obj = PushUpCounter()
if 'bot_rival' not in st.session_state:
    st.session_state.bot_rival = AIBotRival(target_reps=15, duration_seconds=60)
if 'start_match_time' not in st.session_state:
    st.session_state.start_match_time = time.time()

# ---------------------------------------------------------
# الشريط الجانبي (Sidebar)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🦍 Gym Enterprise AI")
    st.caption("منصة المدرب الذكي للكاليستشينكس")
    st.markdown("---")
    
    selected_page = st.radio(
        "القائمة الرئيسية",
        ["لوحة التحكم", "بدء التحدي (Battle Mode)", "تحليل تمرين", "سجل التمارين", "الإحصائيات", "الإعدادات"]
    )
    
    st.markdown("---")
    # اختيار مصدر الفيديو (كاميرا حية أو رفع فيديو تجريبي)
    video_source = st.selectbox(
        "اختر مصدر الإدخال للتحليل",
        ["كاميرا الويب المباشرة (Live Webcam)", "رفع فيديو تجريبي (Upload Video)"]
    )

# ---------------------------------------------------------
# الواجهة الرئيسية (Dashboard)
# ---------------------------------------------------------
st.markdown("### منصة المدرب الذكي للكاليستشينكس - تحليل الحركة بالذكاء الاصطناعي • منافسة مباشرة")

# الوقت المتبقي
elapsed_time = int(time.time() - st.session_state.start_match_time)
time_left = max(0, 60 - elapsed_time)
formatted_time = f"00:{time_left:02d}"

user_reps = st.session_state.counter_obj.counter
bot_reps = st.session_state.bot_rival.get_current_score()

# شريط المنافسة العلوي (Battle HUD)
col_hud1, col_hud2, col_hud3 = st.columns([2, 1, 2])
with col_hud1:
    st.markdown(f"""
        <div style="background:#1f6feb; padding:10px; border-radius:8px; text-align:center;">
            <b>👑 أنت (المتقدم حالياً)</b><br>
            <span style="font-size:28px; font-weight:bold;">{user_reps}</span> تكرار
        </div>
    """, unsafe_allow_html=True)
with col_hud2:
    st.markdown(f"""
        <div style="background:#21262d; padding:10px; border-radius:8px; text-align:center; border:1px solid #f85149;">
            <span style="font-size:12px; color:#8b949e;">الوقت المتبقي</span><br>
            <span style="font-size:20px; font-weight:bold; color:#f85149;">⏱ {formatted_time}</span>
        </div>
    """, unsafe_allow_html=True)
with col_hud3:
    st.markdown(f"""
        <div style="background:#da3633; padding:10px; border-radius:8px; text-align:center;">
            <b>🤖 البوت الذكي (AI Bot)</b><br>
            <span style="font-size:28px; font-weight:bold;">{bot_reps}</span> تكرار
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# تقسيم الشاشة الرئيسي: نافذة الفيديو والتحليل (يسار) وإحصائيات الأداء (يمين)
col_video, col_stats = st.columns([2, 1])

with col_video:
    st.markdown("#### 📷 منطقة تحليل التمرين الحي (Live / Video Analysis)")
    video_placeholder = st.empty()
    
    # محرك المعالجة المشترك لعرض الإطارات
    def process_frame(frame):
        h, w, _ = frame.shape
        # محاكاة حساب زاوية الكوع (أو استخراجها عبر نموذج YOLOv8 Pose الحقيقي)
        # يمكنك استبدالها لاحقاً باستخراج إحداثيات مفاصل YOLO الفعلية
        elbow_angle = float(np.random.randint(70, 165)) 
        
        current_count, stage, feedback, depth_status = st.session_state.counter_obj.update(elbow_angle)
        
        # رسم مؤشرات على الإطار
        cv2.putText(frame, f"Angle: {int(elbow_angle)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, feedback, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if video_source == "رفع فيديو تجريبي (Upload Video)":
        uploaded_file = st.file_uploader("قم برفع فيديو التمرين بصيغة MP4 أو AVI", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            
            if st.button("تشغيل تحليل الفيديو المرفوع"):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    processed_frame = process_frame(frame)
                    video_placeholder.image(processed_frame, channels="RGB", use_container_width=True)
                    time.sleep(0.03)
                cap.release()
    else:
        run_webcam = st.toggle("تفعيل كاميرا الويب المباشرة", value=False)
        if run_webcam:
            cap = cv2.VideoCapture(0)
            while run_webcam and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("تعذر الوصول للكاميرا.")
                    break
                frame = cv2.flip(frame, 1)
                processed_frame = process_frame(frame)
                video_placeholder.image(processed_frame, channels="RGB", use_container_width=True)
                if time_left <= 0:
                    break
            cap.release()
        else:
            st.info("قم بتفعيل كاميرا الويب أو اختر رفع فيديو من القائمة الجانبية للبدء.")

with col_stats:
    st.markdown("#### 📊 إحصائيات الأداء الفوري")
    progress_pct = int((user_reps / 15) * 100)
    st.metric(label="التقدم نحو الهدف (الهدف: 15 تكرار)", value=f"{progress_pct}%", delta=f"{user_reps} / 15")
    st.progress(min(progress_pct, 100) / 100)
    
    st.markdown("---")
    
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown(f"""
            <div style="background:#21262d; padding:10px; border-radius:8px; text-align:center; border-top: 3px solid #238636;">
                <span style="font-size:12px; color:#8b949e;">المقبولة</span><br>
                <span style="font-size:20px; font-weight:bold; color:#238636;">{st.session_state.counter_obj.accepted_reps}</span>
            </div>
        """, unsafe_allow_html=True)
    with col_c2:
        st.markdown(f"""
            <div style="background:#21262d; padding:10px; border-radius:8px; text-align:center; border-top: 3px solid #da3633;">
                <span style="font-size:12px; color:#8b949e;">الفاشلة</span><br>
                <span style="font-size:20px; font-weight:bold; color:#da3633;">{st.session_state.counter_obj.failed_reps}</span>
            </div>
        """, unsafe_allow_html=True)
    with col_c3:
        st.markdown(f"""
            <div style="background:#21262d; padding:10px; border-radius:8px; text-align:center; border-top: 3px solid #d29922;">
                <span style="font-size:12px; color:#8b949e;">قيد المراجعة</span><br>
                <span style="font-size:20px; font-weight:bold; color:#d29922;">{st.session_state.counter_obj.review_reps}</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="background:#161b22; padding:15px; border-radius:8px; border:1px solid #30363d;">
            <b>📌 آخر تقييم للحركة</b><br>
            <span style="color:#238636; font-weight:bold;">تكرار ممتاز، استمر!</span><br>
            <span style="font-size:13px; color:#8b949e;">
            • العمق: مكتمل ✓<br>
            • الإيقاع: مناسب (2.1 ث) ✓<br>
            • وضعية الجسم: مستقرة ✓
            </span>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# القسم السفلي: الرسوم البيانية وملخص التحدي
# ---------------------------------------------------------
st.markdown("---")
col_g1, col_g2, col_g3 = st.columns(3)

with col_g1:
    st.markdown("##### 📈 تطور الأداء")
    st.line_chart(np.random.randn(15, 1) * 5 + 80)

with col_g2:
    st.markdown("##### 🏆 ملخص التحدي")
    st.markdown("""
        * **إجمالي المحاولات:** 15
        * **دقة الأداء:** 80%
        * **متوسط زمن التكرار:** 2.3 ثانية
    """)

with col_g3:
    st.markdown("##### 🥇 المركز في التحدي")
    st.markdown("""
        <div style="background: linear-gradient(135deg, #9e6a03 0%, #d29922 100%); padding:15px; border-radius:8px; text-align:center; color:white;">
            <b>🏆 أنت المتقدم حالياً!</b><br>
            تهانينا، الأداء ممتاز وتتفوق على البوت.
        </div>
    """, unsafe_allow_html=True)
