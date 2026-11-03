# -*- coding: utf-8 -*-
"""
====================================================================
Ultimate Calisthenics Enterprise AI Coach — Streamlit Edition
Prepared by Eng : Rafat Hazza Qasim (Rafat Al-Wajeh)
====================================================================
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

# إعداد الصفحة وتصميم الواجهة المؤسسية
st.set_page_config(
    page_title="Ultimate Calisthenics AI Coach | منصة المدرب الذكي",
    page_icon="💪",
    layout="wide"
)

# حقن أكواد CSS لدعم اللغة العربية وتوجيه العناصر من اليمين إلى اليسار (RTL) والتوافق مع الجوال
st.markdown("""
    <style>
    /* توجيه كافة النصوص والعناصر من اليمين إلى اليسار */
    html, body, [class*="css"] {
        direction: rtl;
        text-align: right;
        font-family: 'Cairo', Tahoma, sans-serif;
    }
    
    /* تنسيق اللوحة الجانبية */
    section[data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
        background-color: #f8f9fa;
    }

    /* تحسين مظهر الأزرار لتكون متجاوبة ومريحة للمستخدم على الجوال */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        font-weight: bold;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    
    /* تنسيق بطاقات الإحصائيات */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# تحميل نموذج الذكاء الاصطناعي (يتم تخزينه مؤقتاً لسرعة التشغيل)
@st.cache_resource
def load_model():
    return YOLO("yolov8n-pose.pt")

model = load_model()

# دالة حساب الزاوية الهندسية
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# لوحة التحكم المؤسسية في الشريط الجانبي (Sidebar)
st.sidebar.title("🏋️‍♂️ لوحة تحكم النظام المؤسسي")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox("اختر وضع التشغيل المؤسسي:", ["تحليل فيديو مسجل (AI Video Processing)", "البث المباشر الحي (Live Camera)"])

target_reps = st.sidebar.slider("الهدف المستهدف للعدادات (Reps):", 5, 30, 10)
depth_angle = st.sidebar.slider("حساسية زاوية العمق (Depth Angle):", 50, 100, 75)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 إعداد وتطوير الشركة")
st.sidebar.markdown("**Eng : Rafat Hazza Qasim**")
st.sidebar.markdown("[WF.DIETNCHEAT.COM](https://wf.dietncheat.com)")

# العنوان الرئيسي للمنصة
st.title("💪 نظام تحليل الكالسثانكس والمدرب الذكي (Enterprise AI Coach)")
st.markdown("منصة ذكية مؤسسية متكاملة لتحليل الأداء الرياضي، تتبع الهيكل العظمي في الوقت الفعلي، وتصحيح الزوايا الحركية بدقة فائقة.")
st.markdown("---")

# ==========================================
# الخيار الأول: تحليل فيديو مسجل مع العرض اللحظي
# ==========================================
if app_mode == "تحليل فيديو مسجل (AI Video Processing)":
    st.header("📁 وحدة معالجة وتحليل الفيديوهات المسجلة")
    uploaded_file = st.file_uploader("قم بإرفاق ملف الفيديو الخاص بالتمرين (MP4, AVI, MOV)...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # حفظ الفيديو المرفوع مؤقتاً
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.info("تم إرفاق الفيديو بنجاح. انقر أدناه لبدء تطبيق خوارزميات الذكاء الاصطناعي ومعالجة الإطارات لحظة بلحظة.")
        
        if st.button("🚀 بدء تشغيل خوارزميات التحليل والمعالجة"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # تخصيص مكان لعرض الفيديو المعالج إطاراً بإطار في الوقت الفعلي
            frame_placeholder = st.empty()
            
            cap = cv2.VideoCapture(video_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            stage = "up"
            counter = 0
            frame_idx = 0

            status_text.text("⚙️ جاري معالجة الإطارات وتطبيق خوارزميات الهيكل العظمي...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # تطبيق نموذج YOLOv8 Pose على الإطار الحالي
                results = model.track(frame, persist=True, verbose=False)
                annotated = results[0].plot(boxes=False, labels=False)

                try:
                    if results[0].keypoints is not None:
                        keypoints_all = results[0].keypoints.xy.cpu().numpy()
                        if len(keypoints_all) > 0:
                            keypoints = keypoints_all[0]
                            if len(keypoints) >= 10:
                                shoulder = keypoints[5][:2]
                                elbow = keypoints[7][:2]
                                wrist = keypoints[9][:2]
                                angle = calculate_angle(shoulder, elbow, wrist)

                                if angle > 160:
                                    stage = "up"
                                if angle < depth_angle and stage == "up":
                                    stage = "down"
                                    counter += 1
                except Exception:
                    pass

                # رسم إحصائيات النظام المؤسسي على الإطار
                cv2.putText(annotated, f"Reps: {counter} / {target_reps}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 128), 3)
                cv2.putText(annotated, "Enterprise AI Coach - Eng: Rafat Hazza", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # تحويل الألوان وعرض الإطار المعالج مباشرة للمستخدم (Real-time Streaming)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_rgb, channels="RGB", use_column_width=True)

                frame_idx += 1
                if total_frames > 0:
                    progress_bar.progress(min(1.0, frame_idx / total_frames))

            cap.release()
            status_text.success("✅ تمت معالجة الفيديو بالكامل بنجاح بواسطة خوارزميات الشركة!")

            # لوحة مؤشرات الأداء المؤسسية النهائية
            st.markdown("### 📊 تقرير الأداء النهائي:")
            col1, col2, col3 = st.columns(3)
            col1.metric("إجمالي العدات المقبولة", counter)
            col2.metric("الهدف المؤسسي المحدد", target_reps)
            col3.metric("تقييم مستوى الأداء", "مستوى قياسي 🔥" if counter >= target_reps else "جيد، يتطلب تحسين 💪")

# ==========================================
# الخيار الثاني: البث الحي (Live Camera)
# ==========================================
elif app_mode == "البث المباشر الحي (Live Camera)":
    st.header("🎥 وحدة البث المباشر والتحليل اللحظي للمنشآت")
    st.info("ملاحظة: يتطلب تشغيل الكاميرا الحية السماح للمتصفح بالوصول إلى الكاميرا من جهازك (جوال أو لابتوب).")
    
    run_cam = st.checkbox("تفعيل وتشغيل البث المباشر")
    
    if run_cam:
        cam_placeholder = st.empty()
        cap = cv2.VideoCapture(0)
        
        counter = 0
        stage = "up"
        
        while run_cam and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("تعذر الوصول إلى كاميرا الجهاز.")
                break

            frame = cv2.flip(frame, 1)
            results = model.track(frame, persist=True, verbose=False)
            annotated = results[0].plot(boxes=False, labels=False)

            try:
                if results[0].keypoints is not None:
                    keypoints_all = results[0].keypoints.xy.cpu().numpy()
                    if len(keypoints_all) > 0:
                        keypoints = keypoints_all[0]
                        if len(keypoints) >= 10:
                            shoulder = keypoints[5][:2]
                            elbow = keypoints[7][:2]
                            wrist = keypoints[9][:2]
                            angle = calculate_angle(shoulder, elbow, wrist)

                            if angle > 160:
                                stage = "up"
                            if angle < depth_angle and stage == "up":
                                stage = "down"
                                counter += 1
            except Exception:
                pass

            h, w = frame.shape[:2]
            cv2.putText(annotated, f"Live Reps: {counter}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 128), 3)
            cv2.putText(annotated, "Enterprise AI Coach - Eng: Rafat Hazza", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(annotated_rgb, channels="RGB", use_column_width=True)

        cap.release()
