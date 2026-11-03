# -*- coding: utf-8 -*-
"""
====================================================================
Ultimate Calisthenics Web App — Streamlit Edition
Prepared by Eng : Rafat Hazza Qasim (Rafat Al-Wajeh)
====================================================================
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

# إعداد الصفحة وتصميم الواجهة
st.set_page_config(
    page_title="Ultimate Calisthenics AI Coach",
    page_icon="💪",
    layout="wide"
)

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

# واجهة المستخدم في الموقع (Sidebar)
st.sidebar.title("🏋️‍♂️ لوحة تحكم المدرب الذكي")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox("اختر وضع التشغيل:", ["اختبار نموذج عبر رفع فيديو (Upload Video)", "البث المباشر الحي (Live Camera)"])

target_reps = st.sidebar.slider("الهدف المطلوب (Reps):", 5, 30, 10)
depth_angle = st.sidebar.slider("حساسية العمق (Depth Angle):", 50, 100, 75)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 إعداد وتطوير")
st.sidebar.markdown("**Eng : Rafat Hazza Qasim**")
st.sidebar.markdown("[WF.DIETNCHEAT.COM](https://wf.dietncheat.com)")

# العنوان الرئيسي للموقع
st.title("💪 نظام تحليل الكالسثانكس والمدرب الذكي (AI Calisthenics Coach)")
st.markdown("منصة ذكية متكاملة لتحليل الأداء الرياضي، تصحيح الزوايا، ومنافسة الأنظمة الذكية في الوقت الفعلي.")

# ==========================================
# الخيار الأول: اختبار الموديل عبر رفع فيديو
# ==========================================
if app_mode == "اختبار نموذج عبر رفع فيديو (Upload Video)":
    st.header("📁 اختبار الموديل على فيديو مسجل")
    uploaded_file = st.file_uploader("اختر فيديو التمرين (MP4, AVI, MOV)...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # حفظ الفيديو المرفوع مؤقتاً
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.video(video_path)
        
        if st.button("🚀 بدء تحليل ومعالجة الفيديو"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            cap = cv2.VideoCapture(video_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_path = "processed_output.mp4"
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))

            stage = "up"
            counter = 0
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

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

                # رسم إحصائيات بسيطة على الفيديو
                cv2.putText(annotated, f"Reps: {counter} / {target_reps}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 128), 3)
                cv2.putText(annotated, "Prepared by Eng : Rafat Hazza Qasim", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                out.write(annotated)
                frame_idx += 1
                if total_frames > 0:
                    progress_bar.progress(min(1.0, frame_idx / total_frames))

            cap.release()
            out.release()
            status_text.success("✅ تمت المعالجة بنجاح!")

            st.markdown("### 📊 نتيجة التحليل النهائية:")
            col1, col2, col3 = st.columns(3)
            col1.metric("إجمالي العدات المقبولة", counter)
            col2.metric("الهدف المحدد", target_reps)
            col3.metric("حالة الأداء", "ممتاز 🔥" if counter >= target_reps else "جيد، استمر 💪")

            st.video(output_path)

# ==========================================
# الخيار الثاني: البث الحي (Live Camera)
# ==========================================
elif app_mode == "البث المباشر الحي (Live Camera)":
    st.header("🎥 البث المباشر والتحليل اللحظي")
    st.info("ملاحظة: يتطلب تشغيل الكاميرا الحية السماح للمتصفح بالوصول إليها.")
    
    run_cam = st.checkbox("تشغيل الكاميرا المباشرة")
    
    if run_cam:
        cam_placeholder = st.empty()
        cap = cv2.VideoCapture(0)
        
        counter = 0
        stage = "up"
        
        while run_cam and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("تعذر الوصول إلى الكاميرا.")
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
            cv2.putText(annotated, "Prepared by Eng : Rafat Hazza Qasim", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cam_placeholder.image(annotated, channels="BGR", use_column_width=True)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
