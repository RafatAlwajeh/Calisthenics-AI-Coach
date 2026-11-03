# Gym Enterprise AI — Foundation v4.0

## التشغيل محلياً
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## النشر على Streamlit Community Cloud
1. ارفع `app.py` و`requirements.txt` إلى مستودع GitHub.
2. اربط المستودع بتطبيق Streamlit Cloud.
3. اختر `app.py` كملف رئيسي.
4. أول تشغيل قد يحتاج تنزيل أوزان `yolov8n-pose.pt` من Ultralytics؛ تأكد من السماح بالاتصال الخارجي أو أضف الأوزان للمستودع إذا كانت سياسة التخزين تسمح.

## حدود النسخة
- تحليل الذراع/زاوية الكوع كنموذج أولي، وليس مدرباً عاماً لكل تمارين الكاليسثينكس.
- لا يوجد تتبع أصابع/يدين؛ YOLOv8-Pose القياسي لا ينتج 21 نقطة لليد.
- وضع الكاميرا صورة ثابتة عبر `st.camera_input` وليس بثاً حياً.
- لا يمكن ضمان الدقة الطبية/الرياضية أو الاستقرار المطلق دون اختبار ميداني على فيديوهات متنوعة وأجهزة نشر فعلية.
