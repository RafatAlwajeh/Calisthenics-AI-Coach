class PushUpCounter:
    def __init__(self):
        self.counter = 0
        self.stage = "UP"  # الحالات: UP, DOWN
        self.accepted_reps = 0
        self.failed_reps = 0
        self.review_reps = 0
        self.feedback = "استعد للبدء..."
        self.depth_status = "مرتفع (غير مكتمل)"
        
    def update(self, elbow_angle):
        # مرحلة الصعود (الذراع ممدودة)
        if elbow_angle > 160:
            if self.stage == "DOWN":
                self.stage = "UP"
                self.accepted_reps += 1
                self.counter = self.accepted_reps
                self.feedback = "✅ تكرار مقبول، استمر!"
                self.depth_status = "العمق المطلوب ✓"
            else:
                self.stage = "UP"
                self.feedback = "حافظ على استقامة الجسم"

        # مرحلة النزول والعمق المطلوب
        elif elbow_angle < 90:
            if self.stage == "UP":
                self.stage = "DOWN"
                self.depth_status = "في النزول / العمق مثالي"
                self.feedback = "ممتاز، اصعد الآن!"
        
        elif 90 <= elbow_angle <= 130:
            self.depth_status = "في النزول"

        return self.counter, self.stage, self.feedback, self.depth_status
