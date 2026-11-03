import time

class AIBotRival:
    def __init__(self, target_reps=15, duration_seconds=60):
        self.target_reps = target_reps
        self.duration_seconds = duration_seconds
        self.start_time = time.time()
        
    def get_current_score(self):
        elapsed = time.time() - self.start_time
        if elapsed >= self.duration_seconds:
            return self.target_reps
        
        # زيادة التكرارات تدريجياً بمرور الوقت
        progress_ratio = elapsed / self.duration_seconds
        current_bot_reps = int(self.target_reps * progress_ratio)
        return min(current_bot_reps, self.target_reps)
