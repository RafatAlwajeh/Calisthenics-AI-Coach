import numpy as np

def calculate_angle(a, b, c):
    """
    حساب الزاوية بين ثلاث نقاط (الكتف، الكوع، الرسغ) لتحديد حركة الضغط
    """
    a = np.array(a) # الكتف
    b = np.array(b) # الكوع (رأس الزاوية)
    c = np.array(c) # الرسغ
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle
