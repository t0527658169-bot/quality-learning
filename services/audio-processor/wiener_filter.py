# ===================================================
# wiener_filter.py — אלגוריתם פילטר וינר לניקוי רעשים
# ===================================================
# נוסחה: H(f) = max(0, 1 - Noise(f) / Signal(f))
# כאשר H(f) = מקדם הסינון בכל תדר

import numpy as np
# פונקציות מעבר בין תחום הזמן לתחום התדר ובחזרה
# STFT = Short-Time Fourier Transform (פורייה קצר-טווח)
from scipy.signal import stft, istft
from config import SAMPLE_RATE, WIENER_NOISE_DURATION_SEC, WIENER_NPERSEG


class WienerFilter:
    # אתחול המחלקה — שמירת קצב הדגימה לשימוש בכל הפונקציות
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        # שמירת קצב הדגימה (כמה דגימות לשנייה — ברירת מחדל 16,000)
        self.sample_rate = sample_rate

    # ------- הערכת פרופיל הרעש -------
    # מקבלת את מערך השמע ואת משך החלון בשניות לבדיקת רעש
    def _estimate_noise(self, audio: np.ndarray, noise_duration: float = WIENER_NOISE_DURATION_SEC) -> np.ndarray:
        # מכפילה את השניות בקצב הדגימה והופכת את זה למספר שלם
        #  זה נותן לנו את מספר הדגימות הפיזי
        noise_samples = int(noise_duration * self.sample_rate)

       
        # שמירת העוצמה והדגימה
        rms_list = []
        chunks = []
        # סורקת את כל הקובץ מתחילתו ועד סופו
        for i in range(0, max(1, len(audio) - noise_samples), noise_samples // 2):
            # חיתוך קטע 
            chunk = audio[i:i + noise_samples]
            # float64 מונע overflow בריבוע — חשוב בדגימות float32 גבוהות
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            rms_list.append(rms)
            chunks.append(chunk)

        # שלב 2: מציאת סף RMS שמייצג את 30% החלונות השקטים ביותר
        threshold = np.percentile(rms_list, 30)
        # סינון: שמירה רק על חלונות שה-RMS שלהם ≤ הסף
        quiet_chunks = [c for c, r in zip(chunks, rms_list) if r <= threshold]

        # שלב 3: חישוב STFT לכל חלון שקט ואיסוף פרופיל הספקטרום שלו
        all_profiles = []
        for chunk in quiet_chunks:
            # STFT — ממיר חלון מתחום הזמן לתחום התדר
            _, _, noise_stft = stft(chunk, fs=self.sample_rate, nperseg=WIENER_NPERSEG)
            # הספק הספקטרלי: |STFT|² ממוצע על ציר הזמן → וקטור תדרים
            all_profiles.append(np.mean(np.abs(noise_stft) ** 2, axis=1))

        # התוצאה = וקטור של הספק רעש ממוצע לכל תדר — משמש לחישוב מקדם וינר
        noise_profile = np.mean(all_profiles, axis=0)
        return noise_profile

    # ------- הפעלת הפילטר -------
    def apply(self, audio: np.ndarray) -> np.ndarray:
    
        # שלב 1: קריאה לפונקציה הפנימית לחישוב פרופיל הרעש
        noise_profile = self._estimate_noise(audio)

        
        # freqs = וקטור התדרים, times = ציר זמן, audio_stft = הספקטרוגרמה המרוכבת
        freqs, times, audio_stft = stft(audio, fs=self.sample_rate, nperseg=WIENER_NPERSEG)
        # חישוב האנרגיה  על כל חלק בקובץ
        audio_power = np.abs(audio_stft) ** 2

       
       
        noise_expanded = noise_profile[:, np.newaxis]
        # חישוב לכל פריים -מדד ניקוי 0=רעש 1 =דיבור
        wiener_gain = np.maximum(0.0, 1.0 - noise_expanded / (audio_power + 1e-10))

        # הכפלת הספקטרוגרמה במקדמי הסינון — מדכאים תדרים עם רעש
        cleaned_stft = audio_stft * wiener_gain
        # ISTFT — מחזיר את האות הנקי לתחום הזמן (מאות תדרים בחזרה לאות)
        _, cleaned_audio = istft(cleaned_stft, fs=self.sample_rate, nperseg=WIENER_NPERSEG)

        # חיתוך לאורך המקורי — ISTFT לפעמים מוסיף כמה דגימות עודפות בסוף
        return cleaned_audio[:len(audio)]
