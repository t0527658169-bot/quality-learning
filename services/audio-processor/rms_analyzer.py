# ===================================================
# rms_analyzer.py — אלגוריתם מדידת עוצמת קול (RMS)
# ===================================================
# RMS = Root Mean Square — שורש הממוצע הריבועי
# מודד את העוצמה האמיתית של אות שמע.
#
# נוסחה:  RMS = √( (1/N) × Σ(xᵢ²) )
#   xᵢ = ערך הדגימה ה-i
#   N  = מספר הדגימות בחלון
#
# סיווג ל-3 רמות:
#    שקט  — רעש רקע בלבד (RMS < 0.02)
#    רגיל — דיבור תקין   (RMS < 0.06)
#    רועש — רעש חזק       (RMS ≥ 0.06)

# יייבוא numpy לחישובים מתמטיים
import numpy as np
# ייבוא טיפוסים: Dict=מילון, List=רשימה, Tuple=זוג ערכים
from typing import Dict, List, Tuple
# יייבוא קצב דגימה וסף RMS מקובץ ההגדרות
from config import SAMPLE_RATE, RMS_QUIET_THRESHOLD, RMS_NORMAL_THRESHOLD


class RMSAnalyzer:


    # תוויות ואייקונים לכל רמה — מילון שממפה מפתח פנימי לשם עברי ואייקון
    LEVELS = {
        # שקט: עוצמה נמוכה מאוד — רעש רקע בלבד, ללא דיבור
        'quiet':  ("שקט",  "[+]"),
        # רגיל: עוצמה בינונית — דיבור תקין של מורה או תלמיד
        'normal': ("רגיל", "[~]"),
        # רועש: עוצמה גבוהה — רעש חזק, כנראה הפרעה או רעש מכני
        'noisy':  ("רועש", "[-]")
    }

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        # שמירת קצב הדגימה לשימוש בחישובי אורך חלון ו-RMS
        self.sample_rate = sample_rate

    # ------- חישובים בסיסיים -------
    def compute_rms(self, audio: np.ndarray) -> float:
        
        # בדיקת קלט ריק — אם אין דגימות, מחזירים 0 כדי למנוע שגיאת חלוקה
        if len(audio) == 0:
            # אין מה לחשב — החזרת אפס כברירת מחדל
            return 0.0
        # audio**2 = ריבוע כל דגימה, np.mean = ממוצע, np.sqrt = שורש → RMS
        return float(np.sqrt(np.mean(audio ** 2)))

    @staticmethod
    def rms_to_db(rms_value: float) -> float:
        
        # log10(0) לא מוגדר — RMS ≤ 0 מייצג שקט מוחלט, ערך -80dB כינוי לאין-סוף שלילי
        if rms_value <= 0:
            return -80.0  # ערך מינימלי (שקט מוחלט)
        # נוסחת המרה: 20×log10(RMS) — כפל ב-20 כי אנו מודדים אמפליטודה ולא הספק
        return float(20 * np.log10(rms_value))

    def classify(self, rms_value: float) -> Tuple[str, str]:
        """סיווג עוצמה ל-3 רמות. מחזיר (תווית, אייקון)."""
        # RMS מתחת לסף השקט (ברירת מחדל 0.02) — רעש רקע בלבד
        if rms_value < RMS_QUIET_THRESHOLD:
            # החזרת תווית "שקט" ואייקון [+]
            return self.LEVELS['quiet']
        # RMS בין הסף השקט לסף הרגיל (0.02–0.06) — דיבור תקין
        elif rms_value < RMS_NORMAL_THRESHOLD:
            # החזרת תווית "רגיל" ואייקון [~]
            return self.LEVELS['normal']
        # RMS מעל הסף הרגיל (≥ 0.06) — רעש חזק / הפרעה
        return self.LEVELS['noisy']

    # ------- ניתוח מקטע בודד -------
    def analyze_chunk(self, audio: np.ndarray, start_sec: float, end_sec: float) -> Dict:
        
        # שלב 1: חישוב ערך RMS הגולמי לכל הדגימות בחלון
        rms = self.compute_rms(audio)
        # שלב 2: המרת RMS לדציבלים לתצוגה ולהשוואה
        db = self.rms_to_db(rms)
        # שלב 3: סיווג העוצמה — label=שם העברי, icon=סמל ASCII
        label, icon = self.classify(rms)
        # החזרת מילון מלא עם כל הנתונים לשמירה ב-DB ולתצוגה
        return {
            # זמן התחלה וסיום של החלון בשניות
            'start_sec': start_sec, 'end_sec': end_sec,
            # ערכי עוצמה גולמיים ומסווגים
            'rms': rms, 'db': db, 'level': label, 'icon': icon
        }

    