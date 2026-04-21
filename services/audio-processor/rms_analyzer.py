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
#   🟢 שקט  — רעש רקע בלבד (RMS < 0.02)
#   🟡 רגיל — דיבור תקין   (RMS < 0.06)
#   🔴 רועש — רעש חזק       (RMS ≥ 0.06)

import numpy as np
from typing import Dict, List, Tuple
from config import SAMPLE_RATE, RMS_QUIET_THRESHOLD, RMS_NORMAL_THRESHOLD


class RMSAnalyzer:
    """מנתח עוצמת קול מבוסס RMS."""

    # תוויות ואייקונים לכל רמה
    LEVELS = {
        'quiet':  ("שקט",  "[+]"),
        'normal': ("רגיל", "[~]"),
        'noisy':  ("רועש", "[-]")
    }

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

    # ------- חישובים בסיסיים -------
    def compute_rms(self, audio: np.ndarray) -> float:
        """
        חישוב ערך RMS יחיד לקטע אודיו שלם.
        מעלה כל דגימה בריבוע, ממוצע, ושורש — נותן את העוצמה האמיתית.
        """
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))

    @staticmethod
    def rms_to_db(rms_value: float) -> float:
        """
        המרה מ-RMS לדציבלים.
        נוסחה: dB = 20 × log₁₀(RMS)
        ערך שלילי = חלש, קרוב ל-0 = חזק.
        """
        if rms_value <= 0:
            return -80.0  # ערך מינימלי (שקט מוחלט)
        return float(20 * np.log10(rms_value))

    def classify(self, rms_value: float) -> Tuple[str, str]:
        """סיווג עוצמה ל-3 רמות. מחזיר (תווית, אייקון)."""
        if rms_value < RMS_QUIET_THRESHOLD:
            return self.LEVELS['quiet']
        elif rms_value < RMS_NORMAL_THRESHOLD:
            return self.LEVELS['normal']
        return self.LEVELS['noisy']

    # ------- ניתוח מקטע בודד -------
    def analyze_chunk(self, audio: np.ndarray, start_sec: float, end_sec: float) -> Dict:
        """
        ניתוח מלא של מקטע אודיו: RMS, dB, וסיווג.
        """
        rms = self.compute_rms(audio)
        db = self.rms_to_db(rms)
        label, icon = self.classify(rms)
        return {
            'start_sec': start_sec, 'end_sec': end_sec,
            'rms': rms, 'db': db, 'level': label, 'icon': icon
        }

    # ------- סיכום סטטיסטי לשיעור שלם -------
    def get_summary(self, results: List[Dict]) -> Dict:
        """
        חישוב סיכום סטטיסטי: ממוצע, מקסימום, ואחוז כל רמה.
        """
        if not results:
            return {}
        rms_vals = [r['rms'] for r in results]
        levels = [r['level'] for r in results]
        total = len(levels)

        quiet_n = levels.count("שקט")
        normal_n = levels.count("רגיל")
        noisy_n = levels.count("רועש")

        return {
            'avg_rms': float(np.mean(rms_vals)),
            'max_rms': float(np.max(rms_vals)),
            'avg_db': float(np.mean([r['db'] for r in results])),
            'quiet_pct': quiet_n / total,
            'normal_pct': normal_n / total,
            'noisy_pct': noisy_n / total,
            'dominant_level': max(
                [(quiet_n, "שקט"), (normal_n, "רגיל"), (noisy_n, "רועש")],
                key=lambda x: x[0]
            )[1]
        }
