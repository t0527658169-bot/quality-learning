# ===================================================
# audio_classifier.py — סיווג אקוסטי מבוסס MFCC
# ===================================================
# מודל סיווג המבחין בין 4 סוגי פעילות קולית:
#   1. שקט — אין פעילות קולית משמעותית
#   2. רעש סביבתי — מזגן, דלת, כיסאות (לא דיבור)
#   3. פטפטת חלשה — דיבור אנושי בעוצמה נמוכה
#   4. דיבור ברור — מורה או תלמיד מדברים בקול
#
# MFCC (Mel-Frequency Cepstral Coefficients) —
# מקדמים שמייצגים את "צורת" הצליל בצורה דומה לשמיעה אנושית.
# הם המאפיין המרכזי להבחנה בין סוגי צלילים.

import numpy as np
from typing import Tuple
from config import SAMPLE_RATE
from audio_utils import compute_mfcc


class AudioClassifier:
    """מסווג אקוסטי — מזהה את סוג הפעילות הקולית בכל חלון."""

    # קטגוריות סיווג
    SILENCE = "שקט"
    BACKGROUND_NOISE = "רעש_סביבתי"
    LOW_SPEECH = "פטפטת_חלשה"
    CLEAR_SPEECH = "דיבור_ברור"

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

    # ------- חילוץ מקדמי MFCC -------
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        חילוץ 13 מקדמי MFCC מקטע אודיו.
        MFCC מחקה את תפיסת השמיעה האנושית — מבדיל בין סוגי צלילים.
        מחזיר ממוצע על פני הזמן = וקטור מאפיינים אחד לכל קטע.
        """
        return compute_mfcc(audio, self.sample_rate, n_mfcc)

    # ------- סיווג קטע אודיו -------
    def classify(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        סיווג קטע אודיו ל-4 קטגוריות על בסיס כללים אקוסטיים:
        - אנרגיה (RMS): כמה חזק הצליל
        - ZCR: קצב חציית אפס (גבוה ברעש אקראי)
        - שונות MFCC: גבוהה בדיבור, נמוכה ברעש אחיד

        מחזיר: (קטגוריה, ציון_ביטחון)
        """
        # חישוב 3 מאפיינים פשוטים
        rms = float(np.sqrt(np.mean(audio ** 2)))
        # חישוב ZCR ידני — ספירת חציות אפס
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0))
        mfcc = self.extract_mfcc(audio)
        # שונות MFCC (ללא מקדם ראשון שמייצג רק אנרגיה)
        mfcc_var = float(np.var(mfcc[1:]))

        # כללי סיווג לפי ספים
        # שקט: אנרגיה נמוכה מאוד
        if rms < 0.01:
            return self.SILENCE, 0.95

        # רעש סביבתי: ZCR גבוה + MFCC אחיד (לא דיבורי)
        if zcr > 0.15 and mfcc_var < 5.0:
            return self.BACKGROUND_NOISE, 0.80

        # פטפטת חלשה: אנרגיה נמוכה אבל יש מאפייני דיבור
        if rms < 0.04 and mfcc_var > 5.0:
            return self.LOW_SPEECH, 0.75

        # דיבור ברור: אנרגיה גבוהה + מאפייני דיבור
        return self.CLEAR_SPEECH, 0.85
