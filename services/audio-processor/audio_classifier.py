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
from config import SAMPLE_RATE, CLASSIFIER_SILENCE_RMS, CLASSIFIER_NOISE_ZCR, CLASSIFIER_NOISE_MFCC_VAR, CLASSIFIER_LOW_SPEECH_RMS
from audio_utils import compute_mfcc


# הגדרת מחלקת הסיווג האקוסטי — מחלקה שמזהה מה סוג הצליל בכל חלון
class AudioClassifier:

    # --- קבועי קטגוריות (תוויות לסיווגים האפשריים) ---
    # שקט = כמעט אין צליל, רק רעש מיקרופון זעיר
    SILENCE = "שקט"
    # רעש סביבתי = מזגן, כיסאות, צלצול — לא דיבור
    BACKGROUND_NOISE = "רעש_סביבתי"
    # פטפטת חלשה = דיבור בעוצמה נמוכה, לדוגמה ילדים מלחשים
    LOW_SPEECH = "פטפטת_חלשה"
    # דיבור ברור = מורה או תלמיד מדברים בקול נורמלי
    CLEAR_SPEECH = "דיבור_ברור"

    # אתחול המחלקה — שמירת קצב הדגימה
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

    # ------- חילוץ מקדמי MFCC -------
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        return compute_mfcc(audio, self.sample_rate, n_mfcc)

    # ------- סיווג קטע אודיו -------
    def classify(self, audio: np.ndarray) -> Tuple[str, float]:
        # --- חישוב מאפיין 1: RMS (עוצמת הצליל) ---
        # מעלה כל דגימה בריבוע, מחשב ממוצע, ולוקח שורש — נותן עוצמה אמיתית
        rms = float(np.sqrt(np.mean(audio ** 2)))

        # --- חישוב מאפיין 2: ZCR (קצב חציית אפס) ---
        # np.sign() ממיר כל דגימה ל-+1 או -1
        # np.diff() מחשב הפרש בין דגימות סמוכות — שינוי סימן = חצייה
        # np.abs() לוקח ערך מוחלט, > 0 מזהה שינויי סימן
        # np.mean() נותן אחוז החציות מכלל הדגימות
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0))

        # --- חישוב מאפיין 3: MFCC ---
        # 13 מקדמים המייצגים את "צורת" הצליל בסולם שמיעה אנושי
        mfcc = self.extract_mfcc(audio)

        # --- חישוב שונות MFCC (ללא מקדם ראשון) ---
        # mfcc[0] = אנרגיה כללית (לא מעניין אותנו כאן)
        # mfcc[1:] = המקדמים הספקטרליים — שונות גבוהה = דיבור משתנה
        # np.var() מחשב שונות — כמה המקדמים משתנים
        mfcc_var = float(np.var(mfcc[1:]))

        # --- כללי סיווג (בדיקה לפי סדר עדיפות) ---

        # בדיקה 1: שקט מוחלט
        # RMS < 0.01 = עוצמה נמוכה מ-1% מהמקסימום — כמעט אין צליל
        # ביטחון 0.95 = מאוד בטוחים בסיווג זה
        # if rms < 0.01:
        if rms < CLASSIFIER_SILENCE_RMS:
            return self.SILENCE, 0.95

        # בדיקה 2: רעש סביבתי (מזגן, מאוורר, רחש רקע)
        # ZCR > 0.15 = 15% מהדגימות חוצות אפס — אופייני לרעש אקראי
        # mfcc_var < 5.0 = MFCC אחיד — אין שינויים כמו בדיבור
        # if zcr > 0.15 and mfcc_var < 5.0:
        if zcr > CLASSIFIER_NOISE_ZCR and mfcc_var < CLASSIFIER_NOISE_MFCC_VAR:
            return self.BACKGROUND_NOISE, 0.80

        # בדיקה 3: פטפטת חלשה (ילדים מלחשים בזמן הפרעה שקטה)
        # RMS < 0.04 = עוצמה נמוכה (4% מהמקסימום) אבל לא שקט מוחלט
        # mfcc_var > 5.0 = MFCC משתנה = יש מאפייני דיבור
        # if rms < 0.04 and mfcc_var > 5.0:
        if rms < CLASSIFIER_LOW_SPEECH_RMS and mfcc_var > CLASSIFIER_NOISE_MFCC_VAR:
            return self.LOW_SPEECH, 0.75

        # ברירת מחדל: דיבור ברור — הגיע לכאן = אנרגיה מספיקה + מאפייני דיבור
        # ביטחון 0.85 = בטוח אבל לא מוחלט (יכול להיות קצת רעש)
        return self.CLEAR_SPEECH, 0.85
