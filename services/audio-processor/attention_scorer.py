# ===================================================
# attention_scorer.py — מדד איכות השיעור
# ===================================================
# מסווג כל חלון זמן כחיובי (דובר יחיד / ריבוי לצורך השיעור)
# או שלילי (ריבוי הפרעה). רעש ללא דוברים לא נספר.
#
# הציון הסופי: אחוז החלונות החיוביים מתוך הרלוונטיים.

# ייבוא numpy — ספריית חישובים מתמטיים (כאן משמשת לחישוב אחוזים)
import numpy as np
# ייבוא טיפוסים לתיעוד — Dict=מילון, List=רשימה, Optional=ערך שיכול להיות None
from typing import Dict, List, Optional
from config import GRADE_EXCELLENT, GRADE_GOOD, GRADE_MEDIUM

# הגדרת המחלקה הראשית — מחשבת ציון איכות לשיעור שלם
class AttentionScorer:
    """מסווג כל חלון כחיובי/שלילי ומחשב אחוזי איכות לשיעור."""

    # --- קבועי תוויות (ערכים קבועים שמשמשים לאורך כל הקוד) ---
    # חלון חיובי = מורה מדברת לבד, או ריבוי דוברים לצורך השיעור (דיון/משימה)
    POSITIVE = "חיובי"
    # חלון שלילי = ריבוי דוברים שלא לצורך השיעור (ילדים מפריעים)
    NEGATIVE = "הפרעה"
    # רעש = אין דיבור אנושי כלל — לא נספר בחישוב הציון
    NOISE = None

    # ------- סיווג חלון בודד -------
    def score_window(self, window_result: Dict) -> Optional[str]:
        """
        סיווג חלון זמן:
          - אין דיבור (has_speech=False) -> None (רעש, לא נספר)
          - רעש (ללא דובר) -> None (לא נספר)
          - דובר יחיד -> "חיובי" (תמיד — מורה מדברת)
          - ריבוי + למידה/דיון -> "חיובי"
          - ריבוי + הפרעה -> "הפרעה"
        """
        # שליפת הדגל has_speech מתוך תוצאת החלון — האם זוהה דיבור בכלל?
        # ברירת מחדל True — אם המפתח חסר, מניחים שיש דיבור
        has_speech = window_result.get('has_speech', True)
        # אם אין דיבור בחלון הזה — זה רעש סביבתי בלבד, לא נספר בציון
        if not has_speech:
            return None

        # שליפת סוג הדובר שזוהה ע"י OverlapDetector: 'דובר_יחיד' / 'ריבוי_דוברים' / 'רעש'
        speaker = window_result.get('speaker_type', '')

        # אם OverlapDetector סיווג את החלון כרעש — לא נספר
        if speaker == 'רעש':
            return None

        # דובר יחיד = מורה מדברת לכיתה — תמיד חיובי ללא תלות בתוכן
        if speaker == 'דובר_יחיד':
            return self.POSITIVE

        # הגענו לכאן = ריבוי דוברים — צריך לבדוק הקשר פדגוגי
        # שליפת קטגוריית ההקשר שניתחה מערכת HEBERT
        context = window_result.get('context_category', '')
        # אם ריבוי דוברים נוצר בגלל דיון/משימה שהמורה פתחה — חיובי
        if context in ('למידה_פעילה', 'פתיחה_לדיון'):
            # דיון מובנה או משימה כיתתית — זה מה שאמור לקרות בשיעור טוב
            return self.POSITIVE
        # הגענו לכאן = ריבוי דוברים ללא הצדקה פדגוגית — הפרעה
        return self.NEGATIVE

    # ------- סיכום לשיעור שלם -------
    def score_lesson(self, window_labels: List[Optional[str]]) -> Dict:
        """
        חישוב אחוזי איכות לשיעור:
          - אחוז חלונות חיוביים (מתוך הרלוונטיים)
          - אחוז חלונות הפרעה
          - מספר חלונות רעש (לא נספרו)
        """
        # סינון חלונות הרעש (None) — רק חלונות עם דיבור אמיתי נספרים
        relevant = [l for l in window_labels if l is not None]
        # סופרים את כל החלונות כולל רעש — לצורך הדיווח
        total_with_noise = len(window_labels)
        # כמה חלונות היו רעש (ולא נספרו בציון)
        n_noise = total_with_noise - len(relevant)

        # מקרה קצה: אם כל החלונות היו רעש — מחזירים ציון N/A
        if not relevant:
            return {
                'positive_pct': 0,
                'negative_pct': 0,
                'noise_count': n_noise,
                'total_relevant': 0,
                'grade': 'N/A'
            }

        # ספירת חלונות חיוביים (דיבור מורה / דיון מובנה)
        n_positive = sum(1 for l in relevant if l == self.POSITIVE)
        # ספירת חלונות שליליים (הפרעות)
        n_negative = sum(1 for l in relevant if l == self.NEGATIVE)
        # חישוב אחוז חיובי: (חיובי / סה"כ רלוונטי) × 100, מעוגל לספרה אחת
        pos_pct = round(n_positive / len(relevant) * 100, 1)
        # חישוב אחוז הפרעה: (שלילי / סה"כ רלוונטי) × 100
        neg_pct = round(n_negative / len(relevant) * 100, 1)

        # --- המרה לדירוג מילולי לפי ספי אחוז ---
        # 80% ומעלה = שיעור מצוין — מעט מאוד הפרעות
        # if pos_pct >= 80:
        if pos_pct >= GRADE_EXCELLENT:
            grade = "מצוין"
        # 60-79% = שיעור טוב — הפרעות קיימות אך לא שולטות
        # elif pos_pct >= 60:
        elif pos_pct >= GRADE_GOOD:
            grade = "טוב"
        # 40-59% = שיעור בינוני — הפרעות משמעותיות
        # elif pos_pct >= 40:
        elif pos_pct >= GRADE_MEDIUM:
            grade = "בינוני"
        # מתחת ל-40% = שיעור ברמה נמוכה — הפרעות שולטות
        else:
            grade = "נמוך"

        # החזרת מילון עם כל נתוני הציון
        return {
            # אחוז הזמן שהשיעור היה חיובי (ללא הפרעות)
            'positive_pct': pos_pct,
            # אחוז הזמן שהיו הפרעות
            'negative_pct': neg_pct,
            # כמה חלונות היו רעש בלבד (לא נספרו)
            'noise_count': n_noise,
            # כמה חלונות עם דיבור אמיתי היו בשיעור
            'total_relevant': len(relevant),
            # הדירוג המילולי: מצוין / טוב / בינוני / נמוך
            'grade': grade
        }
