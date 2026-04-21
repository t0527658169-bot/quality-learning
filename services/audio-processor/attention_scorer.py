# ===================================================
# attention_scorer.py — מדד איכות השיעור
# ===================================================
# מסווג כל חלון זמן כחיובי (דובר יחיד / ריבוי לצורך השיעור)
# או שלילי (ריבוי הפרעה). רעש ללא דוברים לא נספר.
#
# הציון הסופי: אחוז החלונות החיוביים מתוך הרלוונטיים.

import numpy as np
from typing import Dict, List, Optional


class AttentionScorer:
    """מסווג כל חלון כחיובי/שלילי ומחשב אחוזי איכות לשיעור."""

    # תוויות
    POSITIVE = "חיובי"     # דובר יחיד / ריבוי לצורך השיעור
    NEGATIVE = "הפרעה"     # ריבוי דוברים שלא לצורך השיעור
    NOISE = None           # רעש ללא דוברים — לא נספר

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
        # אם אין דיבור בחלון — רעש, לא נספר
        has_speech = window_result.get('has_speech', True)
        if not has_speech:
            return None

        speaker = window_result.get('speaker_type', '')

        # רעש ללא דובר — לא נספר
        if speaker == 'רעש':
            return None

        # דובר יחיד — תמיד חיובי (מורה מדברת)
        if speaker == 'דובר_יחיד':
            return self.POSITIVE

        # ריבוי דוברים — תלוי הקשר
        context = window_result.get('context_category', '')
        if context in ('למידה_פעילה', 'פתיחה_לדיון'):
            return self.POSITIVE  # דיון / משימה כיתתית
        return self.NEGATIVE      # הפרעה

    # ------- סיכום לשיעור שלם -------
    def score_lesson(self, window_labels: List[Optional[str]]) -> Dict:
        """
        חישוב אחוזי איכות לשיעור:
          - אחוז חלונות חיוביים (מתוך הרלוונטיים)
          - אחוז חלונות הפרעה
          - מספר חלונות רעש (לא נספרו)
        """
        relevant = [l for l in window_labels if l is not None]
        total_with_noise = len(window_labels)
        n_noise = total_with_noise - len(relevant)

        if not relevant:
            return {
                'positive_pct': 0,
                'negative_pct': 0,
                'noise_count': n_noise,
                'total_relevant': 0,
                'grade': 'N/A'
            }

        n_positive = sum(1 for l in relevant if l == self.POSITIVE)
        n_negative = sum(1 for l in relevant if l == self.NEGATIVE)
        pos_pct = round(n_positive / len(relevant) * 100, 1)
        neg_pct = round(n_negative / len(relevant) * 100, 1)

        # דירוג מילולי
        if pos_pct >= 80:
            grade = "מצוין"
        elif pos_pct >= 60:
            grade = "טוב"
        elif pos_pct >= 40:
            grade = "בינוני"
        else:
            grade = "נמוך"

        return {
            'positive_pct': pos_pct,
            'negative_pct': neg_pct,
            'noise_count': n_noise,
            'total_relevant': len(relevant),
            'grade': grade
        }
