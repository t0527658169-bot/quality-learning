# ===================================================
# main_pipeline.py — צנרת העיבוד הראשית
# ===================================================

import sys
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from collections import deque
from typing import Dict, List

from config import SAMPLE_RATE, WINDOW_DURATION_SEC,VAD_SPEECH_RATIO_THRESHOLD,NOISE_OVERLAP_SCORE
from wiener_filter import WienerFilter
from vad_detector import VADDetector
from rms_analyzer import RMSAnalyzer
from audio_classifier import AudioClassifier
from overlap_detector import OverlapDetector
from hebert_context_analyzer import HEBERTContextAnalyzer
from attention_scorer import AttentionScorer


class AudioPipeline:
    """צנרת העיבוד הראשית - מנהלת את כל שלבי הניתוח."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        # שמירת קצב הדגימה לשימוש בכל הפונקציות
        self.sr = sample_rate
        # אתחול כל מודולי הניתוח
        # פילטר וינר — ינקה רעשי רקע לפני כל עיבוד
        self.wiener = WienerFilter(sample_rate)
        # VAD — יזהה האם בכל חלון יש דיבור
        self.vad = VADDetector(sample_rate)
        # RMS — ימדוד את עוצמת הקול
        self.rms = RMSAnalyzer(sample_rate)
        # AudioClassifier — יסווג את סוג הצליל (שקט/רעש/דיבור)
        self.classifier = AudioClassifier(sample_rate)
        # OverlapDetector — יזהה דובר יחיד מול ריבוי דוברים
        self.overlap = OverlapDetector(sample_rate)
        # HEBERTContextAnalyzer — ינתח הקשר פדגוגי (שאלה פתוחה / השתקה)
        self.context = HEBERTContextAnalyzer(sample_rate)
        # AttentionScorer — יחשב ציון קשב לכל חלון ולשיעור כולו
        self.scorer = AttentionScorer()

    # ------- טעינת קובץ שמע -------
    def load_audio(self, file_path: str) -> np.ndarray:
        """טעינת קובץ שמע (WAV) והמרה לקצב דגימה אחיד."""
        # טעינת הקובץ: float32 = ייצוג נוח לחישובים, soundfile מחזיר (audio, sample_rate)
        # מספרים עשרונים ב32 ביטים שזה בין 1 ל 1- ממיר את האודיו ל
        audio, orig_sr = sf.read(file_path, dtype='float32')
        # אם הקובץ סטריאו (2 ערוצים) — ממוצע ל-mono
        # audio.ndim > 1 מזהה מטריצה (ערוצים × דגימות)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        # אם קצב הדגימה של הקובץ שונה מ-16kHz — ממירים
        if orig_sr != self.sr:
            # GCD מחשב מכנה משותף גדול ביותר לחישוב מדויק של יחס ה-resampling
            from math import gcd
            g = gcd(self.sr, orig_sr)
            # מרחיבים את המערך ואז מצמצים ביחס של הדגימה
            audio = resample_poly(audio, self.sr // g, orig_sr // g)
        return audio

    # ------- עיבוד קובץ שלם -------
    def process_file(self, file_path: str) -> Dict:
       
        print(f"\n{'='*60}")
        print(f"  Quality Learning — ניתוח שיעור")
        print(f"{'='*60}")
        # טעינת הקובץ לזיכרון כמערך numpy
        raw_audio = self.load_audio(file_path)
        # חישוב משך ההקלטה בשניות
        total_sec = len(raw_audio) / self.sr
        print(f"  קובץ: {file_path}")
        print(f"  משך:  {total_sec:.1f} שניות ({total_sec/60:.1f} דקות)")

        # שלב 1: ניקוי רעשים
        print(f"\n  [1/8] ניקוי רעשים (פילטר וינר)...")
        # פילטר וינר מנכה את פרופיל הרעש מחצי השנייה הראשונה מכל התדרים
        clean_audio = self.wiener.apply(raw_audio)

        # שלב 2: חלוקה לחלונות
        print(f"  [2/8] חלוקה לחלונות של {WINDOW_DURATION_SEC} שניות...")
        # מספר הדגימות בכל חלון = WINDOW_DURATION_SEC × SAMPLE_RATE
        window_samples = int(WINDOW_DURATION_SEC * self.sr)
        # רשימת תוצאות מפורטות לכל חלון
        results = []
        # רשימת תוויות (חיובי/הפרעה/None) לחישוב ציון שיעור
        labels = []

        # מכונת מצבים: None = רגיל, 'lesson' = לצורך השיעור, 'disruption' = הפרעה
        multi_state = None

        # כיול per-recording
        # אותו גודל חלון — משמש לחישוב harm_threshold ספציפי להקלטה הנוכחית
        window_samples_cal = int(WINDOW_DURATION_SEC * self.sr)
        # harm_threshold נגזר מפרצנטיל  של ה-harmonicity בהקלטה עצמה
        harm_threshold = self.overlap.calibrate(clean_audio, window_samples_cal)
        print(f"  [OD] כיול per-recording: harm_threshold={harm_threshold:.2f}", file=sys.stderr)

        #בפר של 2 חלוני דובר-יחיד אחרונים (6 שניות סה"כ) 
        # maxlen=2 = תור עגול — שומר רק 2 חלונות אחרונים אוטומטית
        prev_single_chunks: deque = deque(maxlen=2)

        # כותרת טבלה
        print(f"\n  {'זמן':<16} | {'דיבור':>6} | {'עוצמה':<6} | {'דוברים':<14} | {'סטטוס':>6}")
        print(f"  {'-'*55}")

        # לולאה על כל חלוני הניתוח (3 שניות כל אחד)
        for i in range(0, len(clean_audio), window_samples):
            # חיתוך חלון נוכחי מהאות הנקי
            chunk = clean_audio[i:i + window_samples]
            # דילוג על חלון קצר מדי מ-1 שנייה — לא מספיק לניתוח
            if len(chunk) >= self.sr:

                # זמן התחלה וסיום של החלון בשניות
                start_sec = i / self.sr
                end_sec = (i + len(chunk)) / self.sr

                # ניתוח החלון עם מכונת המצבים — מחזיר תוצאה ומצב חדש
                result, multi_state = self._analyze_window_stateful(
                    chunk, start_sec, end_sec, multi_state, prev_single_chunks, harm_threshold
                )

                # שמירת חלון דובר-יחיד  לניתוח הקשרי של הריבוי הבא
                if result['speaker_type'] == 'דובר_יחיד' and result['has_speech']:
                    prev_single_chunks.append(chunk)

                # הוספה לרשימות הסיכום
                results.append(result)
                labels.append(result['attention_label'])
                # הדפסת שורת סטטוס בזמן אמת
                self._print_row(result)

        # סיכום
        # חישוב ציון כולל לשיעור מכל התוויות שנאספו
        lesson_score = self.scorer.score_lesson(labels)
        print(f"\n{'='*60}")
        print(f"  איכות השיעור: {lesson_score['positive_pct']}% חיובי — {lesson_score['grade']}")
        print(f"  ({lesson_score['negative_pct']}% הפרעות | {lesson_score['noise_count']} חלונות רעש לא נספרו)")
        print(f"{'='*60}\n")

        # החזרת מילון מלא עם כל תוצאות הניתוח
        return {
            'file': file_path,
            'duration_sec': total_sec,
            'windows': results,
            'lesson_score': lesson_score
        }

    # ------- ניתוח חלון עם מכונת מצבים -------
    def _analyze_window_stateful(
        self,
        chunk: np.ndarray,
        start_sec: float,
        end_sec: float,
        multi_state,
        prev_single_chunks: deque,
        harm_threshold: float = 0.70
    ):

        # [3] VAD — בדיקה אם יש דיבור בחלון
        # get_speech_ratio מחזיר ערך 0.0-1.0: כמה מהחלון מכיל דיבור
        speech_ratio = self.vad.get_speech_ratio(chunk)
        # חלון עם 30%+ דיבור נחשב "יש דיבור" — פחות = רעש
        has_speech = speech_ratio > VAD_SPEECH_RATIO_THRESHOLD

        # [4] RMS — מדידת עוצמת הקול
        rms_result = self.rms.analyze_chunk(chunk, start_sec, end_sec)

        # [5] סיווג אקוסטי לא קשור לניתוח
        audio_type, audio_conf = self.classifier.classify(chunk)

        # [6] זיהוי חפיפת דוברים
        if not has_speech:
            speaker_type, overlap_score = OverlapDetector.NOISE, NOISE_OVERLAP_SCORE
        else:
            # harm_threshold — ערך שחושב בכיול ספציפי להקלטה הנוכחית
            speaker_type, overlap_score = self.overlap.detect(chunk, harm_threshold)

        # [7] ניתוח הקשרי — HEBERT
        # ערכי ברירת מחדל — יוחלפו בתוצאה אמיתית בהמשך
        context_result = {
            'context_category': 'לא_ידוע',
            'confidence': 0.0,
            'transcribed_text': None
        }

        # שמירת מצב המכונה הנוכחי לשימוש בסוף הפונקציה
        new_state = multi_state

        if speaker_type == 'דובר_יחיד' and has_speech:
            new_state = None
            context_result['context_category'] = 'דובר_יחיד'
            context_result['confidence'] = 1.0

        elif speaker_type == 'ריבוי_דוברים' and has_speech:
            # -------------------------------------------------------
            # ריבוי דוברים — החלטה לפי מכונת המצבים
            # -------------------------------------------------------

            if multi_state == 'lesson':
                # רצף ריבוי שהתחיל אחרי שאלה פתוחה.
                # בודק האם המורה ניסתה להשתיק בתוך הריבוי.
                is_silencing, sil_score, sil_text = self.context.check_silencing(chunk)
                if is_silencing:
                    # נמצאה השתקה — מהחלון הבא הריבוי = הפרעה
                    new_state = 'disruption'
                    context_result['context_category'] = 'הפרעה'
                    context_result['confidence'] = sil_score
                    context_result['transcribed_text'] = sil_text
                else:
                    # אין השתקה — ממשיך להיות לצורך השיעור
                    context_result['context_category'] = 'למידה_פעילה'
                    context_result['confidence'] = 0.9

            elif multi_state == 'disruption':
                # הריבוי הזה כבר הוגדר כהפרעה — ממשיך כך
                context_result['context_category'] = 'הפרעה'
                context_result['confidence'] = 0.9

            else:
                # state = None — בודק את 2 החלונות הקודמים (דובר יחיד)
                if prev_single_chunks:
                    combined = np.concatenate(list(prev_single_chunks))
                    is_opening, open_score, open_text = self.context.check_opening(combined)
                    if is_opening:
                        new_state = 'lesson'
                        context_result['context_category'] = 'למידה_פעילה'
                        context_result['confidence'] = open_score
                        context_result['transcribed_text'] = open_text
                    else:
                        # אין פתיחה לדיון — הפרעה
                        context_result['context_category'] = 'הפרעה'
                        context_result['confidence'] = 0.8
                else:
                    # אין חלונות קודמים כלל — ברירת מחדל הפרעה
                    context_result['context_category'] = 'הפרעה'
                    context_result['confidence'] = 0.7

        # [8] ציון קשב לחלון — AttentionScorer מקבל מילון קטן עם הנתונים הדרושים
        score_input = {
            # האם זוהה דיבור בכלל?
            'has_speech': has_speech,
            # רמת עוצמה: שקט / רגיל / רועש
            'rms_level': rms_result['level'],
            # דובר_יחיד / ריבוי_דוברים / רעש
            'speaker_type': speaker_type,
            # פתיחה_לדיון / למידה_פעילה / הפרעה / לא_ידוע
            'context_category': context_result['context_category']
        }
        # AttentionScorer מחזיר "חיובי" / "הפרעה" / None (רעש)
        attention_label = self.scorer.score_window(score_input)

        # בניית מילון תוצאה מלא לחלון זה
        result = {
            # זמן התחלה של החלון (שניות מתחילת ההקלטה)
            'start_sec': start_sec,
            # זמן סיום החלון
            'end_sec': end_sec,
            # אחוז הדיבור: 0.0 = שקט, 1.0 = דיבור רצוף
            'speech_ratio': speech_ratio,
            # האם יש דיבור (True/False)
            'has_speech': has_speech,
            # ערך RMS הגולמי
            'rms': rms_result['rms'],
            # ערך RMS בדציבלים
            'rms_db': rms_result['db'],
            # תווית עוצמה: שקט / רגיל / רועש
            'rms_level': rms_result['level'],
            # סוג אקוסטי: שקט / רעש_סביבתי / פטפטת_חלשה / דיבור_ברור
            'audio_type': audio_type,
            # סוג דובר: דובר_יחיד / ריבוי_דוברים / רעש
            'speaker_type': speaker_type,
            # ציון חפיפה: 0.0-1.0 (גבוה = ריבוי)
            'overlap_score': overlap_score,
            # קטגוריה הקשרית מ-HEBERT: פתיחה_לדיון / למידה_פעילה / הפרעה / לא_ידוע
            'context_category': context_result['context_category'],
            # תמלול (או None אם לא בוצע)
            'transcribed_text': context_result.get('transcribed_text'),
            # רמת ביטחון ההקשר (0.0–1.0) — עד כמה האלגוריתם בטוח בסיווג ה-context_category
            'context_confidence': context_result.get('confidence'),
            # מצב מכונת המצבים בסוף החלון: None / 'lesson' / 'disruption'
            'state_machine': new_state,
            # ציון הקשב הסופי: חיובי / הפרעה / None
            'attention_label': attention_label
        }
        # מחזיר את תוצאת החלון + מצב המכונה החדש (לשימוש בחלון הבא)
        return result, new_state

    # ------- הדפסת שורה בטבלה -------
    @staticmethod
    def _print_row(r: Dict):
        """הדפסת שורת סטטוס עבור חלון אחד."""
        # בניית מחרוזת זמן: '  5.0s- 8.0s'
        time_str = f"{r['start_sec']:5.1f}s-{r['end_sec']:5.1f}s"
        # שליפת תווית הקשב
        label = r['attention_label']
        # קביעת אייקון וסטטוס לפי התווית
        if label is None:
            # רעש — לא נספר בציון
            icon = " "
            status = "---"
        elif label == "חיובי":
            # חלון חיובי: דיבור מורה / דיון מובנה
            icon = "+"
            status = "חיובי"
        else:
            # חלון שלילי: הפרעה
            icon = "-"
            status = "הפרעה"
        # הדפסת שורה מפורמטת: זמן | דיבור% | עוצמה | דוברים | סטטוס
        print(f"  {icon} {time_str} | {r['speech_ratio']:5.0%} | {r['rms_level']:<6} | "
              f"{r['speaker_type']:<14} | {status}")