# ===================================================
# main_pipeline.py — DISABLED (data-service — גרסה ישנה)
# *** קובץ זה שייך לגרסה הישנה לפני HEBERT ***
# *** ניתן למחוק את תיקיית data-service/audio-processor כולה ***
# ===================================================

# ===================================================
# main_pipeline.py — צנרת העיבוד הראשית
# ===================================================
# מרכיב את כל האלגוריתמים לתהליך עיבוד רציף:
#
#   קובץ שמע
#     ↓
#   [1] פילטר וינר — ניקוי רעשים
#     ↓
#   [2] חלוקה לחלונות (3 שניות כל אחד)
#     ↓  לכל חלון:
#   [3] VAD — האם יש דיבור?
#   [4] RMS — מה עוצמת הקול?
#   [5] סיווג אקוסטי — מה סוג הצליל?
#   [6] זיהוי חפיפה — דובר יחיד או ריבוי?
#     ↓  אם ריבוי דוברים:
#   [7] ASR + NLP — ניתוח הקשרי (למידה פעילה / הפרעה)
#     ↓
#   [8] ממוצע משוקלל — ציון קשב לחלון
#     ↓
#   ציון כולל לשיעור

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from typing import Dict, List

from config import SAMPLE_RATE, WINDOW_DURATION_SEC
from wiener_filter import WienerFilter
from vad_detector import VADDetector
from rms_analyzer import RMSAnalyzer
from audio_classifier import AudioClassifier
from overlap_detector import OverlapDetector
from context_analyzer import ContextAnalyzer
from attention_scorer import AttentionScorer


class AudioPipeline:
    """צנרת העיבוד הראשית — מנהלת את כל שלבי הניתוח."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sr = sample_rate
        # אתחול כל מודולי הניתוח
        self.wiener = WienerFilter(sample_rate)
        self.vad = VADDetector(sample_rate)
        self.rms = RMSAnalyzer(sample_rate)
        self.classifier = AudioClassifier(sample_rate)
        self.overlap = OverlapDetector(sample_rate)
        self.context = ContextAnalyzer(sample_rate)
        self.scorer = AttentionScorer()

    # ------- טעינת קובץ שמע -------
    def load_audio(self, file_path: str) -> np.ndarray:
        """טעינת קובץ שמע (WAV/MP3) והמרה לקצב דגימה אחיד."""
        # טעינה באמצעות soundfile (יציב יותר מ-librosa.load)
        audio, orig_sr = sf.read(file_path, dtype='float32')
        # המרה למונו אם צריך
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        # המרת קצב דגימה אם שונה מהמבוקש
        if orig_sr != self.sr:
            from math import gcd
            g = gcd(self.sr, orig_sr)
            audio = resample_poly(audio, self.sr // g, orig_sr // g)
        return audio

    # ------- עיבוד קובץ שלם -------
    def process_file(self, file_path: str) -> Dict:
        """
        עיבוד קובץ שמע מתחילה ועד סוף.
        משתמש במכונת מצבים לזיהוי ריבוי דוברים לצורך השיעור:
          - מצב רגיל: ריבוי = הפרעה (ברירת מחדל)
          - מצב "לצורך השיעור": מופעל כשזוהתה שאלה פתוחה/הנחיה בחלון הקודם
          - מצב "הפרעה": מופעל כשזוהו מילות השתקה
        """
        # טעינת הקובץ
        print(f"\n{'='*60}")
        print(f"  Quality Learning — ניתוח שיעור")
        print(f"{'='*60}")
        raw_audio = self.load_audio(file_path)
        total_sec = len(raw_audio) / self.sr
        print(f"  קובץ: {file_path}")
        print(f"  משך:  {total_sec:.1f} שניות ({total_sec/60:.1f} דקות)")

        # שלב 1: ניקוי רעשים באמצעות פילטר וינר
        print(f"\n  [1/8] ניקוי רעשים (פילטר וינר)...")
        clean_audio = self.wiener.apply(raw_audio)

        # שלב 2: חלוקה לחלונות וניתוח כל חלון
        print(f"  [2/8] חלוקה לחלונות של {WINDOW_DURATION_SEC} שניות...")
        window_samples = int(WINDOW_DURATION_SEC * self.sr)
        results = []
        labels = []

        # מכונת מצבים: None = רגיל, 'lesson' = לצורך השיעור, 'disruption' = הפרעה
        multi_state = None

        # כותרת טבלה
        print(f"\n  {'זמן':<16} | {'דיבור':>6} | {'עוצמה':<6} | {'דוברים':<14} | {'סטטוס':>6}")
        print(f"  {'-'*55}")

        for i in range(0, len(clean_audio), window_samples):
            chunk = clean_audio[i:i + window_samples]
            # דילוג על חלון קצר מדי (פחות משנייה)
            if len(chunk) < self.sr:
                continue

            start_sec = i / self.sr
            end_sec = (i + len(chunk)) / self.sr

            # ניתוח החלון עם מכונת המצבים
            result, multi_state = self._analyze_window_stateful(
                chunk, raw_audio, i, start_sec, end_sec, multi_state
            )
            results.append(result)
            labels.append(result['attention_label'])

            # הדפסת שורת סטטוס
            self._print_row(result)

        # שלב 3: סיכום איכות השיעור
        lesson_score = self.scorer.score_lesson(labels)
        print(f"\n{'='*60}")
        print(f"  איכות השיעור: {lesson_score['positive_pct']}% חיובי — {lesson_score['grade']}")
        print(f"  ({lesson_score['negative_pct']}% הפרעות | {lesson_score['noise_count']} חלונות רעש לא נספרו)")
        print(f"{'='*60}\n")

        return {
            'file': file_path,
            'duration_sec': total_sec,
            'windows': results,
            'lesson_score': lesson_score
        }

    # ------- ניתוח חלון עם מכונת מצבים -------
    def _analyze_window_stateful(self, chunk, raw_audio, offset, start_sec, end_sec, multi_state):
        """
        ניתוח חלון עם מכונת מצבים לריבוי דוברים.

        סדר עדיפויות לקביעת הקשר ריבוי דוברים:
          1. מודל RF מאומן מתיקוני המשתמש (אם קיים) — עדיפות ראשונה
          2. מכונת מצבים (שאלה פתוחה / מילות השתקה) — משלים
          3. ניתוח ASR + מודל טקסט — גיבוי

        מצבים:
          None         = רגיל
          'lesson'     = לצורך השיעור (ריבוי = חיובי)
          'disruption' = הפרעה (ריבוי = שלילי)

        מעברי מצב:
          1. דובר יחיד + שאלה פתוחה/הנחיה → lesson
          2. ריבוי + מילות השתקה → disruption
          3. דובר יחיד (רגיל, ללא שאלה) → איפוס ל-None
        """
        # [3] VAD: האם יש דיבור בחלון?
        speech_ratio = self.vad.get_speech_ratio(chunk)
        has_speech = speech_ratio > 0.3

        # [4] RMS: מה רמת העוצמה?
        rms_result = self.rms.analyze_chunk(chunk, start_sec, end_sec)

        # [5] סיווג אקוסטי: מה סוג הצליל?
        audio_type, audio_conf = self.classifier.classify(chunk)

        # [6] זיהוי חפיפת דוברים
        speaker_type, overlap_score = self.overlap.detect(chunk)

        # [7] ניתוח הקשרי — משולב: מודל מאומן + מכונת מצבים
        context_result = {
            'context_category': 'לא_ידוע',
            'confidence': 0,
            'transcribed_text': None
        }

        new_state = multi_state

        if speaker_type == 'דובר_יחיד' and has_speech:
            # דובר יחיד — בדיקה אם יש שאלה פתוחה שתפעיל מצב "לצורך השיעור"
            if self.context.has_open_question(chunk):
                new_state = 'lesson'
                context_result['context_category'] = 'פתיחה_לדיון'
                context_result['confidence'] = 0.9
            else:
                # ניסיון עם מודל trigger — לומד מתיקוני המשתמש
                trigger_pred = self.context.predict_trigger(chunk)
                if trigger_pred is not None:
                    new_state = trigger_pred
                else:
                    # דובר יחיד רגיל — איפוס מצב
                    new_state = None

        elif speaker_type == 'ריבוי_דוברים' and has_speech:
            # --- עיקרון: state נקבע מחלון דובר יחיד (המורה) ---
            # כל עוד יש ריבוי דוברים רצוף, ממשיכים את ה-state.
            # ASR לא אמין על ריבוי דוברים, לכן לא בודקים מילות מפתח כאן.
            # ה-state משתנה רק כשמגיע חלון דובר יחיד חדש.

            if multi_state == 'lesson':
                # רצף ריבוי שהתחיל אחרי שאלה פתוחה → לצורך השיעור
                context_result['context_category'] = 'למידה_פעילה'
                context_result['confidence'] = 0.9
            elif multi_state == 'disruption':
                # רצף ריבוי שהתחיל אחרי הפרעה → הפרעה
                context_result['context_category'] = 'הפרעה'
                context_result['confidence'] = 0.9

            # --- אין state? מודל RF מאומן מתיקוני המשתמש ---
            elif self.context.rf_model is not None:
                context_result = self.context._analyze_with_rf(chunk)
                if context_result['context_category'] in ('למידה_פעילה', 'פתיחה_לדיון'):
                    new_state = 'lesson'
                elif context_result['context_category'] == 'הפרעה':
                    new_state = 'disruption'

            # --- מודל trigger — חלון קודם מנבא הקשר ---
            elif self.context.trigger_model is not None:
                trigger_pred = self.context.predict_trigger(chunk)
                if trigger_pred == 'lesson':
                    new_state = 'lesson'
                    context_result['context_category'] = 'למידה_פעילה'
                    context_result['confidence'] = 0.8
                elif trigger_pred == 'disruption':
                    new_state = 'disruption'
                    context_result['context_category'] = 'הפרעה'
                    context_result['confidence'] = 0.8

            # --- ניתוח ASR רגיל (גיבוי אחרון) ---
            else:
                lookback_samples = int(5 * self.sr)
                start_idx = max(0, offset - lookback_samples)
                preceding = raw_audio[start_idx:offset]
                if len(preceding) > self.sr:
                    context_result = self.context.analyze(preceding)

        # [8] סיווג איכות חלון
        score_input = {
            'has_speech': has_speech,
            'rms_level': rms_result['level'],
            'speaker_type': speaker_type,
            'context_category': context_result['context_category']
        }
        attention_label = self.scorer.score_window(score_input)

        # החזרת כל הנתונים של החלון + מצב חדש
        result = {
            'start_sec': start_sec,
            'end_sec': end_sec,
            'speech_ratio': speech_ratio,
            'has_speech': has_speech,
            'rms': rms_result['rms'],
            'rms_db': rms_result['db'],
            'rms_level': rms_result['level'],
            'audio_type': audio_type,
            'speaker_type': speaker_type,
            'overlap_score': overlap_score,
            'context_category': context_result['context_category'],
            'transcribed_text': context_result.get('transcribed_text'),
            'attention_label': attention_label
        }
        return result, new_state

    # ------- הדפסת שורה בטבלה -------
    @staticmethod
    def _print_row(r: Dict):
        """הדפסת שורת סטטוס עבור חלון אחד."""
        time_str = f"{r['start_sec']:5.1f}s-{r['end_sec']:5.1f}s"
        label = r['attention_label']
        if label is None:
            icon = " "
            status = "---"
        elif label == "חיובי":
            icon = "+"
            status = "חיובי"
        else:
            icon = "-"
            status = "הפרעה"
        print(f"  {icon} {time_str} | {r['speech_ratio']:5.0%} | {r['rms_level']:<6} | "
              f"{r['speaker_type']:<14} | {status}")


# ===================================================
# DISABLED: הקוד מעל אינו פעיל
# ===================================================
raise SystemExit(f"[DISABLED] main_pipeline.py — גרסה ישנה, אינו פעיל")
