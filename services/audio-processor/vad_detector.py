# ===================================================
# vad_detector.py — אלגוריתם זיהוי פעילות דיבור (VAD)
# ===================================================
# VAD = Voice Activity Detection
# מזהה בכל חלון זמן האם קיים דיבור אנושי או לא.
#
# הזיהוי מבוסס על שילוב 4 מאפיינים אקוסטיים:
#   1. אנרגיה (RMS) — כמה חזק הצליל
#   2. קצב חציית אפס (ZCR) — תדירות שינויי סימן באות
#   3. מרכז ספקטרלי — "בהירות" הצליל (תדר ממוצע משוקלל)
#   4. שטחיות ספקטרלית — מידת ה"אקראיות" (רעש לבן = שטוח)
#
# כל מאפיין "מצביע" אם הפריים הוא דיבור.
# אם לפחות 2 מתוך 4 אומרים "דיבור" → הפריים נחשב כדיבור.

import numpy as np
from typing import List, Tuple, Dict
from config import (
    SAMPLE_RATE, FRAME_LENGTH_SEC, FRAME_HOP_SEC,
    VAD_ENERGY_PERCENTILE, VAD_MIN_SPEECH_SEC, VAD_MIN_SILENCE_SEC
)
from audio_utils import frame_audio, compute_rms, compute_zcr, compute_spectrum, compute_spectral_centroid, compute_spectral_flatness


class VADDetector:
    """מזהה פעילות קולית (Voice Activity Detection) מבוסס מאפיינים אקוסטיים."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        # המרת זמנים (שניות) למספר דגימות
        self.frame_len = int(FRAME_LENGTH_SEC * sample_rate)
        self.hop_len = int(FRAME_HOP_SEC * sample_rate)
        # ספי זמן מינימליים (בפריימים)
        self.min_speech_frames = int(VAD_MIN_SPEECH_SEC / FRAME_HOP_SEC)
        self.min_silence_frames = int(VAD_MIN_SILENCE_SEC / FRAME_HOP_SEC)

    # ===============================================
    # שלב 1: חילוץ מאפיינים אקוסטיים מהאות
    # ===============================================
    def _extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        חילוץ 4 מאפיינים אקוסטיים מכל פריים באות.
        כל מאפיין הוא מערך — ערך אחד לכל פריים (חלון של 25ms).
        """
        # חלוקת האות לפריימים
        frames = frame_audio(audio, self.frame_len, self.hop_len)

        # מאפיין 1: אנרגיה (RMS) — ערך גבוה = צליל חזק
        rms = compute_rms(frames)

        # מאפיין 2: קצב חציית אפס (ZCR) — ערך גבוה = רעש, נמוך = דיבור
        zcr = compute_zcr(frames)

        # מאפיין 3+4: מרכז ספקטרלי ושטחיות (דורשים FFT)
        spectrum, freqs = compute_spectrum(frames, self.sample_rate)
        centroid = compute_spectral_centroid(spectrum, freqs)
        flatness = compute_spectral_flatness(spectrum)

        return {
            'rms': rms, 'zcr': zcr,
            'centroid': centroid, 'flatness': flatness
        }

    # ===============================================
    # שלב 2: הצבעה — כל מאפיין מצביע "דיבור" או "לא"
    # ===============================================
    def _vote(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        מנגנון הצבעה: כל מאפיין "מצביע" בנפרד.
        ספים אדפטיביים — מחושבים מהנתונים עצמם (לא קבועים מראש).
        דיבור = לפחות 2 הצבעות מתוך 4.
        """
        # חישוב ספים אדפטיביים מתוך התפלגות הנתונים
        e_thresh = np.percentile(features['rms'], VAD_ENERGY_PERCENTILE)
        z_thresh = np.percentile(features['zcr'], 70)
        c_thresh = np.percentile(features['centroid'], 40)

        # כל מאפיין מצביע: 1 = דיבור, 0 = לא
        votes = (
            (features['rms'] > e_thresh).astype(int) +       # אנרגיה גבוהה
            (features['zcr'] < z_thresh).astype(int) +       # ZCR נמוך
            (features['centroid'] > c_thresh).astype(int) +  # מרכז ספקטרלי גבוה
            (features['flatness'] < 0.5).astype(int)         # לא שטוח (= מובנה)
        )
        # דיבור = לפחות 2 מתוך 4
        return votes >= 2

    # ===============================================
    # שלב 3: עיבוד-אחר — ניקוי תוצאות
    # ===============================================
    def _post_process(self, decisions: np.ndarray) -> np.ndarray:
        """
        ניקוי ההחלטות הגולמיות:
        - מקטע "דיבור" קצר מ-300ms → כנראה רעש רגעי → מוחקים
        - "שקט" קצר מ-200ms → כנראה הפסקה באמצע משפט → ממלאים
        """
        result = decisions.copy()

        # הסרת מקטעי דיבור קצרים מדי (כנראה רעש חד-פעמי)
        for start, end in self._find_runs(result, True):
            if end - start < self.min_speech_frames:
                result[start:end] = False

        # מילוי חורי שקט קצרים (כנראה הפסקה טבעית בתוך משפט)
        for start, end in self._find_runs(result, False):
            if end - start < self.min_silence_frames:
                result[start:end] = True

        return result

    # ===============================================
    # פונקציית עזר: מציאת רצפים במערך
    # ===============================================
    @staticmethod
    def _find_runs(arr: np.ndarray, value: bool) -> List[Tuple[int, int]]:
        """
        מוצאת רצפים רציפים של ערך מסוים (True או False) במערך.
        מחזירה רשימה של (התחלה, סוף) לכל רצף.
        """
        runs = []
        in_run = False
        start = 0
        for i, v in enumerate(arr):
            if v == value and not in_run:
                start, in_run = i, True
            elif v != value and in_run:
                runs.append((start, i))
                in_run = False
        if in_run:
            runs.append((start, len(arr)))
        return runs

    # ===============================================
    # ממשק ציבורי — הפונקציות שקוראים להן מבחוץ
    # ===============================================
    def detect(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """
        הפונקציה הראשית: מקבלת אות שמע ומחזירה רשימת מקטעי דיבור.
        כל מקטע = (שנייה_התחלה, שנייה_סיום).
        """
        # חילוץ מאפיינים → הצבעה → ניקוי → חילוץ מקטעים
        features = self._extract_features(audio)
        raw_decisions = self._vote(features)
        clean_decisions = self._post_process(raw_decisions)

        # המרה מאינדקס פריים לשניות
        segments = [
            (s * FRAME_HOP_SEC, e * FRAME_HOP_SEC)
            for s, e in self._find_runs(clean_decisions, True)
        ]
        return segments

    def get_speech_ratio(self, audio: np.ndarray) -> float:
        """חישוב אחוז הדיבור מתוך כלל האות (0.0 עד 1.0)."""
        segments = self.detect(audio)
        total = len(audio) / self.sample_rate
        speech = sum(end - start for start, end in segments)
        return speech / total if total > 0 else 0.0

    def get_statistics(self, audio: np.ndarray) -> Dict:
        """חישוב סטטיסטיקות מפורטות על הפעילות הקולית."""
        segments = self.detect(audio)
        total = len(audio) / self.sample_rate
        speech = sum(end - start for start, end in segments)
        return {
            'total_duration': total,
            'speech_duration': speech,
            'silence_duration': total - speech,
            'speech_ratio': speech / total if total > 0 else 0,
            'num_segments': len(segments),
            'avg_segment_duration': speech / len(segments) if segments else 0
        }
