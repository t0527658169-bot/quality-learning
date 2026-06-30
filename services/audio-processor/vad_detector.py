# ===================================================
# vad_detector.py — זיהוי פעילות דיבור (VAD)
# ===================================================


import numpy as np
from typing import List, Tuple, Dict
# ייבוא כל פרמטרי ה-VAD מקובץ ההגדרות המרכזי
from config import (
    # קצב דגימה ואורכי פריימים בשניות
    SAMPLE_RATE, FRAME_LENGTH_SEC, FRAME_HOP_SEC,
    # פרצנטיל אנרגיה לסף אדפטיבי, זמני מינימום לדיבור ושקט
    VAD_ENERGY_PERCENTILE, VAD_MIN_SPEECH_SEC, VAD_MIN_SILENCE_SEC,
    # רמת אגרסיביות WebRTC VAD: 0=רגיש, 3=אגרסיבי
    VAD_WEBRTC_AGGRESSIVENESS,
    # ספי fallback: שטחיות, ZCR, מרכז ספקטרלי
    VAD_FLATNESS_THRESHOLD, VAD_ZCR_PERCENTILE, VAD_CENTROID_PERCENTILE,
    # רצפת RMS מינימלית ויחס לממוצע — מגנים על הקלטות חלשות
    VAD_ABS_FLOOR, VAD_ABS_FLOOR_RATIO
)


try:
    import webrtcvad
    # דגל גלובלי: True = ספריה זמינה ואפשר להשתמש בה
    WEBRTC_AVAILABLE = True
# אם לא מותקן — Python זורק ImportError
except ImportError:
    # דגל גלובלי: False = נשתמש ב-fallback מבוסס מאפיינים
    WEBRTC_AVAILABLE = False

# ייבוא פונקציות עזר לעיבוד אות מ-audio_utils.py
from audio_utils import (
    # חיתוך לפריימים, חישוב RMS, ZCR, FFT, מרכז ספקטרלי ושטחיות
    frame_audio, compute_rms, compute_zcr, compute_spectrum,
    compute_spectral_centroid, compute_spectral_flatness
)


# מחלקת זיהוי פעילות קולית — תומכת ב-WebRTC ובמצב fallback
class VADDetector:

    # def __init__(self, sample_rate: int = SAMPLE_RATE, aggressiveness: int = 2):
    # אתחול הגלאי: קצב דגימה ורמת אגרסיביות WebRTC (0-3)
    def __init__(self, sample_rate: int = SAMPLE_RATE, aggressiveness: int = VAD_WEBRTC_AGGRESSIVENESS):
        # שמירת קצב הדגימה — משמש לחישוב אורך פריים בדגימות
        self.sample_rate = sample_rate
        # אורך פריים בדגימות: FRAME_LENGTH_SEC שניות × sample_rate דגימות/שנייה
        self.frame_len   = int(FRAME_LENGTH_SEC * sample_rate)
        # קפיצה בין פריימים (hop): FRAME_HOP_SEC שניות — חפיפה בין פריימים
        self.hop_len     = int(FRAME_HOP_SEC * sample_rate)
        # מינימום פריימי דיבור: VAD_MIN_SPEECH_SEC ÷ FRAME_HOP_SEC — מסנן רעש קצר
        self.min_speech_frames  = int(VAD_MIN_SPEECH_SEC / FRAME_HOP_SEC)
        # מינימום פריימי שקט: VAD_MIN_SILENCE_SEC ÷ FRAME_HOP_SEC — מסנן הפסקות קצרות
        self.min_silence_frames = int(VAD_MIN_SILENCE_SEC / FRAME_HOP_SEC)

        # בדיקה אם WebRTC זמין — נבחר את נתיב העיבוד המתאים
        if WEBRTC_AVAILABLE:
            # יצירת אובייקט WebRTC VAD עם רמת אגרסיביות שהוגדרה ב-config
            self._vad = webrtcvad.Vad(aggressiveness)
            # הודעת אתחול: WebRTC פעיל עם רמת אגרסיביות
            print(f"  [V] VAD: WebRTC (aggressiveness={aggressiveness})")
        else:
            # WebRTC לא זמין — מאפסים לNone כדי שהקוד יבחר fallback
            self._vad = None
            # הודעת אזהרה: fallback יפעיל ניתוח מאפיינים אקוסטיים במקום
            print("  [!] VAD: webrtcvad לא מותקן — משתמש ב-fallback")

    # -------------------------------------------------------
    # WebRTC VAD
    # -------------------------------------------------------
    def _webrtc_decisions(self, audio: np.ndarray) -> np.ndarray:
       
        # המרה ל-int16
        if audio.dtype != np.int16:
            audio_int16 = (audio * 32767).astype(np.int16)
        else:
            audio_int16 = audio

        frame_ms  = 10   # WebRTC דורש 10/20/30ms
        frame_len = int(self.sample_rate * frame_ms / 1000)

        decisions = []
        for start in range(0, len(audio_int16) - frame_len, frame_len):
            frame = audio_int16[start:start + frame_len]
            frame_bytes = frame.tobytes()
            try:
                is_speech = self._vad.is_speech(frame_bytes, self.sample_rate)
            except Exception:
                is_speech = False
            decisions.append(is_speech)

        return np.array(decisions, dtype=bool)

    def _webrtc_speech_ratio(self, audio: np.ndarray) -> float:
        decisions = self._webrtc_decisions(audio)
        if len(decisions) == 0:
            return 0.0

        # post-process: מילוי חורים קצרים
        frame_ms  = 10
        min_speech_frames  = int(VAD_MIN_SPEECH_SEC  * 1000 / frame_ms)
        min_silence_frames = int(VAD_MIN_SILENCE_SEC * 1000 / frame_ms)

        result = decisions.copy()
        for start, end in self._find_runs(result, True):
            if end - start < min_speech_frames:
                result[start:end] = False
        for start, end in self._find_runs(result, False):
            if end - start < min_silence_frames:
                result[start:end] = True

        return float(np.mean(result))

    # -------------------------------------------------------
    # Fallback VAD 
    # -------------------------------------------------------
    def _fallback_extract(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        # חלוקת האות לפריימים לפי אורך הפריים וקפיצה שהוגדרו ב-__init__
        frames = frame_audio(audio, self.frame_len, self.hop_len)
        # חישוב RMS (עוצמה) לכל פריים — ערך גבוה = צליל חזק
        rms= compute_rms(frames)
        # חישוב ZCR (קצב חציית אפס) לכל פריים — גבוה ברעש אקראי
        zcr = compute_zcr(frames)
        # חישוב FFT לכל פריים + וקטור התדרים המתאים
        spectrum, freqs = compute_spectrum(frames, self.sample_rate)
        # מרכז ספקטרלי — "בהירות" הצליל (תדר ממוצע משוקלל)
        centroid = compute_spectral_centroid(spectrum, freqs)
        # שטחיות ספקטרלית — 1.0 = רעש לבן שטוח, 0 = שיא בודד
        flatness = compute_spectral_flatness(spectrum)
        # החזרת מילון עם כל 4 המאפיינים — שמות מפתח אחידים לשימוש ב-_fallback_vote
        return {'rms': rms, 'zcr': zcr, 'centroid': centroid, 'flatness': flatness}

    def _fallback_vote(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        # סף אנרגיה אדפטיבי: ה-30% התחתונים של ה-RMS נחשבים שקטים
        e_thresh = np.percentile(features['rms'], VAD_ENERGY_PERCENTILE)
        # סף ZCR אדפטיבי: ה-70% התחתונים = גבול בין דיבור לרעש
        z_thresh = np.percentile(features['zcr'], VAD_ZCR_PERCENTILE)
        # סף מרכז ספקטרלי אדפטיבי: ה-40% התחתונים (דיבור = מרכז גבוה)
        c_thresh = np.percentile(features['centroid'], VAD_CENTROID_PERCENTILE)

        # סף מינימלי מוחלט
        # ממוצע ה-RMS של כל האות — נותן תמונה על רמת הרעש הבסיסית
        rms_mean  = float(np.mean(features['rms']))
        # רצפה: max(0.015, ממוצע×0.4) — מונעת ספים נמוכים מדי בהקלטות חלשות
        abs_floor = max(VAD_ABS_FLOOR, rms_mean * VAD_ABS_FLOOR_RATIO)
        # הגדלת ספף האנרגיה לפחות לרצפה — מגן מפני שגיאות בהקלטות שקטות
        e_thresh  = max(e_thresh, abs_floor)

        votes = (
            (features['rms'] > e_thresh).astype(int) +
            (features['zcr'] < z_thresh).astype(int) +
            (features['centroid'] > c_thresh).astype(int) +
            (features['flatness'] < VAD_FLATNESS_THRESHOLD).astype(int)
        )
        # רוב: 2 מתוך 4 הצביעו לדיבור → הפריים נחשב דיבור
        return votes >= 2

    def _fallback_speech_ratio(self, audio: np.ndarray) -> float:
        # חילוץ 4 מאפיינים מכל הפריימים
        features = self._fallback_extract(audio)
        # הצבעת כל פריים: True=דיבור, False=שקט — לפי רוב ההצבעות
        raw      = self._fallback_vote(features)
        # עותק שניתן לשנות — לא משנים את raw ישירות
        result   = raw.copy()
        # הסרת רצפי דיבור קצרים מדי (< min_speech_frames) — הם כנראה רעש רגעי
        for start, end in self._find_runs(result, True):
            if end - start < self.min_speech_frames:
                # רצף דיבור קצר מ-300ms — מסיר אותו (הופך לשקט)
                result[start:end] = False
        # מילוי חורי שקט קצרים (< min_silence_frames) — הפסקה קצרה = חלק מהדיבור
        for start, end in self._find_runs(result, False):
            if end - start < self.min_silence_frames:
                # חור שקט קצר מ-200ms בתוך דיבור — ממלא אותו (הופך לדיבור)
                result[start:end] = True
        # אחוז הפריימים שסווגו כדיבור מתוך כל הפריימים
        return float(np.mean(result))

    # -------------------------------------------------------
    # ממשק ציבורי
    # -------------------------------------------------------
    @staticmethod
    def _find_runs(arr: np.ndarray, value: bool) -> List[Tuple[int, int]]:
        # רשימת הרצפים שנמצאו, דגל in_run = האם אנחנו בתוך רצף, start = אינדקס התחלת הרצף הנוכחי
        runs, in_run, start = [], False, 0
        # מעבר על כל פריים בסדר
        for i, v in enumerate(arr):
            # זוהה תחילת רצף חדש של הערך המבוקש — שמירת האינדקס
            if v == value and not in_run:
                start, in_run = i, True
            # סוף הרצף הנוכחי — שמירת הזוג (התחלה, סיום)
            elif v != value and in_run:
                runs.append((start, i))
                in_run = False
        # אם הרצף האחרון מגיע עד סוף המערך — סוגר אותו
        if in_run:
            runs.append((start, len(arr)))
        return runs

    def detect(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """מחזיר רשימת מקטעי דיבור (התחלה, סיום) בשניות."""
        # נתיב ראשי: WebRTC VAD זמין — משתמשים בו (דיוק גבוה יותר)
        if self._vad is not None:
            # קבלת החלטות בינאריות (True/False) לכל פריים של 10ms
            decisions = self._webrtc_decisions(audio)
            # WebRTC עובד בפריימים של 10ms — משמש להמרה חזרה לשניות
            frame_ms  = 10
            # עותק לעיבוד post-processing ללא שינוי המקור
            result    = decisions.copy()
            # המרת ספי זמן (שניות → מספר פריימים של 10ms)
            min_s = int(VAD_MIN_SPEECH_SEC  * 1000 / frame_ms)
            # מינימום שקט: 200ms = 20 פריימים של 10ms
            min_q = int(VAD_MIN_SILENCE_SEC * 1000 / frame_ms)
            # הסרת רצפי דיבור קצרים מדי — רעש רגעי, לא דיבור אמיתי
            for start, end in self._find_runs(result, True):
                if end - start < min_s:
                    result[start:end] = False
            # מילוי חורי שקט קצרים — הפסקה טבעית בין מילים, לא סוף מקטע
            for start, end in self._find_runs(result, False):
                if end - start < min_q:
                    result[start:end] = True
            # המרת אינדקסי פריים לשניות: אינדקס × 0.010 שניות/פריים
            dt = frame_ms / 1000
            # בניית רשימת זוגות (התחלה_בשניות, סיום_בשניות)
            return [(s * dt, e * dt) for s, e in self._find_runs(result, True)]
        else:
            # נתיב גיבוי: WebRTC לא מותקן — משתמשים בפרמטרים אדפטיביים
            features = self._fallback_extract(audio)
            # הצבעת כל פריים לפי 4 מאפיינים
            raw  = self._fallback_vote(features)
            # עותק לעיבוד ללא שינוי המקור
            result   = raw.copy()
            # הסרת רצפי דיבור קצרים מדי
            for start, end in self._find_runs(result, True):
                if end - start < self.min_speech_frames:
                    result[start:end] = False
            # מילוי חורי שקט קצרים
            for start, end in self._find_runs(result, False):
                if end - start < self.min_silence_frames:
                    result[start:end] = True
            # המרת אינדקסי פריים לשניות: אינדקס × FRAME_HOP_SEC (0.010 שניות)
            return [
                (s * FRAME_HOP_SEC, e * FRAME_HOP_SEC)
                for s, e in self._find_runs(result, True)
            ]

    def get_speech_ratio(self, audio: np.ndarray) -> float:
        """חישוב אחוז הדיבור (0.0 עד 1.0)."""
        # אם WebRTC זמין — משתמשים בו (מדויק יותר, ספציפי לדיבור)
        if self._vad is not None:
            return self._webrtc_speech_ratio(audio)
        else:
            # אחרת — fallback עם 4 מאפיינים אקוסטיים
            return self._fallback_speech_ratio(audio)

    # def get_statistics(self, audio: np.ndarray) -> Dict:
    #     # זיהוי כל מקטעי הדיבור — רשימת זוגות (התחלה, סיום) בשניות
    #     segments = self.detect(audio)
    #     # משך כל ההקלטה בשניות: מספר דגימות חלקי קצב הדגימה
    #     total    = len(audio) / self.sample_rate
    #     # סכום כל משכי מקטעי הדיבור: sum(סיום - התחלה) לכל מקטע
    #     speech   = sum(e - s for s, e in segments)
    #     return {
    #         # משך ההקלטה הכולל בשניות
    #         'total_duration':        total,
    #         # סך שניות הדיבור
    #         'speech_duration':       speech,
    #         # סך שניות השקט (כולל = דיבור + שקט)
    #         'silence_duration':      total - speech,
    #         # יחס דיבור: 0.0 = שקט מוחלט, 1.0 = דיבור כל הזמן
    #         'speech_ratio':          speech / total if total > 0 else 0,
    #         # מספר מקטעי הדיבור שזוהו
    #         'num_segments':          len(segments),
    #         # משך מקטע ממוצע: 0 אם אין מקטעים (מניעת חלוקה באפס)
    #         'avg_segment_duration':  speech / len(segments) if segments else 0
    #     }