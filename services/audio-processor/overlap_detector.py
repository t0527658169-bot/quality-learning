
# ===================================================
# overlap_detector.py — זיהוי דובר יחיד / ריבוי דוברים
# ===================================================
#          מבוסס Wrigley et al. (2005) + Yousefi & Hansen (2020)
# ===================================================

import numpy as np
import sys
from typing import Tuple
from config import (
    SAMPLE_RATE, OVERLAP_THRESHOLD, NOISE_RMS_FLOOR, KURTOSIS_SINGLE_VETO,
    FRAME_SILENCE_RMS, HARM_DEFAULT_THRESHOLD, HARMONICITY_NOISE_VETO
)
# יייבוא פונקציות עזר לחישוב מאפיינים אקוסטיים
from audio_utils import (
    frame_audio, compute_rms, compute_zcr, compute_spectrum,
    compute_spectral_centroid, compute_spectral_bandwidth,
    compute_spectral_flatness
)


class OverlapDetector:
    
    # --- קבועי תוויות ---
   
    SINGLE   = "דובר_יחיד"
    # תווית ריבוי דוברים (יותר מאחד מדבר בו-זמנית)
    MULTIPLE = "ריבוי_דוברים"
    # תווית רעש (ללא דיבור אנושי) — משמשת ב-main_pipeline
    NOISE    = "רעש"

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        # שמירת קצב הדגימה לשימוש בחישובי פריימים
        self.sample_rate = sample_rate
        # הדפסת הודעה שהמנגנון עלה ופועל 
        print("  [V] Speaker detector: rule-based (no ML)", file=sys.stderr)

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        # חלוקה לפריימים: 512 דגימות = ~32ms, קפיצה של 256 = 50% חפיפה
        frames    = frame_audio(audio, 512, 256)
        # חישוב RMS לכל פריים — עוצמת הצליל
        rms       = compute_rms(frames)
        # חישוב ZCR לכל פריים — קצב חציית אפס
        zcr       = compute_zcr(frames)
        # חישוב ספקטרום וחילוץ תדרים
        spectrum, freqs = compute_spectrum(frames, self.sample_rate)
        # מרכז ספקטרלי — "בהירות" הצליל
        centroid  = compute_spectral_centroid(spectrum, freqs)
        #  כמה מפוזר בטווח. רוחב פס — כמה רחב הצליל בתדרים
        bandwidth = compute_spectral_bandwidth(spectrum, freqs, centroid)
        # שטחיות ספקטרלית — 1=רעש לבן, 0=צליל טהור
        flatness  = compute_spectral_flatness(spectrum)
        spectral_diff = np.diff(spectrum, axis=0)
        spectral_flux = float(np.mean(np.sqrt(np.mean(spectral_diff ** 2, axis=1))))
        # מחזירים וקטור מאפיינים אחד לכל הקלטה
        return np.array([
            # energy_cv: שונות יחסית של האנרגיה — גבוה בריבוי דוברים
            float(np.std(rms) / (np.mean(rms) + 1e-10)),
            # bandwidth_norm: ממוצע רוחב הפס מנורמל ל-4000Hz
            float(np.mean(bandwidth) / 4000),
            # flatness_std: שונות שטחיות — גבוה בריבוי (צלילים שונים)
            float(np.std(flatness)),
            # zcr_std: שונות ZCR — גבוה בריבוי דוברים
            float(np.std(zcr)),
            # centroid_cv: שונות יחסית של מרכז ספקטרלי
            float(np.std(centroid) / (np.mean(centroid) + 1e-10)),
            # rms_range: טווח האנרגיה — גבוה כשיש שינויים גדולים
            float(np.max(rms) - np.min(rms)),
            # spectral_flux: קצב שינוי הספקטרום
            spectral_flux
        ])

    def _compute_kurtosis(self, audio: np.ndarray) -> float:
       
        frame_len = int(0.025 * self.sample_rate)
        # קפיצה של 10ms בין פריימים — חפיפה של 60%
        hop  = int(0.010 * self.sample_rate)
        # רשימת ערכי kurtosis מכל הפריימים הפעילים
        kvals = []
        # לולאה על כל הפריימים
        for start in range(0, len(audio) - frame_len, hop):
            # המרה ל-float64 לדיוק מספרי גבוה
            frame = audio[start:start + frame_len].astype(np.float64)
            # דילוג על פריימים שקטים (RMS < 0.01) — לא מחשבים kurtosis על רעש
            # if float(np.sqrt(np.mean(frame ** 2))) < 0.01:
            if float(np.sqrt(np.mean(frame ** 2))) < FRAME_SILENCE_RMS:
                continue
            # m2 = מומנט שני (שונות) — np.mean(x²)
            m2 = float(np.mean(frame ** 2))
            # m4 = מומנט רביעי — np.mean(x⁴)
            m4 = float(np.mean(frame ** 4))
            # מניעת חלוקה באפס
            if m2 < 1e-14:
                continue
            # kurtosis = m4/m2² — מדד "שפיצות" ההתפלגות
            kvals.append(m4 / (m2 ** 2))
        # חציון על כל הפריימים — עמיד לערכים קיצוניים
        # אם לא היו פריימים פעילים — מחזירים 3.0 (ברירת מחדל גאוסי)
        return float(np.median(kvals)) if kvals else 3.0

    def _compute_harmonicity(self, audio: np.ndarray) -> float:
        
        frame_len = int(0.025 * self.sample_rate)
        # קפיצה של 10ms
        hop  = int(0.010 * self.sample_rate)
        # גבול תחתון של lag: תדר מקסימלי 500Hz (קצר ביותר — גבוה בקול)
        min_lag   = int(self.sample_rate / 500)
        # גבול עליון של lag: תדר מינימלי 50Hz (F0 נמוך ביותר)
        max_lag   = int(self.sample_rate / 50)
        # רשימת שיאי ACF מכל הפריימים
        peaks = []
        for start in range(0, len(audio) - frame_len, hop):
            frame = audio[start:start + frame_len].astype(np.float64)
            # דילוג על פריימים שקטים
            # if float(np.sqrt(np.mean(frame ** 2))) < 0.01:
            if float(np.sqrt(np.mean(frame ** 2))) < FRAME_SILENCE_RMS:
                continue
            # הסרת DC offset (ממוצע) — מסיר רכיב DC שמזייף את ה-ACF
            frame -= np.mean(frame)
            # בדיקת אנרגיה — מניעת חלוקה באפס
            if float(np.dot(frame, frame)) < 1e-10:
                continue
            # ACF = Autocorrelation Function — מתאם עצמי של הגל
            # np.correlate mode='full' מחזיר אורך 2N-1
            acf = np.correlate(frame, frame, mode='full')
            # לוקחים רק את החצי הימני (חיובי) של ה-ACF
            acf = acf[len(acf) // 2:]
            # נרמול: מחלקים ב-acf[0] כך שהמקסימום = 1.0
            acf /= (acf[0] + 1e-10)
            # חיפוש שיא בטווח ה-lag המתאים לתדרי F0 (50-500Hz)
            search = acf[min_lag:max_lag]
            if len(search) > 0:
                # שומרים את השיא המקסימלי כמדד תקופתיות הפריים
                peaks.append(float(np.max(search)))
        # חציון כל השיאים — דובר יחיד ~0.6-0.9, ריבוי ~0.2-0.5
        # ברירת מחדל 0.3 אם לא היו פריימים (אמצעי = לא ניתן לקבוע)
        return float(np.median(peaks)) if peaks else 0.3

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        # אם האות הוא int16 (16-bit PCM) — מחלקים ב-32768 כדי לקבל float32 בטווח [-1, 1]
        if audio.dtype == np.int16:
            return audio.astype(np.float32) / 32768.0
        # אחרת — רק המרה ל-float32 
        return audio.astype(np.float32)

    def calibrate(self, audio: np.ndarray, window_samples: int) -> float:
       
        # נרמול האות לטווח [-1, 1]
        sig = self._normalize(audio)
        # רשימת ערכי harmonicity לכל החלונות
        harm_vals = []
        for i in range(0, len(sig) - window_samples, window_samples):
            # חיתוך חלון
            chunk   = sig[i:i + window_samples]
            # חישוב RMS של החלון
            rms_val = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            
            #בודקים אם החלון חזק מספיק (לא שקט)
            if rms_val >= NOISE_RMS_FLOOR:
                # רק אם הוא עבר את הסף, מחשבים harmonicity ושומרים
                harm_vals.append(self._compute_harmonicity(chunk))
        # אם פחות מ-3 חלונות — אין מספיק נתונים לכיול, מחזירים ברירת מחדל
        if len(harm_vals) < 3:
            return 0.70
        
        threshold = float(np.percentile(harm_vals, 70)) 
        threshold = max(0.55, min(0.85, threshold))
        return round(threshold, 3)

   

    def detect(self, audio: np.ndarray, harm_threshold: float = HARM_DEFAULT_THRESHOLD) -> Tuple[str, float]:
       
        # נרמול האות לטווח [-1, 1] כדי שהחישובים יהיו עצמאיים מהעוצמה
        sig = self._normalize(audio)

        # חישוב RMS (עוצמה) של החלון הנורמלי
        rms_val = float(np.sqrt(np.mean(sig.astype(np.float64) ** 2)))
        # אם הרעש מתחת לרצפה — אין כאן דיבור, מחזירים NOISE עם ביטחון גבוה
        if rms_val < NOISE_RMS_FLOOR:
            return self.NOISE, 0.95

        # אם האות קצר מ-512 דגימות — לא מספיק לחישוב מאפיינים, נניח דובר יחיד
        if len(sig) < 512:
            return self.SINGLE, 0.60

        # חילוץ 7 מאפיינים אקוסטיים מהאות (הפונקציה מחזירה מערך numpy)
        features       = self._extract_features(sig)
        # energy_cv — שונות יחסית של האנרגיה: גבוה = שינויי עוצמה = ריבוי דוברים
        energy_cv      = features[0]
        # bandwidth_norm — רוחב פס ממוצע (מנורמל): גבוה = טווח תדרים רחב = ריבוי
        bandwidth_norm = features[1]
        # flatness_std — שונות שטחיות: גבוה = שינויים בין רעש לצליל = ריבוי
        flatness_std   = features[2]
        # zcr_std — שונות קצב חציית אפס: גבוה = קצב דיבור לא אחיד = ריבוי
        zcr_std        = features[3]

        # חישוב kurtosis — מדד "שפיצות" ההתפלגות:
        #   kurtosis נמוך (< KURTOSIS_SINGLE_VETO=3.5) = גל תקופתי אחיד = דובר יחיד
        #   kurtosis גבוה (≥ 3.5) = ערבוב כאוטי = ריבוי דוברים
        kurtosis     = self._compute_kurtosis(sig)
        # חישוב harmonicity — תקופתיות הגל: גבוה = דובר יחיד, נמוך = ריבוי / רעש
        harmonicity  = self._compute_harmonicity(sig)

        
        overlap_score = min(1.0, (
            0.35 * min(energy_cv, 0.5) +
            0.30 * zcr_std * 10        +
            0.25 * flatness_std * 5    +
            0.10 * bandwidth_norm
        ))

        # הדפסת נתוני debug לקונסול (stderr) — לא מופיע בתשובת ה-API
        print(
            f"    [OD] kurt={kurtosis:.2f}(veto<{KURTOSIS_SINGLE_VETO}) "
            f"e_cv={energy_cv:.3f} zcr_std={zcr_std:.3f} "
            f"bw={bandwidth_norm:.3f} flat_std={flatness_std:.3f} score={overlap_score:.3f}/{OVERLAP_THRESHOLD}",
            file=sys.stderr
        )
        # הדפסת harmonicity לעומת הסף שחושב בכיול
        print(f"    [OD] harm={harmonicity:.2f} (singleThresh={harm_threshold:.2f})", file=sys.stderr)

        # ===== אזור 1: harmonicity גבוה = דובר יחיד ברור =====
        # אם harmonicity עולה על הסף שחושב ב-calibrate() — הגל תקופתי מאוד
        if harmonicity > harm_threshold:
            conf = round(min(0.95, 0.60 + harmonicity * 0.35), 3)
            return self.SINGLE, conf

        # ===== אזור 2: harmonicity נמוך מאוד = ריבוי / רעש =====
        # < HARMONICITY_NOISE_VETO (0.45) = הגל כאוטי מאוד
        if harmonicity < HARMONICITY_NOISE_VETO:
            # אם גם overlap_score גבוה — ריבוי דוברים ברור
            if overlap_score > 0.40:
                return self.MULTIPLE, round(overlap_score, 3)
            # אחרת — רעש סביבתי (לא דיבור) עם ביטחון גבוה
            return self.NOISE, 0.85

        # ===== אזור 3: אמצע (HARMONICITY_NOISE_VETO–harm_threshold) =====
        # overlap_score מכריע — kurtosis מוצג ב-debug אבל לא מתערב
        # הסיבה: בעברית, ילדים עונים בקול ומייצרים harmonicity מספיקה שמשאירה kurtosis נמוך
        # גם בריבוי, ולכן kurtosis לא מהימן כ-veto כאן
        if overlap_score > OVERLAP_THRESHOLD:
            # overlap_score גבוה מהסף — ריבוי דוברים
            return self.MULTIPLE, round(overlap_score, 3)
        # overlap_score נמוך מהסף — דובר יחיד
        return self.SINGLE, round(overlap_score, 3)

    