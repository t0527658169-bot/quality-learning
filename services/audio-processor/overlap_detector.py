import numpy as np
import sys
from typing import Tuple

try:
    from config import SAMPLE_RATE, OVERLAP_THRESHOLD
except ImportError:
    SAMPLE_RATE = 16000
    OVERLAP_THRESHOLD = 0.45

from audio_utils import (
    frame_audio, compute_rms, compute_zcr, compute_spectrum,
    compute_spectral_centroid, compute_spectral_bandwidth,
    compute_spectral_flatness
)

NOISE_RMS_FLOOR = 0.005
# כרטוסיס: דובר יחיד = גל תקופתי → kurtosis גולמי < 2.5
#           ריבוי = ערבוב גאוסי → kurtosis גולמי ≈ 3.0
KURTOSIS_SINGLE_VETO = 2.9


class OverlapDetector:
    SINGLE   = "דובר_יחיד"
    MULTIPLE = "ריבוי_דוברים"
    NOISE    = "רעש"

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        print("  [V] Speaker detector: weighted-sum (no ML)", file=sys.stderr)

    def _compute_kurtosis(self, audio: np.ndarray) -> float:
        """
        Kurtosis גולמי (m4/m2^2) — חציון על פריימי דיבור של 25ms.
        גל סינוס טהור → 1.5  |  גאוסי טהור → 3.0
        דובר יחיד (תקופתי) → ~1.5-2.3
        ריבוי דוברים (מעורב) → ~2.5-4.0
        """
        frame_len = int(0.025 * self.sample_rate)  # 400 @ 16kHz
        hop       = int(0.010 * self.sample_rate)  # 160 @ 16kHz
        kvals = []
        for start in range(0, len(audio) - frame_len, hop):
            frame = audio[start:start + frame_len].astype(np.float64)
            if float(np.sqrt(np.mean(frame ** 2))) < 0.01:
                continue
            m2 = float(np.mean(frame ** 2))
            m4 = float(np.mean(frame ** 4))
            if m2 < 1e-14:
                continue
            kvals.append(m4 / (m2 ** 2))
        return float(np.median(kvals)) if kvals else 3.0
    
    def _compute_harmonicity(self, audio: np.ndarray) -> float:
      """
      שיא ACF — מדד תקופתיות.
      דובר יחיד (קול קבוע): ~0.6-0.9
      ריבוי דוברים (שני F0): ~0.2-0.5
      """
      frame_len = int(0.025 * self.sample_rate)
      hop = int(0.010 * self.sample_rate)
      min_lag = int(self.sample_rate / 500)   # 500Hz
      max_lag = int(self.sample_rate / 50)    # 50Hz
      peaks = []
      for start in range(0, len(audio) - frame_len, hop):
        frame = audio[start:start + frame_len].astype(np.float64)
        if float(np.sqrt(np.mean(frame ** 2))) < 0.01:
            continue
        frame -= np.mean(frame)
        norm = float(np.dot(frame, frame))
        if norm < 1e-10:
            continue
        acf = np.correlate(frame, frame, mode='full')
        acf = acf[len(acf)//2:]
        acf /= (acf[0] + 1e-10)
        search = acf[min_lag:max_lag]
        if len(search) > 0:
            peaks.append(float(np.max(search)))
      return float(np.median(peaks)) if peaks else 0.3

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        if audio.dtype == np.int16:
            return audio.astype(np.float32) / 32768.0
        return audio.astype(np.float32)

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        frames = frame_audio(audio, 512, 256)
        rms = compute_rms(frames)
        zcr = compute_zcr(frames)
        spectrum, freqs = compute_spectrum(frames, self.sample_rate)
        centroid = compute_spectral_centroid(spectrum, freqs)
        bandwidth = compute_spectral_bandwidth(spectrum, freqs, centroid)
        flatness = compute_spectral_flatness(spectrum)

        spectral_diff = np.diff(spectrum, axis=0)
        spectral_flux = float(np.mean(np.sqrt(np.mean(spectral_diff ** 2, axis=1))))

        return np.array([
            float(np.std(rms) / (np.mean(rms) + 1e-10)),           # energy_cv
            float(np.mean(bandwidth) / 4000),                        # bandwidth_norm
            float(np.std(flatness)),                                  # flatness_std
            float(np.std(zcr)),                                       # zcr_std
            float(np.std(centroid) / (np.mean(centroid) + 1e-10)),   # centroid_cv
            float(np.max(rms) - np.min(rms)),                         # rms_range
            spectral_flux                                              # spectral_flux
        ])

    # ===========================================================
    # כיול per-recording (נוסף כעת) — ללא ML
    # מחשב את סף ה-harm באופן יחסי להקלטה עצמה:
    # לוקח את ה-percentile 75 של ערכי harm מכל חלוני הדיבור.
    # כך הקלטה עם אקוסטיקה שונה (מיקרופון רחוק, הד) מקבלת
    # סף מותאם במקום סף קבוע 0.70.
    # ===========================================================
    def calibrate(self, audio: np.ndarray, window_samples: int) -> float:
        sig = self._normalize(audio)
        harm_vals = []
        for i in range(0, len(sig) - window_samples, window_samples):
            chunk = sig[i:i + window_samples]
            rms_val = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            if rms_val < NOISE_RMS_FLOOR:
                continue
            harm_vals.append(self._compute_harmonicity(chunk))
        if len(harm_vals) < 3:
            return 0.70  # ברירת מחדל אם אין מספיק חלונות
        threshold = float(np.percentile(harm_vals, 60))
        # גבול עליון ותחתון — מניעת ערכים קיצוניים
        threshold = max(0.50, min(0.85, threshold))
        return round(threshold, 3)

    def detect(self, audio: np.ndarray, harm_threshold: float = 0.70) -> Tuple[str, float]:
        sig = self._normalize(audio)

        # בדיקת רעש
        rms_val = float(np.sqrt(np.mean(sig.astype(np.float64) ** 2)))
        if rms_val < NOISE_RMS_FLOOR:
            return self.NOISE, 0.95

        if len(sig) < 512:
            return self.SINGLE, 0.60

        f = self._extract_features(sig)
        energy_cv      = f[0]
        bandwidth_norm = f[1]
        flatness_std   = f[2]
        zcr_std        = f[3]  # נוסף חזרה לניקוד — ZCR יציב יותר מ-bandwidth באקוסטיקות שונות

        kurtosis = self._compute_kurtosis(sig)

        # ===== ניקוד משוקלל (עודכן: ZCR במקום bandwidth) =====
        # שונות אנרגיה (35%) + ZCR std (30%) + שונות שטחיות (25%) + רוחב פס (10%)
        # bandwidth_norm הורד כי מושפע מאקוסטיקת חדר; zcr_std יציב יותר
        overlap_score = min(1.0, (
            0.35 * energy_cv +
            0.30 * zcr_std * 20 +
            0.25 * flatness_std * 10 +
            0.10 * bandwidth_norm
        ))

        print(
            f"    [OD] kurt={kurtosis:.2f} e_cv={energy_cv:.3f} zcr_std={zcr_std:.3f} "
            f"bw={bandwidth_norm:.3f} flat_std={flatness_std:.3f} score={overlap_score:.3f}/{OVERLAP_THRESHOLD}",
            file=sys.stderr
        )

        # Kurtosis veto: אם הגל תקופתי (דובר יחיד) — גובר על הניקוד המשוקלל
        # if kurtosis < KURTOSIS_SINGLE_VETO:
        #     conf = round(min(0.95, 0.60 + (KURTOSIS_SINGLE_VETO - kurtosis) * 0.15), 3)
        #     return self.SINGLE, conf
        harmonicity = self._compute_harmonicity(sig)
        print(f"    [OD] harm={harmonicity:.2f} (singleThresh={harm_threshold:.2f})", file=sys.stderr)
        if harmonicity < 0.45 and kurtosis < KURTOSIS_SINGLE_VETO:
           return self.NOISE, 0.90

        # veto רק אם גם kurtosis נמוך וגם הגל תקופתי (harmonicity גבוה)
        # harm_threshold מגיע מכיול per-recording (ולא קבוע 0.70)
        if kurtosis < KURTOSIS_SINGLE_VETO and harmonicity > harm_threshold:
            conf = round(min(0.95, 0.60 + harmonicity * 0.35), 3)
            return self.SINGLE, conf

        if overlap_score > OVERLAP_THRESHOLD:
            return self.MULTIPLE, round(overlap_score, 3)
        return self.SINGLE, round(overlap_score, 3)
