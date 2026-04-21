# ===================================================
# overlap_detector.py — זיהוי חפיפת דוברים (מבוסס מודל)
# ===================================================
# מזהה האם בחלון זמן מדבר דובר יחיד או מספר דוברים בו-זמנית.
#
# שיטה:
#   1. חילוץ 7 מאפיינים אקוסטיים מקטע האודיו
#   2. הזנת המאפיינים למודל Random Forest מאומן
#   3. המודל מחזיר: 0 = דובר יחיד, 1 = ריבוי דוברים + ציון ביטחון
#
# אם המודל לא נמצא — נופל חזרה לשיטת ספים (rule-based).
# כדי לאמן: python train_overlap_model.py

import numpy as np
import os
import pickle
from typing import Tuple
from config import SAMPLE_RATE, OVERLAP_THRESHOLD
from audio_utils import (
    frame_audio, compute_rms, compute_zcr, compute_spectrum,
    compute_spectral_centroid, compute_spectral_bandwidth,
    compute_spectral_flatness
)

# נתיב למודל המאומן
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "overlap_rf_model.pkl")


class OverlapDetector:
    """מזהה חפיפה בין דוברים — משתמש במודל Random Forest אם קיים."""

    SINGLE = "דובר_יחיד"
    MULTIPLE = "ריבוי_דוברים"
    NOISE = "רעש"

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.model = None
        # ניסיון לטעון מודל מאומן
        self._load_model()

    def _load_model(self):
        """טעינת המודל מהקובץ .pkl (אם קיים)."""
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                print(f"  [V] מודל ריבוי דוברים נטען (דיוק: {data['accuracy']:.1%})")

    # ===============================================
    # חילוץ מאפיינים — אותם 7 מאפיינים שהמודל אומן עליהם
    # ===============================================
    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        חילוץ 7 מאפיינים אקוסטיים (זהים לאימון):
          1. שונות אנרגיה (CV)
          2. רוחב פס ממוצע
          3. שונות שטחיות
          4. שונות ZCR
          5. שונות מרכז ספקטרלי
          6. טווח RMS
          7. שטף ספקטרלי
        """
        frames = frame_audio(audio, 512, 256)
        rms = compute_rms(frames)
        zcr = compute_zcr(frames)
        spectrum, freqs = compute_spectrum(frames, self.sample_rate)
        centroid = compute_spectral_centroid(spectrum, freqs)
        bandwidth = compute_spectral_bandwidth(spectrum, freqs, centroid)
        flatness = compute_spectral_flatness(spectrum)

        # שטף ספקטרלי
        spectral_diff = np.diff(spectrum, axis=0)
        spectral_flux = float(np.mean(np.sqrt(np.mean(spectral_diff ** 2, axis=1))))

        return np.array([
            float(np.std(rms) / (np.mean(rms) + 1e-10)),
            float(np.mean(bandwidth) / 4000),
            float(np.std(flatness)),
            float(np.std(zcr)),
            float(np.std(centroid) / (np.mean(centroid) + 1e-10)),
            float(np.max(rms) - np.min(rms)),
            spectral_flux
        ])

    # ===============================================
    # זיהוי — מודל או fallback
    # ===============================================
    def detect(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        בדיקה האם יש ריבוי דוברים בקטע.

        אם יש מודל מאומן — משתמש בו (Random Forest).
        אם אין — חוזר לשיטת ספים פשוטה (rule-based).

        מחזיר: (סוג_דובר, ציון_חפיפה)
        """
        features = self._extract_features(audio)

        if self.model is not None:
            # === שימוש במודל ===
            # predict_proba מחזיר [P(יחיד), P(ריבוי)] או [P(יחיד), P(ריבוי), P(רעש)]
            proba = self.model.predict_proba(features.reshape(1, -1))[0]
            classes = list(self.model.classes_)
            if len(classes) == 3 and 2 in classes:
                # מודל 3 מחלקות: 0=יחיד, 1=ריבוי, 2=רעש
                best = int(np.argmax(proba))
                score = float(proba[best])
                if classes[best] == 2:
                    return self.NOISE, score
                elif classes[best] == 1:
                    return self.MULTIPLE, score
                return self.SINGLE, score
            else:
                # מודל 2 מחלקות (תאימות אחורה)
                overlap_score = float(proba[1])  # הסתברות לריבוי דוברים
                if overlap_score > 0.5:
                    return self.MULTIPLE, overlap_score
                return self.SINGLE, overlap_score
        else:
            # === fallback: שיטת ספים (ללא מודל) ===
            overlap_score = min(1.0, (
                0.4 * features[0] +    # שונות אנרגיה
                0.3 * features[1] +    # רוחב פס
                0.3 * features[2] * 10  # שונות שטחיות
            ))
            if overlap_score > OVERLAP_THRESHOLD:
                return self.MULTIPLE, overlap_score
            return self.SINGLE, overlap_score
