# ===================================================
# train_overlap_model.py — אימון מודל לזיהוי ריבוי דוברים
# ===================================================
# מודל: Random Forest Classifier (יער אקראי)
#
# למה Random Forest ולא רשת נוירונים?
#   - שקוף: ניתן לראות אילו מאפיינים חשובים (feature importance)
#   - יציב: לא רגיש ל-overfitting עם מעט נתונים
#   - מהיר: אימון ב-שניות, לא שעות
#   - מובן: כל "עץ" מקבל החלטה פשוטה מבוססת ספים
#
# שלבי האימון:
#   1. יצירת דאטה סינתטי — דגימות של "דובר יחיד" ו"ריבוי דוברים"
#   2. חילוץ 7 מאפיינים אקוסטיים מכל דגימה
#   3. אימון Random Forest על 80% מהנתונים
#   4. הערכה על 20% שנותרו
#   5. שמירת המודל לקובץ .pkl
#
# הרצה: python train_overlap_model.py

import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from config import SAMPLE_RATE
from audio_utils import (
    frame_audio, compute_rms, compute_zcr, compute_spectrum,
    compute_spectral_centroid, compute_spectral_bandwidth,
    compute_spectral_flatness, compute_mfcc
)

# --- נתיב שמירת המודל ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "overlap_rf_model.pkl")

# שמות המאפיינים — חשוב להבין מה כל אחד מייצג
FEATURE_NAMES = [
    "energy_variance",       # שונות באנרגיה (CV) — ריבוי דוברים = תנודות חדות
    "mean_bandwidth",        # רוחב פס ממוצע — ריבוי = פיזור תדרים רחב
    "flatness_variance",     # שונות שטחיות — ריבוי = שינויים במרקם
    "zcr_variance",          # שונות ZCR — ריבוי = שינוי תכוף בין קולות
    "centroid_variance",     # שונות מרכז ספקטרלי — ריבוי = קפיצות בתדר
    "rms_range",             # טווח RMS (max-min) — ריבוי = פערים גדולים
    "spectral_flux",         # שטף ספקטרלי — ריבוי = שינויים מהירים בספקטרום
]


# ===================================================
# שלב 1: יצירת אודיו סינתטי
# ===================================================
def _generate_single_speaker(duration_sec: float = 3.0) -> np.ndarray:
    """
    יצירת סימולציה של דובר יחיד.
    דובר יחיד = גל אחיד עם תדר יציב + רעש קל.
    """
    sr = SAMPLE_RATE
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # תדר בסיס אקראי (80-300Hz = טווח קול אנושי)
    f0 = np.random.uniform(80, 300)
    # אות דיבור — גל עם הרמוניקות (מחקה קול אנושי)
    signal = np.sin(2 * np.pi * f0 * t)
    signal += 0.3 * np.sin(2 * np.pi * 2 * f0 * t)  # הרמוניקה 2
    signal += 0.15 * np.sin(2 * np.pi * 3 * f0 * t)  # הרמוניקה 3
    # מעטפת עוצמה חלקה (דובר יחיד = עוצמה יציבה)
    envelope = 0.3 + 0.1 * np.sin(2 * np.pi * 0.5 * t)
    signal *= envelope
    # הוספת רעש קל
    signal += np.random.normal(0, 0.02, len(signal))
    # הוספת הפסקות קצרות (כמו בדיבור טבעי)
    pause_start = np.random.randint(0, len(signal) // 2)
    pause_len = int(0.2 * sr)  # 200ms הפסקה
    signal[pause_start:pause_start + pause_len] *= 0.05
    return signal.astype(np.float32)


def _generate_multiple_speakers(duration_sec: float = 3.0) -> np.ndarray:
    """
    יצירת סימולציה של ריבוי דוברים (2-4 דוברים).
    ריבוי דוברים = כמה גלים עם תדרים שונים שמתערבבים.
    """
    sr = SAMPLE_RATE
    n_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    mixed = np.zeros(n_samples, dtype=np.float32)
    n_speakers = np.random.randint(2, 5)  # 2 עד 4 דוברים

    for _ in range(n_speakers):
        # כל דובר — תדר בסיס שונה
        f0 = np.random.uniform(80, 350)
        voice = np.sin(2 * np.pi * f0 * t)
        voice += 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
        voice += 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
        # מעטפת אקראית — כל דובר מדבר בזמנים שונים
        env = np.random.uniform(0.1, 0.5)
        # הפעלה חלקית (לא כולם מדברים כל הזמן)
        start = np.random.randint(0, n_samples // 3)
        end = np.random.randint(n_samples // 2, n_samples)
        mask = np.zeros(n_samples)
        mask[start:end] = 1.0
        # החלקה עם חלון
        from scipy.ndimage import uniform_filter1d
        mask = uniform_filter1d(mask, size=int(0.1 * sr))
        voice *= mask * env
        mixed += voice

    # רעש סביבתי חזק יותר
    mixed += np.random.normal(0, 0.05, n_samples)
    # נרמול
    mixed /= np.max(np.abs(mixed)) + 1e-10
    mixed *= 0.5
    return mixed


# ===================================================
# שלב 2: חילוץ מאפיינים מקטע אודיו
# ===================================================
def extract_features(audio: np.ndarray) -> np.ndarray:
    """
    חילוץ 7 מאפיינים אקוסטיים מקטע אודיו.
    כל מאפיין מודד היבט שונה של האות:
      - מאפיינים 1-3: קשורים לתדרים (ספקטרום)
      - מאפיינים 4-5: קשורים לזמן (אנרגיה, ZCR)
      - מאפיינים 6-7: משלבים זמן + תדר
    """
    frames = frame_audio(audio, 512, 256)
    rms = compute_rms(frames)
    zcr = compute_zcr(frames)
    spectrum, freqs = compute_spectrum(frames, SAMPLE_RATE)
    centroid = compute_spectral_centroid(spectrum, freqs)
    bandwidth = compute_spectral_bandwidth(spectrum, freqs, centroid)
    flatness = compute_spectral_flatness(spectrum)

    # שטף ספקטרלי — שינוי הספקטרום בין פריימים עוקבים
    # ערך גבוה = שינוי מהיר = ריבוי קולות שנכנסים ויוצאים
    spectral_diff = np.diff(spectrum, axis=0)
    spectral_flux = float(np.mean(np.sqrt(np.mean(spectral_diff ** 2, axis=1))))

    return np.array([
        float(np.std(rms) / (np.mean(rms) + 1e-10)),    # שונות אנרגיה (CV)
        float(np.mean(bandwidth) / 4000),                 # רוחב פס מנורמל
        float(np.std(flatness)),                           # שונות שטחיות
        float(np.std(zcr)),                                # שונות ZCR
        float(np.std(centroid) / (np.mean(centroid) + 1e-10)),  # שונות מרכז ספקטרלי
        float(np.max(rms) - np.min(rms)),                  # טווח RMS
        spectral_flux                                       # שטף ספקטרלי
    ])


# ===================================================
# שלב 3: אימון המודל
# ===================================================
def train_model(n_samples: int = 1000):
    """
    יצירת דאטהסט סינתטי, חילוץ מאפיינים, ואימון Random Forest.

    n_samples: כמות הדגימות לכל מחלקה (סה"כ n_samples×2 דגימות).
    """
    print(f"\n{'='*50}")
    print(f"  אימון מודל ריבוי דוברים (Random Forest)")
    print(f"{'='*50}")

    # --- יצירת דאטהסט ---
    print(f"\n  [1/4] יצירת {n_samples} דגימות לכל מחלקה...")
    X_list = []
    y_list = []

    for i in range(n_samples):
        # מחלקה 0: דובר יחיד
        audio_single = _generate_single_speaker()
        feat = extract_features(audio_single)
        X_list.append(feat)
        y_list.append(0)

        # מחלקה 1: ריבוי דוברים
        audio_multi = _generate_multiple_speakers()
        feat = extract_features(audio_multi)
        X_list.append(feat)
        y_list.append(1)

        if (i + 1) % 200 == 0:
            print(f"    ...נוצרו {(i+1)*2} דגימות")

    X = np.array(X_list)
    y = np.array(y_list)

    # --- חלוקה לאימון ומבחן ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  [2/4] אימון: {len(X_train)} דגימות | מבחן: {len(X_test)} דגימות")

    # --- אימון המודל ---
    print(f"\n  [3/4] אימון Random Forest (100 עצים)...")
    model = RandomForestClassifier(
        n_estimators=100,      # 100 עצי החלטה
        max_depth=10,          # עומק מקסימלי = 10 (מונע overfitting)
        min_samples_leaf=5,    # מינימום 5 דגימות בעלה (שמירה על הכללה)
        random_state=42,       # seed קבוע לשחזור תוצאות
        n_jobs=-1              # שימוש בכל הליבות
    )
    model.fit(X_train, y_train)

    # --- הערכת ביצועים ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  תוצאות על סט המבחן:")
    print(f"  דיוק כולל: {acc:.1%}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['דובר_יחיד', 'ריבוי_דוברים'])}")

    # --- חשיבות מאפיינים (מה המודל למד) ---
    print(f"  חשיבות מאפיינים (מה הכי משפיע על ההחלטה):")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx:
        bar = "█" * int(importances[idx] * 40)
        print(f"    {FEATURE_NAMES[idx]:<25s} {importances[idx]:.3f} {bar}")

    # --- שמירה ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': FEATURE_NAMES,
            'accuracy': acc,
            'description': 'Random Forest classifier for speaker overlap detection'
        }, f)
    print(f"\n  [4/4] מודל נשמר: {MODEL_PATH}")
    print(f"{'='*50}\n")

    return model


if __name__ == "__main__":
    train_model(n_samples=500)


