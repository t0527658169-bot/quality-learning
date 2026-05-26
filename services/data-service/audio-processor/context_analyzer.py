# ===================================================
# context_analyzer.py — DISABLED (data-service — גרסה ישנה)
# *** קובץ זה שייך לגרסה הישנה לפני HEBERT ***
# *** ניתן למחוק את תיקיית data-service/audio-processor כולה ***
# ===================================================

# ===================================================
# context_analyzer.py — ניתוח הקשרי: ASR + סיווג פדגוגי (מבוסס מודל)
# ===================================================
# שילוב של 2 אלגוריתמים:
#
# 1. ASR (Automatic Speech Recognition) — המרת דיבור לטקסט
#    משתמש ב-Google Speech API לתמלול אוטומטי (תומך עברית).
#
# 2. מסווג הקשר פדגוגי — מודל Logistic Regression + TF-IDF מאומן
#    מסווג את הטקסט ל-3 קטגוריות:
#      - "הפרעה"       — רעש, בלגן ללא הקשר לימודי
#      - "למידה_פעילה" — עבודת קבוצות, תרגול משותף
#      - "פתיחה_לדיון" — המורה פותח/ת דיון מכוון
#
# אם המודל לא נמצא — נופל חזרה לשיטת מילות מפתח (rule-based).
# כדי לאמן: python train_context_model.py

import numpy as np
import tempfile
import os
import pickle
from typing import Tuple, Optional, Dict

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

try:
    import soundfile as sf
    SF_AVAILABLE = True
except ImportError:
    SF_AVAILABLE = False

from config import SAMPLE_RATE, PEDAGOGICAL_KEYWORDS, OPEN_QUESTION_KEYWORDS, SILENCING_KEYWORDS

# נתיב למודל המאומן
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "context_lr_model.pkl")
# נתיב למודל RF שאומן מתיוגי המשתמש (למידה מקוונת)
RF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "context_rf_model.pkl")
# נתיב למודל trigger — לומד מהחלון שלפני רצף ריבוי מה הכוונה
TRIGGER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "trigger_rf_model.pkl")


class ContextAnalyzer:
    """מנתח הקשרי: ממיר דיבור לטקסט ומסווג את המשמעות הפדגוגית."""

    # קטגוריות סיווג — 3 מחלקות (לא רק 2 כמו קודם)
    DISTURBANCE = "הפרעה"            # רעש, חוסר שקט — ריבוי דוברים שלילי
    ACTIVE_LEARNING = "למידה_פעילה"  # עבודת קבוצות, תרגול — ריבוי דוברים תקין
    DISCUSSION = "פתיחה_לדיון"       # המורה פתח/ה דיון מכוון — ריבוי דוברים חיובי
    UNKNOWN = "לא_ידוע"              # לא ניתן לקבוע

    # מיפוי מספרי הקטגוריות לשמות (מתאים לסדר באימון)
    LABEL_MAP = {0: "הפרעה", 1: "למידה_פעילה", 2: "פתיחה_לדיון"}

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.model = None
        self.vectorizer = None
        self.rf_model = None  # מודל RF שאומן מתיוגי המשתמש
        self.trigger_model = None  # מודל trigger — חלון קודם → חיזוי הקשר ריבוי
        # אתחול מנוע הזיהוי (רק אם הספרייה מותקנת)
        if SR_AVAILABLE:
            self.recognizer = sr.Recognizer()
        # ניסיון לטעון מודל מאומן
        self._load_model()

    def _load_model(self):
        """טעינת מודל Logistic Regression + TF-IDF מקובץ .pkl."""
        # מודל RF מאומן ע"י המשתמש (עדיפות ראשונה)
        if os.path.exists(RF_MODEL_PATH):
            with open(RF_MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
                self.rf_model = data['model']
                print(f"  [V] מודל הקשר (RF, אומן ע\"י משתמש) נטען (דיוק: {data['accuracy']:.1%})")
        # מודל trigger — חלון קודם → חיזוי הקשר
        if os.path.exists(TRIGGER_MODEL_PATH):
            with open(TRIGGER_MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
                self.trigger_model = data['model']
                print(f"  [V] מודל trigger נטען (דיוק: {data['accuracy']:.1%})")
        # מודל LR מקורי (גיבוי)
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.vectorizer = data['vectorizer']
                if self.rf_model is None:
                    print(f"  [V] מודל סיווג הקשרי נטען (דיוק: {data['accuracy']:.1%})")

    # ===============================================
    # שלב 1: המרת דיבור לטקסט (ASR)
    # ===============================================
    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """
        המרת קטע אודיו לטקסט באמצעות Google Speech Recognition.
        דורש חיבור אינטרנט.
        מחזיר את הטקסט, או None אם ההמרה נכשלה.
        """
        # בדיקה שהספריות הנדרשות מותקנות
        if not SR_AVAILABLE or not SF_AVAILABLE:
            return None

        temp_path = None
        try:
            # שמירת האודיו לקובץ WAV זמני (נדרש עבור ספריית הזיהוי)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                sf.write(temp_path, audio, self.sample_rate)

            # טעינת הקובץ לספריית הזיהוי
            with sr.AudioFile(temp_path) as source:
                audio_data = self.recognizer.record(source)

            # ניסיון זיהוי בעברית
            text = self.recognizer.recognize_google(audio_data, language="he-IL")
            return text

        except (sr.UnknownValueError, sr.RequestError):
            # UnknownValueError = לא הצליח לזהות דיבור
            # RequestError = בעיית חיבור לשרת
            return None
        except Exception:
            return None
        finally:
            # ניקוי הקובץ הזמני
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    # ===============================================
    # שלב 2: סיווג הקשר פדגוגי — מודל או fallback
    # ===============================================
    def classify_context(self, text: Optional[str]) -> Tuple[str, float]:
        """
        סיווג הטקסט המתומלל ל-3 קטגוריות פדגוגיות.

        אם יש מודל מאומן — משתמש ב-Logistic Regression + TF-IDF:
          1. TF-IDF ממיר את הטקסט לוקטור מספרי
          2. Logistic Regression מחשב הסתברות לכל קטגוריה
          3. הקטגוריה עם ההסתברות הגבוהה ביותר נבחרת

        אם אין מודל — חוזר לשיטת מילות מפתח (rule-based).

        מחזיר: (קטגוריה, ציון_ביטחון)
        """
        # אם אין טקסט — לא ניתן לקבוע
        if not text:
            return self.UNKNOWN, 0.0

        # === ניסיון עם מודל מאומן ===
        if self.model is not None and self.vectorizer is not None:
            # המרת הטקסט לוקטור TF-IDF
            X = self.vectorizer.transform([text])
            # חיזוי עם הסתברויות
            proba = self.model.predict_proba(X)[0]
            predicted_label = int(np.argmax(proba))
            confidence = float(proba[predicted_label])
            category = self.LABEL_MAP.get(predicted_label, self.UNKNOWN)
            return category, confidence

        # === fallback: שיטת מילות מפתח ===
        return self._keyword_fallback(text)

    def _keyword_fallback(self, text: str) -> Tuple[str, float]:
        """
        שיטת גיבוי — חיפוש מילות מפתח פדגוגיות בטקסט.
        משמש כשאין מודל מאומן זמין.
        """
        text_lower = text.lower()
        found = [kw for kw in PEDAGOGICAL_KEYWORDS if kw in text_lower]
        count = len(found)

        # מילות מפתח ספציפיות לפתיחת דיון
        discussion_words = ["דיון", "נדון", "מה דעתכם", "מי יכול", "חשבו"]
        discussion_found = [w for w in discussion_words if w in text_lower]

        if len(discussion_found) >= 1 and count >= 2:
            return self.DISCUSSION, min(1.0, 0.5 + count * 0.15)
        elif count >= 2:
            confidence = min(1.0, 0.5 + count * 0.15)
            return self.ACTIVE_LEARNING, confidence
        elif count == 1:
            return self.ACTIVE_LEARNING, 0.55
        else:
            return self.DISTURBANCE, 0.6

    # ===============================================
    # ממשק ציבורי: ניתוח מלא (ASR + סיווג)
    # ===============================================
    def analyze(self, audio: np.ndarray) -> Dict:
        """
        ניתוח מלא: המרה לטקסט + סיווג הקשרי.
        אם יש מודל RF מאומן — משתמש בו (עדיפות על ASR).
        מחזיר מילון עם: טקסט, קטגוריה, ביטחון.
        """
        # אם יש מודל RF מתיוגי המשתמש — עדיפות ראשונה
        if self.rf_model is not None:
            return self._analyze_with_rf(audio)
        # אחרת — ASR + סיווג טקסט
        text = self.transcribe(audio)
        category, confidence = self.classify_context(text)
        return {
            'transcribed_text': text,
            'context_category': category,
            'confidence': confidence
        }

    def _analyze_with_rf(self, audio: np.ndarray) -> Dict:
        """ניתוח באמצעות מודל RF מאומן על features אקוסטיים."""
        from train_overlap_model import extract_features
        features = extract_features(audio)

        # בדיקה אם המודל תומך ב-predict_proba (DummyClassifier עם מחלקה אחת)
        try:
            proba = self.rf_model.predict_proba(features.reshape(1, -1))[0]
            predicted = int(np.argmax(proba))
            confidence = float(proba[predicted])
        except Exception:
            predicted = int(self.rf_model.predict(features.reshape(1, -1))[0])
            confidence = 1.0

        # 0 = הפרעה, 1 = לצורך השיעור (למידה/דיון)
        if predicted == 1:
            category = self.ACTIVE_LEARNING
        else:
            category = self.DISTURBANCE
        return {
            'transcribed_text': None,
            'context_category': category,
            'confidence': confidence
        }

    # ===============================================
    # זיהוי שאלה פתוחה / הנחיית מורה (מפעיל מצב "לצורך השיעור")
    # ===============================================
    def has_open_question(self, audio: np.ndarray) -> bool:
        """
        בדיקה האם בקטע אודיו יש שאלה פתוחה או הנחיית מורה.
        מתמלל את האודיו ובודק מילות מפתח.
        """
        text = self.transcribe(audio)
        if not text:
            return False
        text_lower = text.lower()
        for kw in OPEN_QUESTION_KEYWORDS:
            if kw in text_lower:
                return True
        return False

    # ===============================================
    # זיהוי מילות השתקה (מפעיל מצב "הפרעה")
    # ===============================================
    def has_silencing_words(self, audio: np.ndarray) -> bool:
        """
        בדיקה האם בקטע אודיו יש ניסיון השתקה.
        מתמלל את האודיו ובודק מילות מפתח.
        """
        text = self.transcribe(audio)
        if not text:
            return False
        text_lower = text.lower()
        for kw in SILENCING_KEYWORDS:
            if kw in text_lower:
                return True
        return False

    # ===============================================
    # חיזוי trigger — מהחלון הנוכחי, חוזה מה יקרה בריבוי הבא
    # ===============================================
    def predict_trigger(self, audio: np.ndarray) -> Optional[str]:
        """
        חיזוי מהחלון הנוכחי (דובר יחיד) האם ריבוי הדוברים שאחריו
        הוא לצורך השיעור או הפרעה.
        מחזיר 'lesson', 'disruption', או None אם אין מודל.
        """
        if self.trigger_model is None:
            return None
        try:
            from train_overlap_model import extract_features
            features = extract_features(audio)
            features = np.array(features).reshape(1, -1)
            pred = self.trigger_model.predict(features)[0]
            return 'lesson' if pred == 1 else 'disruption'
        except Exception:
            return None


# ===================================================
# DISABLED: הקוד מעל אינו פעיל
# ===================================================
raise SystemExit(f"[DISABLED] context_analyzer.py — גרסה ישנה, אינו פעיל")
