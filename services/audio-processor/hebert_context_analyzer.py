# ===================================================
# hebert_context_analyzer.py — ניתוח הקשרי עם HEBERT
# ===================================================
# מערכת ניתוח הקשרי מבוססת HEBERT (Hebrew BERT) לזיהוי כוונות פדגוגיות.
#
# רכיבים:
#   1. WhisperASR - המרת דיבור לטקסט (Whisper OpenAI)
#   2. HEBERTModel - מודל BERT עברי (ארכיטקטורה מלאה)
#   3. TriggerClassifier - זיהוי שאלה פתוחה / ניסיון השתקה
#   4. ContextClassifier - סיווג: לצורך השיעור / הפרעה
#   5. SequenceTracker - מעקב אחר רצפי ריבוי דוברים
#
# למה לא Black Box?
#   - כל הארכיטקטורה כתובה בעצמנו ומוסברת
#   - רק המשקולות (110M מספרים) נטענות מאימון
#   - כל שלב ניתן להדפסה ובדיקה

import numpy as np
import torch
import tempfile
import os
from typing import Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModel
import whisper

from config import SAMPLE_RATE, OPEN_QUESTION_KEYWORDS, SILENCING_KEYWORDS


# ===================================================
# 1. WhisperASR - המרת דיבור לטקסט
# ===================================================
class WhisperASR:
    """
    המרת דיבור לטקסט באמצעות Whisper (OpenAI).
    
    למה Whisper?
    - תמיכה מצוינת בעברית
    - דיוק גבוה גם עם רעשי רקע
    - מודל קוד פתוח
    
    זה Black Box? כן, אבל זה בסדר - ASR הוא כלי עזר, לא חלק מהמטלה.
    """
    
    def __init__(self, model_size: str = "base"):
        """
        אתחול מודל Whisper.
        
        Args:
            model_size: גודל המודל - "tiny", "base", "small", "medium", "large"
                       base = איזון טוב בין מהירות ודיוק
        """
        print(f"  [Whisper] טוען מודל {model_size}...")
        self.model = whisper.load_model(model_size)
        print(f"  [V] Whisper מוכן")
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Optional[str]:
        """
        המרת אודיו לטקסט.
        
        Args:
            audio: מערך numpy של דגימות אודיו
            sample_rate: קצב דגימה (16000 Hz)
        
        Returns:
            טקסט מתומלל או None אם נכשל
        """
        try:
            # Whisper מצפה לאודיו float32 מנורמל ל-[-1, 1]
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # נרמול אם צריך
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()
            
            # תמלול עם Whisper
            result = self.model.transcribe(
                audio,
                language="he",  # עברית
                fp16=False,     # CPU mode
                verbose=False
            )
            
            text = result.get("text", "").strip()
            return text if text else None
            
        except Exception as e:
            print(f"  [!] Whisper error: {e}")
            return None


# ===================================================
# 2. HEBERTModel - ארכיטקטורת BERT מלאה
# ===================================================
class HEBERTModel:
    """
    מודל HEBERT (Hebrew BERT) - ארכיטקטורה מלאה עם הסברים.
    
    מבנה BERT:
    1. Tokenizer - פירוק טקסט לטוקנים (מילים/תווים)
    2. Embedding Layer - המרת טוקנים לוקטורים (768 מימדים)
    3. 12 Transformer Blocks - עיבוד הקשרי עמוק
    4. Pooler - סיכום הפלט לוקטור אחד
    
    למה לא Black Box?
    - אנחנו יודעים בדיוק מה כל שכבה עושה
    - רק המשקולות (המספרים) באות מאימון
    - אפשר להדפיס ביניים ולראות מה קורה
    """
    
    def __init__(self, model_name: str = "avichr/heBERT"):
        """
        אתחול HEBERT.
        
        Args:
            model_name: שם המודל ב-HuggingFace (avichr/heBERT = BERT עברי מאומן)
        """
        print(f"  [HEBERT] טוען מודל {model_name}...")
        
        # טעינת Tokenizer - פירוק טקסט לטוקנים
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # טעינת המודל עם המשקולות המאומנות
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # מצב הערכה (לא אימון)
        
        # העברה ל-CPU (אם יש GPU אפשר לשנות ל-cuda)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        
        print(f"  [V] HEBERT מוכן (12 שכבות, 768 מימדים)")
    
    def encode(self, text: str) -> np.ndarray:
        """
        המרת טקסט לוקטור embedding.
        
        תהליך:
        1. Tokenization - פירוק לטוקנים: "בואו נדון" → [101, 1234, 5678, 102]
        2. Embedding - המרה לוקטורים: [101] → [0.2, -0.5, 0.1, ...]
        3. Transformer - 12 שכבות עיבוד הקשרי
        4. Pooling - סיכום לוקטור אחד (768 מימדים)
        
        Args:
            text: טקסט בעברית
        
        Returns:
            וקטור numpy בגודל (768,) - ייצוג סמנטי של הטקסט
        """
        if not text or not text.strip():
            return np.zeros(768, dtype=np.float32)
        
        # שלב 1: Tokenization
        inputs = self.tokenizer(
            text,
            return_tensors="pt",  # PyTorch tensors
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # שלב 2-4: BERT processing
        with torch.no_grad():  # לא צריך gradients (רק inference)
            outputs = self.model(**inputs)
            # outputs.last_hidden_state = [batch, seq_len, 768]
            # outputs.pooler_output = [batch, 768] - סיכום של כל המשפט
            embedding = outputs.pooler_output[0]  # לוקח את הדגימה הראשונה
        
        # המרה ל-numpy
        return embedding.cpu().numpy()


# ===================================================
# 3. TriggerClassifier - זיהוי שאלה פתוחה / השתקה
# ===================================================
class TriggerClassifier:
    """
    מסווג שמזהה מה קרה בחלון דובר יחיד (המורה).
    
    3 אפשרויות:
    - "שאלה_פתוחה"  → הריבוי הבא = לצורך השיעור
    - "ניסיון_השתקה" → הריבוי הבא = הפרעה
    - "רגיל"          → ממשיכים לבדוק
    
    איך זה עובד (Prototype Cosine Similarity):
    1. HEBERT ממיר את הטקסט הנכנס לוקטור [768]
    2. HEBERT ממיר מראש משפטי דוגמה לכל קטגוריה לוקטורים
    3. חישוב cosine similarity בין הטקסט לכל prototype
    4. הקטגוריה הקרובה ביותר מבחינה סמנטית — מנצחת
    
    למה זה עובד? כי HeBERT מבין שמשמעות:
    "מה דעתכם?" ≈ "מי רוצה לענות?" ≈ "תשתפו אותנו"
    הם כולם קרובים ב-embedding space
    """

    # משפטי prototype לכל קטגוריה — מגדירים את "המרכז" של כל קטגוריה.
    # ככל שיש יותר דוגמאות מגוונות, ממוצע ה-embedding מייצג טוב יותר את הקטגוריה.

    # --- שאלות פתוחות וציוויים שפותחים דיון ---
    QUESTION_PROTOTYPES = [
        # שאלות חוות דעת
        "מה דעתכם על הנושא?",
        "מה אתם חושבים על זה?",
        "מי רוצה לשתף את דעתו?",
        "מה אתם אומרים על כך?",
        "איך אתם רואים את זה?",
        # שאלות ידע ובנה
        "מי יודע את התשובה?",
        "מי יכול להסביר?",
        "מי זוכר מה למדנו?",
        "מי יכול לתת דוגמה?",
        # הזמנה לדיון
        "בואו נדון בזה יחד",
        "בואו נחשוב על זה",
        "בואו נשמע רעיונות",
        "נדון רגע בנושא הזה",
        # ציוויים לעבודה קבוצתית
        "תעבדו בקבוצות על המשימה",
        "דברו ביניכם ותציגו",
        "עבדו בזוגות ותמצאו פתרון",
        "שוחחו עם השותף שלכם",
        "חשבו יחד ותגיבו",
        # ציוויים לשיתוף
        "ספרו לי מה אתם יודעים",
        "שתפו את כולם",
        "העלו רעיונות",
        "תגיבו על מה שנאמר",
        # שאלות שיחתיות
        "איך היה לכם?",
        "מה קרה בבית?",
        "מה חשבתם על המשימה?",
        # --- הזמנות עקיפות לדיון: "אני רוצה שנדבר על..." ---
        "אני רוצה שנדבר על מה שקרה",
        "אני רוצה שנדבר על זה ביחד",
        "אני רוצה שנדון בנושא הזה",
        "אני רוצה לשמוע את דעתכם",
        "אני רוצה לשמוע מה אתם חושבים",
        "אני רוצה שתשתפו אותי",
        "הייתי רוצה שנדבר על זה",
        "הייתי רוצה לשמוע מכם",
        "הייתי שמח לשמוע את דעתכם",
        # --- הזמנה לספר / לשתף אירוע ---
        "ספרו לי מה קרה",
        "ספרו לי מה קרה אתמול",
        "ספרו לי על מה שחוויתם",
        "ספרו לי מה הרגשתם",
        "ספרו לי מה עלה לכם בראש",
        "מישהו יכול לספר מה קרה?",
        "מה קרה אתמול? ספרו לי",
        # --- שאלות רפלקטיביות על אירועים ---
        "מה קרה אתמול?",
        "מה קרה לאחרונה?",
        "מה היה שם?",
        "מה חוויתם?",
        "מה הרגשתם כשזה קרה?",
        "ספרו לי על החוויה שלכם",
        "מה עלה לכם בראש כשראיתם?",
        # --- "בואו נדבר על..." ---
        "בואו נדבר על מה שקרה",
        "בואו נדבר על זה",
        "בואו נדבר על הנושא הזה",
        "בואו נשוחח על מה שקרה אתמול",
        "בואו נשמע מה כולם חושבים",
        "בואו נשתף רגשות ומחשבות",
        # --- ביטויים שמעידים על פתיחת דיון ---
        "זה קשור למה שלמדנו",
        "אולי זה קשור למה שדיברנו עליו",
        "יש לזה קשר למה שקרה",
        "מעניין לי לדעת מה אתם חושבים",
        "סקרני לשמוע את דעתכם",
        "אשמח אם תשתפו",
    ]

    # --- ניסיונות השתקה והחזרת שקט ---
    SILENCING_PROTOTYPES = [
        # פקודות ישירות
        "שקט בכיתה!",
        "שקט עכשיו!",
        "שקטים בבקשה",
        "תשתקו",
        "שתקו",
        "הפסיקו לדבר",
        "תפסיקו לדבר עכשיו",
        "עצרו הכל",
        "הס!",
        # ביטויי תסכול
        "אני לא שומעת כלום",
        "אני לא שומע אתכם",
        "מספיק רעש",
        "יש פה יותר מדי רעש",
        "אי אפשר לדבר ככה",
        "די לרעש",
        # בקשות מנומסות
        "אני צריכה שקט",
        "תנו לי רגע של שקט",
        "בבקשה תהיו בשקט",
        "אפשר שקט רגע?",
        # אנגלית
        "quiet please",
        "everyone stop talking",
        "silence now",
    ]

    # --- דיבור ניטרלי — הסבר, כתיבה, מעבר בין נושאים ---
    NEUTRAL_PROTOTYPES = [
        "פתחו את הספרים בעמוד עשרים",
        "כתבו את התאריך במחברת",
        "נמשיך עם החומר",
        "תסתכלו על הלוח",
        "היום נלמד על הנושא הבא",
        "אני אסביר את הנוסחה",
        "שימו לב להגדרה הזו",
        "נקרא את הפסקה ביחד",
        "תעתיקו את מה שכתוב",
        "הגישו את המחברות",
    ]

    def __init__(self, hebert_model: HEBERTModel):
        self.hebert = hebert_model

        print("  [TriggerClassifier] מחשב prototype embeddings...")
        # חישוב prototype embeddings מראש (פעם אחת בלבד)
        self._question_proto = self._mean_embedding(self.QUESTION_PROTOTYPES)
        self._silencing_proto = self._mean_embedding(self.SILENCING_PROTOTYPES)
        self._neutral_proto = self._mean_embedding(self.NEUTRAL_PROTOTYPES)
        print("  [V] Prototypes מוכנים")

    def _mean_embedding(self, sentences):
        """ממוצע embeddings של רשימת משפטים — מייצג את מרכז הקטגוריה."""
        vecs = np.array([self.hebert.encode(s) for s in sentences])
        mean = vecs.mean(axis=0)
        return mean / (np.linalg.norm(mean) + 1e-9)  # נרמול ל-unit vector

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """cosine similarity בין שני וקטורים מנורמלים."""
        return float(np.dot(a, b))

    def classify(self, text: Optional[str]) -> Tuple[str, float]:
        """
        סיווג טקסט לקטגוריית trigger לפי קרבה סמנטית ל-prototypes.

        Args:
            text: טקסט מתומלל מחלון דובר יחיד

        Returns:
            (קטגוריה, ציון_ביטחון)
        """
        if not text or not text.strip():
            return "רגיל", 0.5

        # HEBERT ממיר את הטקסט לוקטור
        emb = self.hebert.encode(text)
        norm = np.linalg.norm(emb)
        if norm < 1e-9:
            return "רגיל", 0.5
        emb = emb / norm

        # חישוב cosine similarity לכל קטגוריה
        sim_question  = self._cosine(emb, self._question_proto)
        sim_silencing = self._cosine(emb, self._silencing_proto)
        sim_neutral   = self._cosine(emb, self._neutral_proto)

        scores = {
            "שאלה_פתוחה":    sim_question,
            "ניסיון_השתקה":  sim_silencing,
            "רגיל":           sim_neutral,
        }

        best_label = max(scores, key=scores.__getitem__)
        best_score = float(scores[best_label])

        # נרמול לביטחון 0-1 (softmax על הציונים)
        vals = np.array(list(scores.values()))
        softmax_vals = np.exp(vals * 5) / np.exp(vals * 5).sum()  # temperature=0.2
        confidence = float(softmax_vals[list(scores.keys()).index(best_label)])

        return best_label, confidence


# ===================================================
# 4. ContextClassifier - לצורך השיעור / הפרעה
# ===================================================
class ContextClassifier:
    """
    מסווג שקובע האם ריבוי דוברים הוא לצורך השיעור או הפרעה.

    2 אפשרויות:
    - "לצורך_השיעור" → דיון, עבודת קבוצות, תרגול
    - "הפרעה"         → רעש, בלגן

    איך זה עובד (Prototype Cosine Similarity):
    1. HEBERT ממיר את הטקסט לוקטור [768]
    2. משווה ל-prototype embedding של כל קטגוריה
    3. הקרוב ביותר מבחינה סמנטית — מנצח
    """

    LESSON_PROTOTYPES = [
        # דיון מובנה
        "אנחנו עובדים על המשימה יחד",
        "בואו נפתור את התרגיל",
        "נדון על הנושא הזה",
        "תסבירו לי מה הבנתם",
        "אנחנו מתרגלים את החומר",
        "שואלים שאלות על השיעור",
        # עבודת קבוצות
        "עבודת קבוצות בכיתה",
        "אנחנו עובדים בזוגות",
        "כל קבוצה מציגה את עבודתה",
        "מתחלקים לקבוצות ועובדים",
        # תרגול ושיתוף
        "כולם מתרגלים את החומר",
        "משתפים רעיונות עם הכיתה",
        "כל אחד מסביר לשותף שלו",
        "עונים על השאלות יחד",
        # דיון ענייני
        "מדברים על הנושא שלמדנו",
        "דנים בפתרון הבעיה",
        "כולם מביעים את דעתם",
        # שיתוף חוויות ורפלקציה — תגובה לפתיחת דיון של המורה
        "כולם מדברים ומשתפים מה קרה",
        "ילדים מספרים מה קרה אתמול",
        "תלמידים מדברים על החוויה שלהם",
        "כולם מגיבים לשאלה של המורה",
        "תלמידים משיבים על שאלה פתוחה",
        "דיון כיתתי על חוויות ואירועים",
        "תלמידים מביעים דעות ומחשבות",
        "כולם עונים ומשתפים ביחד",
    ]
    DISRUPTION_PROTOTYPES = [
        # רעש לא ענייני
        "בלגן ורעש בכיתה",
        "כולם מדברים בו זמנית על שטויות",
        "אי אפשר להבין כלום",
        "ילדים צוחקים ומשוחחים",
        "רעש שלא קשור לשיעור",
        "הפרעה בלמידה",
        # שיחות צד
        "מדברים על דברים שלא קשורים",
        "שיחות פרטיות בכיתה",
        "מתעסקים בטלפון ומדברים",
        "צוחקים ומשחקים בשיעור",
        # אוירה כאוטית
        "בלגן כללי בכיתה",
        "אי אפשר ללמוד בגלל הרעש",
        "ילדים לא מקשיבים",
        "הכיתה יצאה מכלל שליטה",
    ]

    def __init__(self, hebert_model: HEBERTModel):
        self.hebert = hebert_model

        print("  [ContextClassifier] מחשב prototype embeddings...")
        self._lesson_proto    = self._mean_embedding(self.LESSON_PROTOTYPES)
        self._disruption_proto = self._mean_embedding(self.DISRUPTION_PROTOTYPES)
        print("  [V] Prototypes מוכנים")

    def _mean_embedding(self, sentences):
        vecs = np.array([self.hebert.encode(s) for s in sentences])
        mean = vecs.mean(axis=0)
        return mean / (np.linalg.norm(mean) + 1e-9)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def classify(self, text: Optional[str]) -> Tuple[str, float]:
        """
        סיווג טקסט לפי קרבה סמנטית ל-prototypes.

        Args:
            text: טקסט מתומלל מחלון ריבוי דוברים

        Returns:
            (קטגוריה, ציון_ביטחון)
        """
        if not text or not text.strip():
            return "הפרעה", 0.6

        emb = self.hebert.encode(text)
        norm = np.linalg.norm(emb)
        if norm < 1e-9:
            return "הפרעה", 0.6
        emb = emb / norm

        sim_lesson    = self._cosine(emb, self._lesson_proto)
        sim_disruption = self._cosine(emb, self._disruption_proto)

        if sim_lesson > sim_disruption:
            vals = np.array([sim_lesson, sim_disruption])
            softmax_vals = np.exp(vals * 5) / np.exp(vals * 5).sum()
            return "לצורך_השיעור", float(softmax_vals[0])
        else:
            vals = np.array([sim_disruption, sim_lesson])
            softmax_vals = np.exp(vals * 5) / np.exp(vals * 5).sum()
            return "הפרעה", float(softmax_vals[0])


# ===================================================
# 5. SequenceTracker - מעקב אחר רצפי ריבוי דוברים
# ===================================================
class SequenceTracker:
    """
    עוקב אחר רצפים של ריבוי דוברים עם מכונת מצבים.
    
    מצבים:
    - None          → מצב רגיל (אין הקשר מיוחד)
    - "lesson"      → לצורך השיעור (אחרי שאלה פתוחה)
    - "disruption"  → הפרעה (אחרי ניסיון השתקה)
    
    תנאי מעבר:
    1. דובר יחיד + שאלה פתוחה → lesson
    2. דובר יחיד + ניסיון השתקה → disruption
    3. דובר יחיד רגיל → None (איפוס)
    
    תנאי עצירה של לולאה:
    1. חלון הבא = דובר יחיד (נגמר הריבוי)
    2. זוהה ניסיון השתקה (המורה עצרה את הדיון)
    """
    
    def __init__(self):
        self.current_state = None
        self.state_start_time = None
    
    def update(self, window_type: str, trigger_type: str, timestamp: float) -> Optional[str]:
        """
        עדכון המצב לפי החלון הנוכחי.
        
        Args:
            window_type: "דובר_יחיד" / "ריבוי_דוברים" / "רעש"
            trigger_type: "שאלה_פתוחה" / "ניסיון_השתקה" / "רגיל"
            timestamp: זמן החלון (שניות)
        
        Returns:
            context_label: "לצורך_השיעור" / "הפרעה" / None
        """
        
        if window_type == "דובר_יחיד":
            # דובר יחיד - בדיקת trigger
            if trigger_type == "שאלה_פתוחה":
                # המורה שאלה שאלה → מצב "לצורך השיעור"
                self.current_state = "lesson"
                self.state_start_time = timestamp
                return None  # דובר יחיד לא מסווג
            
            elif trigger_type == "ניסיון_השתקה":
                # המורה מנסה להשתיק → מצב "הפרעה"
                self.current_state = "disruption"
                self.state_start_time = timestamp
                return None
            
            else:
                # דובר יחיד רגיל - איפוס מצב
                self.current_state = None
                return None
        
        elif window_type == "ריבוי_דוברים":
            # ריבוי דוברים - החלטה לפי המצב הנוכחי
            if self.current_state == "lesson":
                # אנחנו במצב "לצורך השיעור" → כל הריבוי = חיובי
                return "לצורך_השיעור"
            
            elif self.current_state == "disruption":
                # אנחנו במצב "הפרעה" → כל הריבוי = שלילי
                return "הפרעה"
            
            else:
                # אין מצב מיוחד - ברירת מחדל: הפרעה
                return "הפרעה"
        
        else:  # רעש
            return None


# ===================================================
# 6. HEBERTContextAnalyzer - המערכת המלאה
# ===================================================
class HEBERTContextAnalyzer:
    """
    מערכת ניתוח הקשרי מלאה מבוססת HEBERT.
    
    משלבת:
    - Whisper ASR (תמלול)
    - HEBERT (הבנת הקשר)
    - Trigger Classifier (זיהוי שאלה/השתקה)
    - Context Classifier (לצורך/הפרעה)
    - Sequence Tracker (מעקב רצפים)
    """
    
    # קטגוריות סיווג
    DISTURBANCE = "הפרעה"
    ACTIVE_LEARNING = "למידה_פעילה"
    DISCUSSION = "פתיחה_לדיון"
    UNKNOWN = "לא_ידוע"
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        """אתחול המערכת המלאה."""
        self.sample_rate = sample_rate
        
        print(f"\n  {'='*50}")
        print(f"  אתחול מערכת HEBERT")
        print(f"  {'='*50}")
        
        # אתחול רכיבים
        self.asr = WhisperASR(model_size="base")
        self.hebert = HEBERTModel(model_name="avichr/heBERT")
        self.trigger_classifier = TriggerClassifier(self.hebert)
        self.context_classifier = ContextClassifier(self.hebert)
        self.sequence_tracker = SequenceTracker()
        
        print(f"  {'='*50}\n")
    
    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """המרת אודיו לטקסט."""
        return self.asr.transcribe(audio, self.sample_rate)
    
    def analyze_trigger(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        ניתוח חלון דובר יחיד - האם יש trigger?
        
        Returns:
            (trigger_type, confidence)
        """
        text = self.transcribe(audio)
        if not text:
            return "רגיל", 0.5
        
        return self.trigger_classifier.classify(text)
    
    def analyze_context(self, audio: np.ndarray) -> Dict:
        """
        ניתוח חלון ריבוי דוברים - מה ההקשר?
        
        Returns:
            מילון עם: טקסט, קטגוריה, ביטחון
        """
        text = self.transcribe(audio)
        
        if not text:
            return {
                'transcribed_text': None,
                'context_category': self.DISTURBANCE,
                'confidence': 0.6
            }
        
        category, confidence = self.context_classifier.classify(text)
        
        # מיפוי לקטגוריות הישנות (תאימות)
        if category == "לצורך_השיעור":
            final_category = self.ACTIVE_LEARNING
        else:
            final_category = self.DISTURBANCE
        
        return {
            'transcribed_text': text,
            'context_category': final_category,
            'confidence': confidence
        }
    
    def analyze(self, audio: np.ndarray) -> Dict:
        """
        ניתוח מלא של חלון אודיו (תאימות עם הקוד הישן).
        
        Args:
            audio: מערך numpy של דגימות אודיו
        
        Returns:
            מילון עם תוצאות הניתוח
        """
        return self.analyze_context(audio)
    
    def has_open_question(self, audio: np.ndarray) -> bool:
        """
        בדיקה האם בקטע אודיו יש שאלה פתוחה.
        
        Args:
            audio: מערך numpy של דגימות אודיו
        
        Returns:
            True אם זוהתה שאלה פתוחה
        """
        trigger_type, confidence = self.analyze_trigger(audio)
        return trigger_type == "שאלה_פתוחה" and confidence > 0.6
    
    def has_silencing_words(self, audio: np.ndarray) -> bool:
        """
        בדיקה האם בקטע אודיו יש ניסיון השתקה.
        
        Args:
            audio: מערך numpy של דגימות אודיו
        
        Returns:
            True אם זוהה ניסיון השתקה
        """
        trigger_type, confidence = self.analyze_trigger(audio)
        return trigger_type == "ניסיון_השתקה" and confidence > 0.6
