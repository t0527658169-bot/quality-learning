# ===================================================
# hebert_context_analyzer.py — ניתוח הקשרי עם HEBERT
# ===================================================

import numpy as np
import torch
from typing import Optional, Tuple
from transformers import AutoTokenizer
from transformers import AutoModel
import whisper


from config import SAMPLE_RATE, HEBERT_OPENING_THRESHOLD, HEBERT_SILENCING_THRESHOLD


# ===================================================
# 1. WhisperASR - המרת דיבור לטקסט
# ===================================================

class WhisperASR:
    
    #פונקציית אתחולם instance של המחלקה
    def __init__(self, model_size: str = "base"):
        # הדפסת הודעה לקונסול שניתן לראות שהטעינה מתחילה
        print(f"  [Whisper] טוען מודל {model_size}...")
        # טעינת מודל Whisper מהדיסק (מוריד אוטומטית בפעם הראשונה)
        self.model = whisper.load_model(model_size)
        # הדפסת הודעת הצלחה
        print(f"  [V] Whisper מוכן")

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Optional[str]:
        try:
            
            if audio.dtype != np.float32:
                # המרה ל-float32 אם הפורמט שונה 
                audio = audio.astype(np.float32)
            # בדיקה אם הערכים גדולים מ-1 (לא מנורמלים)
            if np.abs(audio).max() > 1.0:
                # חלוקה בערך המקסימלי — ממירה לטווח [-1, 1] שבו Whisper עובד
                audio = audio / np.abs(audio).max()

            # קריאה לתמלול עם Whisper:
       
            result = self.model.transcribe(
                audio,
                language="he",
                fp16=False,
                verbose=False
            )
            # חילוץ הטקסט מהתוצאה; .strip() מסיר רווחים מתחילה וסוף
            text = result.get("text", "").strip()
            return text if text else None

        except Exception as e:
            # טיפול בשגיאה — מדפיסים הודעה ומחזירים 
            print(f"  [!] Whisper error: {e}")
            return None

# ===================================================
# מחלקה 2: HEBERTModel 
# ===================================================

class HEBERTModel:
   
    # פונקציית אתחול — טוענת Tokenizer + מודל BERT מ-HuggingFace
    def __init__(self, model_name: str = "avichr/heBERT"):
        # הדפסת הודעת התחלת טעינה לקונסול
        print(f"  [HEBERT] טוען מודל {model_name}...")
        # טעינת ה-Tokenizer — ממיר טקסט לרצף מספרים (token IDs)
        # לדוגמה: "שלום וברכה" → [2, 1234, 5678, 3]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # טעינת מבנה המודל עם 110 מיליון פרמטרים מאומנים
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        print(f"  [V] HEBERT מוכן (12 שכבות, 768 מימדים)")

    def encode(self, text: str) -> np.ndarray:
       
        # אם הטקסט ריק או None — מחזירים וקטור אפסים (ניטרלי)
        if not text or not text.strip():
            return np.zeros(768, dtype=np.float32)

        # שלב 1: Tokenization — ממיר טקסט לטנסורים שהמודל מבין
        #   padding=True        — ממלא ברווחים אם המשפט קצר
        #   truncation=True     — חותך אם המשפט ארוך מדי
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        # העברת הטנסורים למכשיר הנכון (CPU) כדי שהמודל יוכל לעבד אותם
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # torch.no_grad() — חוסם חישוב gradients, 
        with torch.no_grad():
            # הרצת המודל — 12 שכבות Transformer מעבדות את הקלט
            outputs = self.model(**inputs)
            # pooler_output — וקטור [768] שמסכם את כל המשפט
            # [0] — לוקחים את הדגימה הראשונה (batch size = 1)
            embedding = outputs.pooler_output[0]

        # המרה מטנסור PyTorch למערך numpy רגיל לשימוש חיצוני
        return embedding.cpu().numpy()


# ===================================================
# 3. OpeningDetector - זיהוי פתיחה לדיון
# ===================================================
class OpeningDetector:
   

    # משפטי דוגמה המייצגים "פתיחה לדיון"
    OPENING_PROTOTYPES = [
        # שאלות חוות דעת
        "מה דעתכם על הנושא?",
        "מה אתם חושבים על זה?",
        "מי רוצה לשתף את דעתו?",
        "למה אתם אומרים כך?",
        "איך אתם רואים את זה?",
        "מה אתם אומרים?",
        # שאלות ידע
        "מי יודע את התשובה?",
        "מי יכול להסביר?",
        "מי זוכר מה למדנו?",
        "מי יכול לתת דוגמה?",
        # הזמנה לדיון
        "בואו נדון בזה יחד",
        "בואו נחשוב על זה",
        "בואו נשמע רעיונות",
        "נדון רגע בנושא הזה",
        "בואו נדבר על מה שקרה",
        "בואו נדבר על זה",
        "בואו נשוחח על מה שקרה אתמול",
        "בואו נשמע מה כולם חושבים",
        "בואו נשתף רגשות ומחשבות",
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
        "ספרו לי מה קרה",
        "ספרו לי מה קרה אתמול",
        "ספרו לי על מה שחוויתם",
        "ספרו לי מה הרגשתם",
        "מישהו יכול לספר מה קרה?",
        # שאלות שיחתיות
        "איך היה לכם?",
        "מה קרה בבית?",
        "מה חשבתם על המשימה?",
        # הזמנות עקיפות
        "אני רוצה שנדבר על מה שקרה",
        "אני רוצה שנדבר על זה ביחד",
        "אני רוצה לשמוע את דעתכם",
        "אני רוצה לשמוע מה אתם חושבים",
        "הייתי רוצה שנדבר על זה",
        "הייתי רוצה לשמוע מכם",
        "מעניין לי לדעת מה אתם חושבים",
        "סקרני לשמוע את דעתכם",
        "אשמח אם תשתפו",
        # שאלות רפלקטיביות
        "מה קרה אתמול?",
        "מה קרה לאחרונה?",
        "מה היה שם?",
        "מה חוויתם?",
        "מה הרגשתם כשזה קרה?",
        "ספרו לי על החוויה שלכם",
        "מה עלה לכם בראש כשראיתם?",
    ]

    # פונקציית אתחול — מחשבת את וקטור ה-prototype פעם אחת בלבד
    def __init__(self, hebert_model: HEBERTModel):
        # שמירת מופע HEBERT לשימוש בפונקציות פנימיות
        self.hebert = hebert_model
        # הדפסת הודעה שמתחיל חישוב ה-prototypes
        print("  [OpeningDetector] מחשב prototype embeddings...")
        # חישוב וקטור ה-prototype — ממוצע כל משפטי הפתיחה
        self._proto = self._compute_prototype(self.OPENING_PROTOTYPES)
        # הדפסת הודעת הצלחה
        print("  [V] OpeningDetector מוכן")

    def _compute_prototype(self, sentences) -> np.ndarray:
        """ממוצע וקטורי כל משפטי הדוגמה — מייצג את מרכז הקטגוריה."""
        # encode כל משפט ברשימה → מערך [N x 768] כאשר N = מספר המשפטים
        vecs = np.array([self.hebert.encode(s) for s in sentences])
        # חישוב ממוצע לאורך ציר 0 — ממוצע בין כל המשפטים → וקטור [768]
        mean = vecs.mean(axis=0)
        # חישוב הנורמה (אורך הוקטור) לצורך נרמול
        norm = np.linalg.norm(mean)
        # נרמול ל-unit vector (אורך=1) כדי ש-cosine similarity יעבוד נכון
        # + 1e-9 מונע חלוקה באפס אם הממוצע הוא וקטור אפסים
        return mean / (norm + 1e-9)

    def detect(self, text: Optional[str], threshold: float = 0.60) -> Tuple[bool, float]:
      
        #אם הטקסט ריק ן
        if not text or not text.strip():
            return False, 0.0

        # שלב 1: HEBERT ממיר את הטקסט לוקטור [768]
        emb = self.hebert.encode(text)
        # שלב 2: חישוב הנורמה (אורך הוקטור)
        norm = np.linalg.norm(emb)
        # אם הנורמה קרובה לאפס — הוקטור ריק, לא ניתן לחשב
        if norm < 1e-9:
            return False, 0.0
        # נרמול ל-unit vector
        emb = emb / norm

       
        # תוצאה: 1.0 = זהים, 0.0 = שונים לחלוטין
        score = float(np.dot(emb, self._proto))
        # שלב 4: החזרת החלטה (True/False) וציון הדמיון
        return score >= threshold, score


# ===================================================
# 4. SilencingDetector - זיהוי ניסיון השתקה
# ===================================================
class SilencingDetector:
   

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
        "רגע רגע",
        "עצור עצור",
        "די די",
        # ביטויי תסכול
        "אני לא שומעת כלום",
        "אני לא שומע אתכם",
        "מספיק רעש",
        "יש פה יותר מדי רעש",
        "אי אפשר לדבר ככה",
        "די לרעש",
        "אי אפשר לשמוע כלום",
        # בקשות מנומסות
        "אני צריכה שקט",
        "תנו לי רגע של שקט",
        "בבקשה תהיו בשקט",
        "אפשר שקט רגע?",
        "רגע של שקט בבקשה",
        # קריאה לסדר
        "בנות בנות בנות",
        "ילדים ילדים",
        "חברים חברים",
        "כולם אלי",
        "תקשיבו לי רגע",
        # אנגלית
        "quiet please",
        "everyone stop talking",
        "silence now",
    ]

    # פונקציית אתחול SilencingDetector — מחשבת prototype פעם אחת
    def __init__(self, hebert_model: HEBERTModel):
        # שמירת מופע HEBERT לשימוש בפונקציות פנימיות
        self.hebert = hebert_model
        # הדפסת הודעה שמתחיל חישוב ה-prototype
        print("  [SilencingDetector] מחשב prototype embeddings...")
        # חישוב וקטור ה-prototype — ממוצע קבוצי ביטויי ההשתקה
        # self._proto = self._comototype(self.SILENCING_PROTOTYPES)
        self._proto = self._compute_prototype(self.SILENCING_PROTOTYPES)
        # הדפסת הודעת הצלחה
        print("  [V] SilencingDetector מוכן")

    # פונקציה פנימית: חישוב prototype ממוצע מרשימת משפטים
    def _compute_prototype(self, sentences) -> np.ndarray:
        # encode כל משפט → [N x 768]
        vecs = np.array([self.hebert.encode(s) for s in sentences])
        # ממוצע בין כל המשפטים → [768]
        mean = vecs.mean(axis=0)
        # חישוב הנורמה לצורך נרמול
        norm = np.linalg.norm(mean)
        # נרמול ל-unit vector
        return mean / (norm + 1e-9)

    # פונקציית זיהוי ניסיון השתקה — הפונקציה הציבורית
    def detect(self, text: Optional[str], threshold: float = 0.60) -> Tuple[bool, float]:
       
        # אם הטקסט ריק — אין ניסיון השתקה, מחזירים False
        if not text or not text.strip():
            return False, 0.0

        # HEBERT ממיר את הטקסט לוקטור [768]
        emb = self.hebert.encode(text)
        # חישוב הנורמה (אורך הוקטור)
        norm = np.linalg.norm(emb)
        # אם הנורמה קרובה לאפס — וקטור לא תקין, לא ניתן לחשב
        if norm < 1e-9:
            return False, 0.0
        # נרמול ל-unit vector
        emb = emb / norm

        # cosine similarity — כמה הטקסט דומה לביטויי ההשתקה
        score = float(np.dot(emb, self._proto))
        # החזרת החלטה (True/False) וציון הדמיון
        return score >= threshold, score


# ===================================================
# מחלקה 5: HEBERTContextAnalyzer — הממשק החיצוני המאחד את הכל
# ===================================================

class HEBERTContextAnalyzer:
    
    # פונקציית אתחול — יוצרת את כל הרכיבים ומחברת אותם יחד
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        # שמירת קצב הדגימה (16,000 Hz) לשימוש בפונקציית התמלול
        self.sample_rate = sample_rate

        # הדפסת כותרת חזותית בקונסול — מסמנת תחילת אתחול HEBERT
        print(f"\n  {'='*50}")
        print(f"  אתחול מערכת HEBERT")
        print(f"  {'='*50}")

        # יצירת מופע Whisper — טוען את מודל התמלול לזיכרון
        self.asr = WhisperASR(model_size="base")
        # יצירת מופע HEBERT — טוען את הטרנספורמר העברי לזיכרון
        self.hebert = HEBERTModel(model_name="avichr/heBERT")
        # יצירת OpeningDetector — מחשב prototype לפתיחת דיון (משתמש ב-HEBERT)
        self.opening_detector = OpeningDetector(self.hebert)
        # יצירת SilencingDetector — מחשב prototype להשתקה (משתמש ב-HEBERT)
        self.silencing_detector = SilencingDetector(self.hebert)

        # הדפסת כותרת סיום אתחול
        print(f"  {'='*50}\n")

    def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        # העברת האודיו וקצב הדגימה ל-Whisper לתמלול
        return self.asr.transcribe(audio, self.sample_rate)

    # בדיקת 2 חלונות קודמים
    def check_opening(self, audio, threshold: float = HEBERT_OPENING_THRESHOLD):

        # שלב 1: תמלול האודיו לטקסט עברי באמצעות Whisper
        text = self._transcribe(audio)
       
        if not text:
            return False, 0.0, None

        #  בדיקת cosine similarity עם prototype 
        is_opening, score = self.opening_detector.detect(text, threshold)
        # החזרת שלושה ערכים: החלטה, ציון, וטקסט 
        return is_opening, score, text

    # פונקציה ציבורית 2: בדיקת ניסיון השתקה — נקראת מ-
    def check_silencing(self, audio, threshold: float = HEBERT_SILENCING_THRESHOLD):
       
        # שלב 1: תמלול האודיו לטקסט עברי באמצעות Whisper
        text = self._transcribe(audio)
    
        if not text:
            return False, 0.0, None

        # שלב 2-3: בדיקת cosine similarity עם prototype ביטויי ההשתקה
        is_silencing, score = self.silencing_detector.detect(text, threshold)
        # החזרת שלושה ערכים: החלטה, ציון, וטקסט
        return is_silencing, score, text