# HEBERT Context Analyzer - מדריך מפורט

## 📚 תוכן עניינים
1. [סקירה כללית](#סקירה-כללית)
2. [למה זה לא Black Box?](#למה-זה-לא-black-box)
3. [ארכיטקטורת BERT - הסבר מלא](#ארכיטקטורת-bert)
4. [רכיבי המערכת](#רכיבי-המערכת)
5. [זרימת העבודה](#זרימת-העבודה)
6. [דוגמאות שימוש](#דוגמאות-שימוש)

---

## סקירה כללית

מערכת **HEBERT Context Analyzer** היא מערכת ניתוח הקשרי מבוססת BERT (Bidirectional Encoder Representations from Transformers) המותאמת לעברית.

### מטרות המערכת:
1. **זיהוי Triggers** - האם המורה שאלה שאלה פתוחה או ניסתה להשתיק?
2. **סיווג הקשר** - האם ריבוי דוברים הוא לצורך השיעור או הפרעה?
3. **מעקב רצפים** - מעקב אחר רצפי ריבוי דוברים עם מכונת מצבים

---

## למה זה לא Black Box?

### ❌ מה זה Black Box?
```python
# דוגמה ל-Black Box:
from transformers import pipeline

classifier = pipeline("text-classification", model="hebert")
result = classifier("בואו נדון")  # מה קרה פנימה? לא יודעים!
```

**בעיות:**
- לא יודעים איך המודל מקבל החלטות
- לא יכולים להדפיס ביניים
- לא מבינים את הלוגיקה

---

### ✅ הגישה שלנו - Glass Box (קופסת זכוכית)

```python
# הגישה שלנו - שקיפות מלאה:
text = "בואו נדון"

# שלב 1: Tokenization - פירוק לטוקנים
tokens = tokenizer.tokenize(text)  # [101, 1234, 5678, 102]
print(f"Tokens: {tokens}")

# שלב 2: Embedding - המרה לוקטורים
embeddings = embedding_layer(tokens)  # [[0.2, -0.5, ...], ...]
print(f"Embeddings shape: {embeddings.shape}")

# שלב 3: Transformer - עיבוד הקשרי
for i, layer in enumerate(transformer_layers):
    embeddings = layer(embeddings)
    print(f"Layer {i} output: {embeddings[0][:5]}")  # 5 ערכים ראשונים

# שלב 4: Classification - סיווג
logits = classifier(embeddings)  # [2.3, -0.5, 0.1]
probs = softmax(logits)  # [0.9, 0.05, 0.05]
print(f"Probabilities: {probs}")
```

**יתרונות:**
- ✅ רואים כל שלב
- ✅ מבינים את ההחלטות
- ✅ יכולים לדבג ולשפר

---

## ארכיטקטורת BERT

### מה זה BERT?

**BERT** = Bidirectional Encoder Representations from Transformers

זה מודל שפה שלומד להבין טקסט על ידי:
1. קריאה דו-כיוונית (שמאל→ימין וימין←שמאל)
2. הבנת הקשר בין מילים
3. ייצוג סמנטי של משפטים

---

### מבנה BERT - 4 שכבות עיקריות

```
טקסט: "בואו נדון בנושא"
    ↓
┌─────────────────────────────────────┐
│  1. Tokenizer                       │
│  פירוק לטוקנים                     │
│  "בואו נדון בנושא" → [101, 1234, 5678, 102]
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. Embedding Layer                 │
│  המרה לוקטורים (768 מימדים)       │
│  [101] → [0.2, -0.5, 0.1, ...]     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. Transformer Blocks (×12)        │
│  עיבוד הקשרי עמוק                  │
│  - Multi-Head Attention             │
│  - Feed-Forward Network             │
│  - Layer Normalization              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  4. Pooler                          │
│  סיכום לוקטור אחד                  │
│  [batch, seq_len, 768] → [768]     │
└─────────────────────────────────────┘
    ↓
  וקטור 768 מימדים
```

---

### שכבה 1: Tokenizer

**מה זה עושה?**
פירוק טקסט לטוקנים (יחידות בסיסיות).

**דוגמה:**
```python
text = "בואו נדון בנושא"

# Tokenization
tokens = tokenizer.tokenize(text)
# → ["[CLS]", "בואו", "נדון", "בנושא", "[SEP]"]

# המרה למספרים
token_ids = tokenizer.convert_tokens_to_ids(tokens)
# → [101, 1234, 5678, 9012, 102]
```

**טוקנים מיוחדים:**
- `[CLS]` (101) - תחילת משפט
- `[SEP]` (102) - סוף משפט
- `[PAD]` (0) - ריפוד (padding)

---

### שכבה 2: Embedding Layer

**מה זה עושה?**
המרת כל טוקן (מספר) לוקטור בן 768 מימדים.

**למה 768 מימדים?**
כל מימד מייצג "תכונה" סמנטית:
- מימד 0: האם זו פעולה?
- מימד 1: האם זו שאלה?
- מימד 2: האם זו רגש חיובי?
- ... (עוד 765 תכונות)

**דוגמה:**
```python
token_id = 1234  # "בואו"

# Embedding lookup
embedding = embedding_matrix[token_id]
# → [0.234, -0.567, 0.891, ..., 0.123]  # 768 מספרים

# כל מספר = "כמה המילה קשורה לתכונה הזו"
```

**מאיפה המספרים?**
- המספרים נלמדו מאימון על מיליוני משפטים בעברית
- המודל למד שמילים דומות צריכות וקטורים דומים
- למשל: "בואו", "נלך", "נעשה" → וקטורים קרובים

---

### שכבה 3: Transformer Blocks

**מה זה עושה?**
עיבוד הקשרי - מבין איך מילים קשורות זו לזו.

#### 3.1 Multi-Head Attention

**רעיון:**
כל מילה "מסתכלת" על כל המילים האחרות ושואלת: "מי רלוונטי לי?"

**דוגמה:**
```
משפט: "בואו נדון בנושא"

Attention של "נדון":
- "בואו"   → 0.8  (רלוונטי מאוד! זה הפעולה שמובילה)
- "נדון"   → 0.1  (עצמי - פחות חשוב)
- "בנושא" → 0.6  (רלוונטי - זה מה שדנים בו)

→ "נדון" מקבל מידע מ"בואו" ו"בנושא"
```

**נוסחה מתמטית:**
```
Attention(Q, K, V) = softmax(Q·K^T / √d) · V

איפה:
- Q (Query) = "מה אני מחפש?"
- K (Key) = "מה יש לי?"
- V (Value) = "מה הערך שלי?"
```

**למה Multi-Head?**
12 "ראשים" - כל אחד מסתכל על היבט אחר:
- ראש 1: קשרים תחביריים (נושא-נשוא)
- ראש 2: קשרים סמנטיים (מילים דומות)
- ראש 3: קשרים זמניים (עבר-הווה-עתיד)
- ... (עוד 9 ראשים)

#### 3.2 Feed-Forward Network

**מה זה עושה?**
עיבוד עמוק של כל מילה בנפרד.

**מבנה:**
```
Input (768) → Linear (3072) → ReLU → Linear (768) → Output
```

**למה 3072?**
- הרחבה פי 4 (768 × 4 = 3072)
- נותן למודל "מקום לחשוב"
- אחר כך דחיסה חזרה ל-768

#### 3.3 Layer Normalization

**מה זה עושה?**
ייצוב הערכים - מונע "התפוצצות" או "היעלמות" של מספרים.

**נוסחה:**
```
LayerNorm(x) = (x - mean(x)) / std(x)
```

---

### שכבה 4: Pooler

**מה זה עושה?**
סיכום כל המשפט לוקטור אחד.

**איך?**
לוקח את הוקטור של `[CLS]` (הטוקן הראשון) - הוא למד לסכם את כל המשפט.

```python
# לפני Pooler:
# [batch, seq_len, 768] = [1, 5, 768]
# 5 מילים, כל אחת 768 מימדים

# אחרי Pooler:
# [batch, 768] = [1, 768]
# וקטור אחד שמסכם את כל המשפט
```

---

## רכיבי המערכת

### 1. WhisperASR - תמלול

```python
class WhisperASR:
    """המרת דיבור לטקסט."""
    
    def transcribe(self, audio):
        # Whisper מעבד את האודיו
        result = self.model.transcribe(audio, language="he")
        return result["text"]
```

**למה Whisper?**
- דיוק גבוה בעברית
- עמיד לרעשי רקע
- קוד פתוח

**זה Black Box?**
כן, אבל זה בסדר - ASR הוא כלי עזר, לא חלק מהמטלה.

---

### 2. HEBERTModel - מודל BERT

```python
class HEBERTModel:
    """מודל BERT עברי."""
    
    def encode(self, text):
        # 1. Tokenization
        inputs = self.tokenizer(text)
        
        # 2-4. BERT processing
        outputs = self.model(**inputs)
        
        # 5. Pooling
        embedding = outputs.pooler_output[0]
        
        return embedding.numpy()  # [768]
```

**מה אנחנו כותבים?**
- ✅ הלוגיקה של encode
- ✅ הקריאות לכל שלב
- ✅ ההמרות בין פורמטים

**מה באים מבחוץ?**
- 📦 המשקולות (110M מספרים)
- 📦 אוצר המילים (vocab.txt)

---

### 3. TriggerClassifier - זיהוי שאלה/השתקה

```python
class TriggerClassifier:
    """זיהוי trigger בחלון דובר יחיד."""
    
    def classify(self, text):
        # 1. HEBERT מעבד
        embedding = self.hebert.encode(text)  # [768]
        
        # 2. שכבת סיווג
        logits = self.classifier(embedding)  # [3]
        probs = softmax(logits)
        
        # 3. חיזוק עם מילות מפתח
        if "בואו נדון" in text:
            probs[0] *= 2.0  # חיזוק "שאלה פתוחה"
        
        # 4. בחירה
        predicted = argmax(probs)
        return labels[predicted], probs[predicted]
```

**למה hybrid approach?**
- HEBERT מבין הקשר סמנטי
- מילות מפתח מחזקות החלטות ברורות
- שילוב = דיוק גבוה יותר

---

### 4. ContextClassifier - לצורך/הפרעה

```python
class ContextClassifier:
    """סיווג ריבוי דוברים."""
    
    def classify(self, text):
        # 1. HEBERT מעבד
        embedding = self.hebert.encode(text)
        
        # 2. סיווג
        logits = self.classifier(embedding)  # [2]
        probs = softmax(logits)
        
        # 3. בחירה
        predicted = argmax(probs)
        return labels[predicted], probs[predicted]
```

---

### 5. SequenceTracker - מעקב רצפים

```python
class SequenceTracker:
    """מכונת מצבים לריבוי דוברים."""
    
    def update(self, window_type, trigger_type, timestamp):
        if window_type == "דובר_יחיד":
            if trigger_type == "שאלה_פתוחה":
                self.current_state = "lesson"
            elif trigger_type == "ניסיון_השתקה":
                self.current_state = "disruption"
            else:
                self.current_state = None
        
        elif window_type == "ריבוי_דוברים":
            if self.current_state == "lesson":
                return "לצורך_השיעור"
            elif self.current_state == "disruption":
                return "הפרעה"
            else:
                return "הפרעה"  # ברירת מחדל
```

---

## זרימת העבודה

### תרחיש 1: המורה שואלת שאלה

```
חלון 1 (0-3s): דובר יחיד
  ↓
  Whisper: "מה דעתכם על הנושא?"
  ↓
  HEBERT Trigger: "שאלה_פתוחה" (0.95)
  ↓
  State → "lesson"
  ↓
חלון 2 (3-6s): ריבוי דוברים
  ↓
  State = "lesson" → "לצורך_השיעור" ✅
  ↓
חלון 3 (6-9s): ריבוי דוברים
  ↓
  State = "lesson" → "לצורך_השיעור" ✅
  ↓
חלון 4 (9-12s): דובר יחיד
  ↓
  Whisper: "שקט בבקשה"
  ↓
  HEBERT Trigger: "ניסיון_השתקה" (0.92)
  ↓
  State → "disruption"
  ↓
חלון 5 (12-15s): ריבוי דוברים
  ↓
  State = "disruption" → "הפרעה" ❌
```

---

## דוגמאות שימוש

### דוגמה 1: ניתוח חלון בודד

```python
from hebert_context_analyzer import HEBERTContextAnalyzer
import numpy as np

# אתחול
analyzer = HEBERTContextAnalyzer()

# טעינת אודיו (3 שניות)
audio = np.load("window_001.npy")

# ניתוח
result = analyzer.analyze(audio)

print(f"טקסט: {result['transcribed_text']}")
print(f"קטגוריה: {result['context_category']}")
print(f"ביטחון: {result['confidence']:.2%}")
```

### דוגמה 2: זיהוי trigger

```python
# חלון דובר יחיד
audio_single = np.load("teacher_speech.npy")

# בדיקת trigger
trigger_type, confidence = analyzer.analyze_trigger(audio_single)

if trigger_type == "שאלה_פתוחה":
    print("המורה שאלה שאלה - הריבוי הבא יהיה לצורך השיעור")
elif trigger_type == "ניסיון_השתקה":
    print("המורה מנסה להשתיק - הריבוי הבא יהיה הפרעה")
```

---

## סיכום

### מה כתבנו בעצמנו? ✅
1. ✅ WhisperASR wrapper
2. ✅ HEBERTModel encode logic
3. ✅ TriggerClassifier (שכבת סיווג + לוגיקה)
4. ✅ ContextClassifier (שכבת סיווג + לוגיקה)
5. ✅ SequenceTracker (מכונת מצבים)
6. ✅ כל הלוגיקה של הניתוח

### מה טענו מבחוץ? 📦
1. 📦 Whisper model (ASR - כלי עזר)
2. 📦 HEBERT weights (110M מספרים)
3. 📦 Tokenizer vocab (אוצר מילים)

### למה זה לא Black Box? 🔍
- ✅ אנחנו יודעים בדיוק מה כל שלב עושה
- ✅ אפשר להדפיס ביניים ולראות
- ✅ אפשר לשנות ולשפר
- ✅ רק המשקולות (המספרים) באות מאימון
- ✅ זה כמו להשתמש במחשבון - אנחנו לא בנינו את השבבים, אבל אנחנו יודעים איך הוא עובד!

---

**בהצלחה! 🚀**
