# Quality Learning — מערכת ניתוח איכות שיעור בזמן אמת

מערכת אוטומטית לניתוח הקלטות כיתה, המזהה אירועי פדגוגיים ומייצרת ציון איכות לשיעור.

---

## תיאור כללי

המערכת מקבלת הקלטת שמע של שיעור ומנתחת אותה חלון אחר חלון (3 שניות כל אחד).  
לכל חלון נקבעת קטגוריה פדגוגית, ובסיום ההקלטה מחושב ציון כולל לאיכות השיעור.

### קטגוריות שמזוהות:
| קטגוריה | משמעות |
|---|---|
| `דובר_יחיד` | המורה מסבירה — למידה ישירה |
| `למידה_פעילה` | ריבוי דוברים שהתחיל אחרי שאלה פתוחה — דיון בריא |
| `הפרעה` | ריבוי דוברים ללא הקשר פדגוגי, או לאחר ניסיון השתקה |
| `רעש` | אין דיבור אנושי — רעש סביבתי |

---

## ארכיטקטורה

```
הקלטת שמע (WAV)
        │
        ▼
┌─────────────────┐
│  audio-receiver │  ← C# ASP.NET Core — קבלת הקלטות ושמירה ל-SQL Server
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ audio-processor │  ← Python FastAPI — ניתוח השמע
│                 │
│  ┌───────────┐  │
│  │  Wiener   │  │  סינון רעש
│  │  Filter   │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │    VAD    │  │  זיהוי דיבור / שקט
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │   RMS     │  │  עוצמת צליל
│  │ Analyzer  │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼──────┐ │
│  │  Overlap   │ │  דובר יחיד / ריבוי דוברים
│  │  Detector  │ │  (7 מאפיינים אקוסטיים)
│  └─────┬──────┘ │
│        │        │
│  ┌─────▼──────┐ │
│  │  HeBERT   │ │  ניתוח הקשרי סמנטי
│  │ Analyzer  │ │  (Whisper + Hebrew BERT)
│  └─────┬──────┘ │
│        │        │
│  ┌─────▼──────┐ │
│  │ Attention  │ │  ציון קשב לכל חלון
│  │  Scorer   │ │
│  └─────┬──────┘ │
└────────┼────────┘
         │
         ▼
    SQL Server
  (תוצאות + ציון)
```

---

## רכיבי המערכת

### `wiener_filter.py` — סינון רעש
מסנן רעש סביבתי מהאות לפני הניתוח.  
מחשב פרופיל רעש מה-30% השקטים של ההקלטה ומחיל Wiener gain בתחום התדר.

### `vad_detector.py` — זיהוי פעילות דיבור
מזהה האם בחלון נתון יש דיבור אנושי, על בסיס סף RMS אדפטיבי.

### `rms_analyzer.py` — ניתוח עוצמה
מסווג כל חלון לרמת עוצמה: `שקט` / `רגיל` / `רועש`.

### `overlap_detector.py` — זיהוי ריבוי דוברים
מחליט האם מדבר דובר יחיד או מספר דוברים בו-זמנית.  
משתמש ב-7 מאפיינים אקוסטיים (harmonicity, kurtosis, energy_cv, ZCR ועוד) ללא מודל ML.

### `hebert_context_analyzer.py` — ניתוח הקשרי
כאשר מזוהה ריבוי דוברים, המערכת בודקת האם מדובר בדיון לגיטימי או הפרעה:
1. **Whisper ASR** — ממיר את 2 החלונות הקודמים לטקסט עברי
2. **HeBERT** (`avichr/heBERT`) — ממיר את הטקסט לוקטור סמנטי [768]
3. **Cosine Similarity** — משווה לvקטור prototype של משפטי פתיחה לדיון
4. אם הדמיון ≥ 0.60 → `למידה_פעילה`, אחרת → `הפרעה`

### `attention_scorer.py` — ציון קשב
מחשב ציון קשב לכל חלון ומצבר ציון כולל לשיעור.

---

## מחסנית טכנולוגית

| שכבה | טכנולוגיה |
|---|---|
| קבלת הקלטות | C# ASP.NET Core 8 |
| עיבוד שמע | Python 3.10, FastAPI |
| זיהוי דיבור | Whisper (OpenAI, מודל `base`) |
| ניתוח סמנטי | HeBERT (`avichr/heBERT`, 110M params) |
| מסד נתונים | SQL Server Express |
| תקשורת שירותים | REST API |

---

## התקנה והרצה

### דרישות מקדימות
- Python 3.8+
- .NET 8 SDK
- SQL Server Express
- חיבור אינטרנט (הורדת מודלים בפעם הראשונה)

### התקנת תלויות Python
```bash
cd services/audio-processor
pip install -r requirements.txt
```

> הערה: ההתקנה כוללת PyTorch + Transformers + Whisper — עשויה לקחת מספר דקות.

### הרצת שירות העיבוד
```bash
cd services/audio-processor
python web_api.py
# זמין על: http://localhost:5000
```

### הרצת שירות הקבלה
```bash
cd services/audio-receiver
dotnet run
# זמין על: https://localhost:5002
```

---

## משתני תצורה מרכזיים (`config.py`)

| משתנה | ערך ברירת מחדל | תיאור |
|---|---|---|
| `SAMPLE_RATE` | 16000 | קצב דגימה (Hz) |
| `WINDOW_DURATION_SEC` | 3.0 | אורך חלון ניתוח (שניות) |
| `OVERLAP_THRESHOLD` | 0.40 | סף ריבוי דוברים |
| `HEBERT_OPENING_THRESHOLD` | 0.60 | סף cosine similarity לפתיחת דיון |
| `HEBERT_SILENCING_THRESHOLD` | 0.60 | סף cosine similarity להשתקה |
| `VAD_SPEECH_RATIO_THRESHOLD` | 0.30 | מינימום אחוז דיבור בחלון |

---

## מבנה הפרויקט

```
├── services/
│   ├── audio-processor/     # Python — עיבוד שמע וניתוח
│   │   ├── main_pipeline.py
│   │   ├── hebert_context_analyzer.py
│   │   ├── overlap_detector.py
│   │   ├── wiener_filter.py
│   │   ├── vad_detector.py
│   │   ├── rms_analyzer.py
│   │   ├── attention_scorer.py
│   │   ├── config.py
│   │   └── web_api.py
│   └── audio-receiver/      # C# — קבלת הקלטות
│       ├── Controllers/
│       ├── Services/
│       └── Data/
├── database/
│   └── migrations/
└── docker-compose.yml
```
