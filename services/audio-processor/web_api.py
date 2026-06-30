# ===================================================
# web_api.py — FastAPI endpoint לקבלת הקלטות מה-C#
# ===================================================
# הפייתון מנתח ומחזיר JSON עם תוצאות.
# ===================================================
import os
import sys
import tempfile
import traceback

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# הוספת תיקיית השירות ל-path כדי שהייבואים יעבדו
sys.path.insert(0, os.path.dirname(__file__))

from main_pipeline import AudioPipeline

# יצירת אפליקציית FastAPI — זה ה"שרת" שמקשיב לבקשות HTTP
# title ו-description מופיעים ב-Swagger UI (http://localhost:5000/docs)
app = FastAPI(
    title="Quality Learning — Audio Processor",
    description="מנתח הקלטות שמע ומחזיר ציוני קשב"
)

# ===================================================
# Singleton Pattern - pipeline נוצר פעם אחת בלבד
# ===================================================
# בעיה: אתחול AudioPipeline לוקח ~30 שניות (טעינת HEBERT + Whisper)
# פתרון: יוצרים pipeline פעם אחת ושומרים בזיכרון
# כל בקשה חדשה מקבלת את אותה pipeline — חוסך זמן עצום
_pipeline: AudioPipeline | None = None

def get_pipeline() -> AudioPipeline:
    """מחזיר את ה-pipeline הקיים, או יוצר חדש אם זו הפעם הראשונה."""
    global _pipeline
    # בפעם הראשונה _pipeline=None — יוצרים ומאחסנים
    # בפעמים הבאות — מחזירים את הקיים (כבר טעון)
    if _pipeline is None:
        _pipeline = AudioPipeline()
    return _pipeline

def reset_pipeline():
    """מאפס את ה-pipeline — ייבנה מחדש בבקשה הבאה."""
    global _pipeline
    # None גורם ל-get_pipeline() ליצור instance חדש בבקשה הבאה
    _pipeline = None


# ===================================================
# GET /health בדיקת
# ===================================================
@app.get("/health")
def health():
    """בדיקת זמינות — ה-C# קורא לזה כדי לוודא ש-Python חי לפני שליחת הקובץ."""
    return {"status": "ok"}


# ===================================================
# POST /process — ניתוח הקלטה
# ===================================================

@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),          # File(...) = שדה חובה
    teacher_name: str = Form(""),          # Form("") = ערך ברירת מחדל ריק
    class_name: str = Form(""),
    lesson_date: str = Form(""),
    start_time: str = Form(""),
):
    
    # שמירת הקובץ לקובץ זמני בדיסק
    # os.path.splitext מחלץ את הסיומת: "audio.wav" → (".wav")
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    # NamedTemporaryFile יוצר קובץ זמני עם שם ייחודי
    # delete=False — לא נמחק אוטומטית (נמחק ידנית ב-finally)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        # קריאה אסינכרונית של תוכן הקובץ (לא חוסמת את השרת)
        content = await file.read()
        # כתיבה לדיסק
        tmp.write(content)

    try:
        # קבלת ה-pipeline (singleton — נוצר רק בפעם הראשונה)
        pipeline = get_pipeline()
        # הרצת ניתוח מלא — זה לוקח כמה דקות לפי אורך ההקלטה
        result = pipeline.process_file(tmp_path)
        # חילוץ ציון השיעור מהתוצאה
        lesson = result["lesson_score"]

        # בניית תשובת JSON מלאה
        return JSONResponse({
            "duration_sec":  result["duration_sec"],  # משך ההקלטה
            "positive_pct":  lesson["positive_pct"],  # אחוז חיובי
            "grade":         lesson["grade"],          # דירוג מילולי
            "negative_pct":  lesson["negative_pct"],  # אחוז הפרעות
            "noise_count":   lesson["noise_count"],    # חלונות רעש
            "total_windows": lesson["total_relevant"], # חלונות רלוונטיים
            "teacher_name":  teacher_name,             # מטא-דאטה שחזרנו כמו שהגיע
            "class_name":    class_name,
            "lesson_date":   lesson_date,
            "start_time":    start_time,
            "windows":       result["windows"]         # פרטים מלאים לכל חלון זמן
        })

    except Exception as e:
        # הדפסת stack trace מלא ל-stderr לצרכי debug
        traceback.print_exc()
        # איפוס ה-pipeline — אם קרסנו באמצע, ה-pipeline עשוי להיות במצב שגוי
        # (לדוגמה: Whisper עם state לא תקין)
        reset_pipeline()
        # 500 Internal Server Error עם הודעת השגיאה — C# יסמן הקלטה כ-Failed
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # מחיקת הקובץ הזמני — תמיד, גם אם נזרקה חריגה
        # os.unlink מוחק קובץ מהדיסק
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("web_api:app", host="0.0.0.0", port=port, reload=False)
