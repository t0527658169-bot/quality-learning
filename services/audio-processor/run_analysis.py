# ===================================================
# run_analysis.py — נקודת הכניסה להרצת המערכת
# ===================================================
# שימוש:
#   python run_analysis.py <קובץ_שמע>     — ניתוח קובץ אמיתי
#   python run_analysis.py --test          — הרצת בדיקה עם אודיו מלאכותי
#
# דוגמאות:
#   python run_analysis.py lesson.wav
#   python run_analysis.py recording.mp3
#   python run_analysis.py --test

import sys
import os


def main():
    """נקודת הכניסה הראשית."""

    # מצב בדיקה — יצירת אודיו מלאכותי
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_test()
        return

    # בדיקה שסופק נתיב לקובץ
    if len(sys.argv) < 2:
        print("שימוש:")
        print("  python run_analysis.py <audio_file>   — ניתוח קובץ שמע")
        print("  python run_analysis.py --test         — הרצת בדיקה")
        print("\nדוגמה: python run_analysis.py lesson.wav")
        return

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"❌ קובץ לא נמצא: {file_path}")
        return

    # הרצת צנרת העיבוד
    from main_pipeline import AudioPipeline
    pipeline = AudioPipeline()
    results = pipeline.process_file(file_path)

    # ניסיון שמירה במסד נתונים (אופציונלי — לא חוסם)
    try:
        from db_manager import DBManager
        db = DBManager()
        db.connect()
        db.create_tables()
        db.save_lesson(results, teacher="מורה_לדוגמה", class_name="כיתה_א")
        db.close()
    except Exception as e:
        print(f"\n  ⚠ לא ניתן לשמור במסד נתונים: {e}")
        print("    (התוצאות הודפסו למסך בהצלחה)")


def run_test():
    """הרצת בדיקה עם אודיו סינתטי (לא דורש קובץ אמיתי)."""
    import numpy as np

    print("\n[TEST] הרצת בדיקה עם אודיו סינתטי (30 שניות)...")

    sr = 16000
    duration = 30
    t = np.linspace(0, duration, sr * duration)
    audio = np.zeros_like(t)

    # 0-10 שניות: סימולציית דיבור (גל סינוס עם הרמוניות)
    f0 = 150  # תדר בסיסי של קול אנושי
    audio[:10*sr] = (
        0.25 * np.sin(2 * np.pi * f0 * t[:10*sr]) +
        0.12 * np.sin(2 * np.pi * 2 * f0 * t[:10*sr]) +
        0.08 * np.sin(2 * np.pi * 3 * f0 * t[:10*sr])
    )

    # 10-20 שניות: סימולציית רעש (אקראי)
    audio[10*sr:20*sr] = np.random.normal(0, 0.08, 10*sr)

    # 20-30 שניות: סימולציית דיבור + רעש (חפיפה)
    audio[20*sr:] = (
        0.20 * np.sin(2 * np.pi * 200 * t[:10*sr]) +
        np.random.normal(0, 0.04, 10*sr)
    )

    # רעש רקע קל בכל ההקלטה
    audio += np.random.normal(0, 0.005, len(audio))

    # שמירה לקובץ זמני והרצת הצנרת
    import tempfile
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        temp_path = f.name

    from main_pipeline import AudioPipeline
    pipeline = AudioPipeline()
    results = pipeline.process_file(temp_path)

    # ניקוי
    os.remove(temp_path)
    print("\n[OK] בדיקה הושלמה בהצלחה!")
    return results


if __name__ == "__main__":
    main()
