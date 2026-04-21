# ===================================================
# db_manager.py — ניהול מסד נתונים SQL Server
# ===================================================
# שומר את תוצאות הניתוח במסד נתונים מובנה (SQL Server).
# מאפשר שאילתות, דוחות, והשוואות בין שיעורים לאורך זמן.
#
# טבלאות:
#   Lessons     — שורה לכל שיעור (מורה, כיתה, תאריך, ציון)
#   TimeWindows — שורה לכל חלון זמן (תוצאות מפורטות)

import pyodbc
from typing import Dict, List, Optional
from datetime import datetime
from config import DB_SERVER, DB_NAME, DB_DRIVER


class DBManager:
    """מנהל חיבור ופעולות מול מסד הנתונים SQL Server."""

    def __init__(self):
        # מחרוזת חיבור — משתמש ב-Windows Authentication
        self.conn_str = (
            f"DRIVER={DB_DRIVER};"
            f"SERVER={DB_SERVER};"
            f"DATABASE={DB_NAME};"
            f"Trusted_Connection=yes;"
        )
        self.conn = None

    # ===============================================
    # חיבור וניתוק
    # ===============================================
    def connect(self):
        """פתיחת חיבור למסד הנתונים."""
        self.conn = pyodbc.connect(self.conn_str)
        print(f"  [V] מחובר למסד {DB_NAME}")

    def close(self):
        """סגירת החיבור."""
        if self.conn:
            self.conn.close()

    # ===============================================
    # יצירת טבלאות (הרצה ראשונה בלבד)
    # ===============================================
    def create_tables(self):
        """יצירת טבלאות המערכת אם לא קיימות."""
        cursor = self.conn.cursor()

        # טבלת שיעורים — שורה אחת לכל שיעור שנותח
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Lessons')
            CREATE TABLE Lessons (
                id INT IDENTITY(1,1) PRIMARY KEY,
                teacher_name NVARCHAR(100),
                class_name NVARCHAR(50),
                lesson_date DATE,
                start_time TIME,
                duration_sec FLOAT,
                avg_attention_score FLOAT,
                grade NVARCHAR(20),
                file_path NVARCHAR(500),
                created_at DATETIME DEFAULT GETDATE()
            )
        """)

        # טבלת חלונות זמן — תוצאות מפורטות לכל חלון
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'TimeWindows')
            CREATE TABLE TimeWindows (
                id INT IDENTITY(1,1) PRIMARY KEY,
                lesson_id INT FOREIGN KEY REFERENCES Lessons(id),
                start_sec FLOAT,
                end_sec FLOAT,
                speech_ratio FLOAT,
                rms_value FLOAT,
                rms_db FLOAT,
                rms_level NVARCHAR(20),
                audio_type NVARCHAR(30),
                speaker_type NVARCHAR(30),
                overlap_score FLOAT,
                context_category NVARCHAR(30),
                attention_score FLOAT
            )
        """)

        self.conn.commit()
        print("  [V] טבלאות נוצרו בהצלחה")

    # ===============================================
    # שמירת תוצאות שיעור
    # ===============================================
    def save_lesson(self, results: Dict, teacher: str, class_name: str) -> int:
        """
        שמירת תוצאות שיעור שלם למסד הנתונים.
        מחזיר את ה-ID של השיעור שנוצר.
        """
        cursor = self.conn.cursor()
        score = results['lesson_score']
        now = datetime.now()

        # הכנסת שורה לטבלת Lessons
        cursor.execute("""
            INSERT INTO Lessons
            (teacher_name, class_name, lesson_date, start_time,
             duration_sec, avg_attention_score, grade, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, teacher, class_name, now.date(), now.time(),
             results['duration_sec'], score['avg_score'],
             score['grade'], results['file'])

        # שליפת ה-ID שנוצר אוטומטית
        cursor.execute("SELECT @@IDENTITY")
        lesson_id = int(cursor.fetchone()[0])

        # הכנסת כל חלונות הזמן לטבלת TimeWindows
        for w in results['windows']:
            cursor.execute("""
                INSERT INTO TimeWindows
                (lesson_id, start_sec, end_sec, speech_ratio,
                 rms_value, rms_db, rms_level, audio_type,
                 speaker_type, overlap_score, context_category,
                 attention_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, lesson_id, w['start_sec'], w['end_sec'],
                 w['speech_ratio'], w['rms'], w['rms_db'],
                 w['rms_level'], w['audio_type'],
                 w['speaker_type'], w['overlap_score'],
                 w['context_category'], w['attention_score'])

        self.conn.commit()
        print(f"  [V] שיעור #{lesson_id} נשמר במסד הנתונים")
        return lesson_id

    # ===============================================
    # שליפת דוחות
    # ===============================================
    def get_lesson_report(self, lesson_id: int) -> Optional[Dict]:
        """שליפת נתוני שיעור לפי מזהה."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM Lessons WHERE id = ?", lesson_id)
        row = cursor.fetchone()
        if not row:
            return None
        columns = [col[0] for col in cursor.description]
        return dict(zip(columns, row))

    def get_teacher_summary(self, teacher_name: str) -> List[Dict]:
        """שליפת סיכום כל השיעורים של מורה מסוימת."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT lesson_date, class_name, avg_attention_score, grade
            FROM Lessons WHERE teacher_name = ?
            ORDER BY lesson_date DESC
        """, teacher_name)
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
