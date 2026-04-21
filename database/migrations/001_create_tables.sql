-- ===================================================
-- 001_create_tables.sql — סקריפט יצירת טבלאות
-- ===================================================
-- פרויקט Quality Learning — מסד נתונים SQL Server
-- הרצה: פתחי את הסקריפט ב-SSMS ולחצי Execute
-- ===================================================

-- יצירת מסד הנתונים (אם לא קיים)
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'QualityLearning')
BEGIN
    CREATE DATABASE QualityLearning;
END
GO

USE QualityLearning;
GO

-- ===================================================
-- טבלת מורים
-- ===================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Teachers')
CREATE TABLE Teachers (
    id INT IDENTITY(1,1) PRIMARY KEY,
    full_name NVARCHAR(100) NOT NULL,       -- שם מלא
    email NVARCHAR(200),                     -- אימייל
    password_hash NVARCHAR(256) NOT NULL,   -- סיסמה מוצפנת (לא טקסט גלוי!)
    role NVARCHAR(20) DEFAULT N'teacher',   -- תפקיד: teacher / admin
    created_at DATETIME DEFAULT GETDATE()
);
GO

-- ===================================================
-- טבלת כיתות
-- ===================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Classes')
CREATE TABLE Classes (
    id INT IDENTITY(1,1) PRIMARY KEY,
    class_name NVARCHAR(50) NOT NULL,       -- שם הכיתה (כגון: "ז'1")
    grade_level INT,                         -- שכבה (1-12)
    created_at DATETIME DEFAULT GETDATE()
);
GO

-- ===================================================
-- טבלת שיעורים — שורה אחת לכל שיעור שנותח
-- ===================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Lessons')
CREATE TABLE Lessons (
    id INT IDENTITY(1,1) PRIMARY KEY,
    teacher_name NVARCHAR(100),             -- שם המורה
    class_name NVARCHAR(50),                -- שם הכיתה
    lesson_date DATE,                        -- תאריך השיעור
    start_time TIME,                         -- שעת התחלה
    duration_sec FLOAT,                      -- משך השיעור בשניות
    avg_attention_score FLOAT,              -- ציון קשב ממוצע (0-100)
    grade NVARCHAR(20),                      -- דירוג מילולי (מצוין/טוב/בינוני/נמוך)
    file_path NVARCHAR(500),                -- נתיב קובץ השמע
    created_at DATETIME DEFAULT GETDATE()
);
GO

-- ===================================================
-- טבלת חלונות זמן — תוצאות מפורטות לכל חלון
-- ===================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'TimeWindows')
CREATE TABLE TimeWindows (
    id INT IDENTITY(1,1) PRIMARY KEY,
    lesson_id INT FOREIGN KEY REFERENCES Lessons(id),  -- קשר לשיעור
    start_sec FLOAT,                        -- שנייה התחלה
    end_sec FLOAT,                          -- שנייה סיום
    speech_ratio FLOAT,                     -- אחוז דיבור (0.0 - 1.0)
    rms_value FLOAT,                        -- ערך RMS גולמי
    rms_db FLOAT,                           -- ערך בדציבלים
    rms_level NVARCHAR(20),                 -- סיווג: שקט/רגיל/רועש
    audio_type NVARCHAR(30),                -- סוג צליל: דיבור/רעש/פטפטת
    speaker_type NVARCHAR(30),              -- דובר_יחיד / ריבוי_דוברים
    overlap_score FLOAT,                    -- ציון חפיפה (0.0 - 1.0)
    context_category NVARCHAR(30),          -- למידה_פעילה / הפרעה / לא_ידוע
    attention_score FLOAT                   -- ציון קשב (0-100)
);
GO

-- ===================================================
-- טבלת דוחות יומיים (סיכום)
-- ===================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'DailyReports')
CREATE TABLE DailyReports (
    id INT IDENTITY(1,1) PRIMARY KEY,
    report_date DATE,
    class_name NVARCHAR(50),
    total_lessons INT,
    avg_score FLOAT,
    best_lesson_id INT,
    worst_lesson_id INT,
    created_at DATETIME DEFAULT GETDATE()
);
GO

PRINT N'✓ כל הטבלאות נוצרו בהצלחה!';
GO
