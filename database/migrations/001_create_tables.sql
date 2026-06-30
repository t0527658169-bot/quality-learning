-- ===================================================
-- 001_create_tables.sql —  יצירת טבלאות
-- ===================================================
-- פרויקט Quality Learning — מסד נתונים SQL Server
--  עם קשרי FK מלאים
-- ===================================================
--
--  תרשים קשרים:
--
--  Teachers ──┐
--             ├──► ScheduledLessons ◄──── Classes
--  SubjectCategories ─► Subjects ─┘      │
--                         ▼
--                     Recordings
--                         │
--                         ▼
--                     TimeWindows
--
-- ===================================================

-- יצירת מסד הנתונים (אם לא קיים)
BEGIN
    CREATE DATABASE QualityLearning;
END
GO

USE QualityLearning;
GO

-- ===================================================
-- 1. טבלת מורות
--    מפתח: id
-- ===================================================
CREATE TABLE Teachers (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    full_name       NVARCHAR(100)   NOT NULL,               -- שם מלא
    email           NVARCHAR(200)   NOT NULL UNIQUE,         -- אימייל — ייחודי לכל מורה
    username        NVARCHAR(50)    NOT NULL UNIQUE,         -- שם משתמש להתחברות
    password_hash   NVARCHAR(256)   NOT NULL,                -- bcrypt/SHA256 — לעולם לא טקסט גלוי
    phone           NVARCHAR(20),                            -- טלפון ליצירת קשר
    role            NVARCHAR(20)    NOT NULL
                        DEFAULT N'teacher'
                        CHECK (role IN (N'teacher', N'admin', N'principal')),
    is_active       BIT             NOT NULL DEFAULT 1,      -- 0 = מורה לא פעילה (לא נמחקת!)
    created_at      DATETIME        NOT NULL DEFAULT GETDATE(),
    last_login_at   DATETIME                                 -- תאריך כניסה אחרון - עדכון בכל כניסה
);
GO

-- ===================================================
-- 2. טבלת כיתות
--    מפתח: id
--    תלויות: אין (טבלת שורש)
-- ===================================================
CREATE TABLE Classes (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    class_name      NVARCHAR(50)    NOT NULL UNIQUE,         -- שם הכיתה: "ז'1", "י"ב2"
    grade_level     INT             NOT NULL
                        CHECK (grade_level BETWEEN 1 AND 12),-- שכבה 1–12
    track           NVARCHAR(50),                            -- מגמה: מדעית / הומניסטית / וכו'
    room_number     NVARCHAR(20),                            -- חדר קבוע של הכיתה
    student_count   INT             CHECK (student_count > 0),
    created_at      DATETIME        NOT NULL DEFAULT GETDATE(),
	academic_year   nvarchar(5)  NOT NULL 
);
GO



-- ===================================================
-- 4. טבלת מקצועות
--    מפתח: id
--    FK: category_id → SubjectCategories
--    ללא כפילות — שם הקטגוריה לא חוזר בכל שורה
-- ===================================================
CREATE TABLE Subjects (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    subject_name    NVARCHAR(100)   NOT NULL UNIQUE,
    subject_code    NVARCHAR(20) ,--יכיל קוד מקצוע אבא (למשל אנגלית עבור אנגלית מתוגבר)
    is_active       BIT             NOT NULL DEFAULT 1
);
GO


-- ===================================================
-- 5. טבלת שיעורי מערכת (לוח זמנים שבועי קבוע)
--    מפתח: id
--    FK: teacher_id → Teachers, class_id → Classes,
--        subject_id → Subjects
-- ===================================================
CREATE TABLE ScheduledLessons (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    teacher_id      INT             NOT NULL
                        REFERENCES Teachers(id),
    class_id        INT             NOT NULL
                        REFERENCES Classes(id),
    subject_id      INT             NOT NULL
                        REFERENCES Subjects(id),
    day_of_week     TINYINT         NOT NULL
                        CHECK (day_of_week BETWEEN 1 AND 7), -- 1=ראשון ... 7=שבת
    lesson_hour     TINYINT         NOT NULL
                        CHECK (lesson_hour BETWEEN 1 AND 12),-- שעה 1–12 ביום
    start_time      TIME            NOT NULL,                 -- שעת תחילה מדויקת: '08:00'
    end_time        TIME            NOT NULL,                 -- שעת סיום: '08:45'
    room_number     NVARCHAR(20),                            -- חדר (אם שונה מחדר הכיתה)
    is_active       BIT             NOT NULL DEFAULT 1,       -- 0 = שיעור בוטל זמנית
    valid_from      DATE            NOT NULL DEFAULT GETDATE(),-- תחילת תוקף המערכת
    valid_until     DATE,                                    -- סוף תוקף (NULL = עדיין פעיל)

    -- מניעת כפילות: אותה כיתה, אותו יום, אותה שעה
    CONSTRAINT uq_class_day_hour
        UNIQUE (class_id, day_of_week, lesson_hour, valid_from),
    -- מניעת כפילות: אותה מורה, אותו יום, אותה שעה
    CONSTRAINT uq_teacher_day_hour
        UNIQUE (teacher_id, day_of_week, lesson_hour, valid_from)
);
GO

-- ===================================================
-- 6. טבלת הקלטות
--    מפתח: id
--    FK: scheduled_lesson_id → ScheduledLessons
--    תיאור: כל שורה = הקלטה אחת שהועלתה + תוצאות עיבוד
-- ===================================================
CREATE TABLE Recordings (
    id                      INT IDENTITY(1,1) PRIMARY KEY,

    -- קישור לשיעור במערכת
    scheduled_lesson_id     INT             NOT NULL
                                REFERENCES ScheduledLessons(id),
    lesson_date             DATE            NOT NULL,        -- תאריך בפועל של השיעור

    -- קובץ השמע
    file_name_new               NVARCHAR(260)   ,        --  אופציונלי שם קובץ על הדיסק (GUID)
    file_path               NVARCHAR(500)   NOT NULL,        -- נתיב מלא לקובץ
    file_size_bytes         BIGINT          NOT NULL
                                CHECK (file_size_bytes > 0),
    original_file_name      NVARCHAR(260) ,                   -- שם הקובץ המקורי שהמורה העלתה
    format_file             NVARCHAR(10),                    -- wav / mp3 / ogg וכו'

    -- מצב עיבוד (תור)
    status_Processing                  TINYINT         NOT NULL DEFAULT 0
                                CHECK (status_Processing IN (0,1,2,3)), -- 0=Pending,1=Processing,2=Done,3=Failed
    status_Processing_error_message           NVARCHAR(MAX),                   -- הודעת שגיאה אם status_Processing=3

    -- תוצאות עיבוד כוללות (ממולאות אחרי status=2)
    duration_sec            FLOAT,                           -- משך ההקלטה
    positive_pct            FLOAT
                                CHECK (positive_pct BETWEEN 0 AND 100),  -- % חלונות חיוביים
    negative_pct            FLOAT
                                CHECK (negative_pct BETWEEN 0 AND 100),  -- % חלונות הפרעה
    noise_windows_count     INT,                             -- מספר חלונות רעש (לא נספרו)
    total_windows_count     INT,                             -- סה"כ חלונות
    grade                   NVARCHAR(20)
                                CHECK (grade IN (N'מצוין',N'טוב',N'בינוני',N'נמוך', NULL)),
    harm_threshold          FLOAT,                           -- סף harmonicity שחושב בכיול

    -- חותמות זמן
    uploaded_at             DATETIME        NOT NULL DEFAULT GETDATE(),
    processing_started_at   DATETIME,
    processing_finished_at  DATETIME,

    -- הערות
    teacher_notes           NVARCHAR(500)                    -- הערת מורה חופשית (אופציונלי)
);
GO

-- ===================================================
-- 7. טבלת פרקי זמן (חלונות ניתוח)
--    מפתח: id
--    FK: recording_id → Recordings
--    תיאור: שורה אחת לכל חלון של 3 שניות
-- ===================================================

CREATE TABLE TimeWindows (
    id                  INT IDENTITY(1,1) PRIMARY KEY,
    recording_id        INT             NOT NULL
                            REFERENCES Recordings(id) ON DELETE CASCADE,
    window_index        INT             NOT NULL,            -- מספר סידורי החלון (0,1,2...)

    -- תחום הזמן
    start_sec           FLOAT           NOT NULL,
    end_sec             FLOAT           NOT NULL,
    CHECK (end_sec > start_sec),

    -- [3] VAD
    speech_ratio        FLOAT           NOT NULL
                            CHECK (speech_ratio BETWEEN 0 AND 1), --אחוז דיבור
    has_speech          BIT             NOT NULL, --יש דיבור

    -- [4] RMS
    rms_value           FLOAT,                               -- ערך RMS גולמי
    rms_db              FLOAT,                               -- dB
    rms_level           NVARCHAR(20)
                            CHECK (rms_level IN (N'שקט',N'רגיל',N'רועש', NULL)),

    -- [5] סיווג אקוסטי
    audio_type          NVARCHAR(30),                        -- שקט/רעש_סביבתי/פטפטת_חלשה/דיבור_ברור

    -- [6] זיהוי דוברים
    speaker_type        NVARCHAR(30)
                            CHECK (speaker_type IN (N'דובר_יחיד',N'ריבוי_דוברים',N'רעש', NULL)),
    overlap_score       FLOAT
                            CHECK (overlap_score BETWEEN 0 AND 1),

    -- [7] ניתוח הקשרי HEBERT
    context_category    NVARCHAR(30)
                            CHECK (context_category IN
                                (N'למידה_פעילה',N'פתיחה_לדיון',N'הפרעה',
                                 N'דובר_יחיד',N'לא_ידוע', NULL)),
    context_confidence  FLOAT
                            CHECK (context_confidence BETWEEN 0 AND 1),
    transcribed_text    NVARCHAR(MAX),                       -- תמלול (אם בוצע)
    state_machine       NVARCHAR(20)
                            CHECK (state_machine IN (N'lesson',N'disruption', NULL)),

    -- [8] ציון קשב
    attention_label     NVARCHAR(20)
                            CHECK (attention_label IN (N'חיובי',N'הפרעה', NULL))
                                                             -- NULL = רעש (לא נספר)
);
GO


