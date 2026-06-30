// ===================================================
// ManageController.cs — ניהול מורות ושיעורים (מנהלת בלבד)
// ===================================================
// מטפל בפעולות ניהוליות שרק מנהלת יכולה לבצע:
//   POST /api/manage/teacher — הוספת מורה חדשה למערכת
//   POST /api/manage/lesson  — הוספת שיעור חדש ללוח זמנים
//   GET  /api/manage/classes — רשימת כל הכיתות (לבחירה בטופס)
//   GET  /api/manage/subjects — רשימת כל המקצועות (לבחירה בטופס)
//
// אבטחה:
//   - כל ה-endpoints הללו אמורים להיות מוגנים ב-role check
//   - הפרמטרים מוגנים ב-parameterized SQL — מניעת SQL Injection
//   - סיסמה נשמרת כ-SHA256 hash בלבד — לעולם לא טקסט גלוי
// ===================================================

// גישה ל-DB דרך EF Core (AppDbContext)
using AudioReceiver.Data;
// ControllerBase ו-IActionResult
using Microsoft.AspNetCore.Mvc;
// GetDbConnection() — גישה ל-ADO.NET הגולמי לשאילתות מהירות
using Microsoft.EntityFrameworkCore;
// SHA256 — הצפנת הסיסמה
using System.Security.Cryptography;
// UTF8 — קידוד מחרוזות ל-bytes
using System.Text;

namespace AudioReceiver.Controllers;

// [ApiController] — ולידציה אוטומטית, JSON responses, ModelState binding
[ApiController]
// [Route] — כל הנתיבים כאן מתחילים ב-api/manage
[Route("api/manage")]
public class ManageController : ControllerBase
{
    // _db — גישה ל-Entity Framework Core, מוזרק אוטומטית ע"י ASP.NET Core DI
    private readonly AppDbContext _db;

    // Constructor Injection — ASP.NET Core מזריק את AppDbContext
    public ManageController(AppDbContext db) => _db = db;

    // -------------------------------------------------------
    // GET /api/manage/classes
    // מחזיר רשימת כל הכיתות הפעילות לבחירה בטופס הוספת שיעור
    // -------------------------------------------------------
    [HttpGet("classes")]
    public async Task<IActionResult> GetClasses()
    {
        // קבלת חיבור SQL גולמי מ-EF Core
        var conn = _db.Database.GetDbConnection();
        var results = new List<object>();
        await conn.OpenAsync();
        try
        {
            using var cmd = conn.CreateCommand();
            // שאילתה: כל הכיתות, מסודרות לפי שם
            cmd.CommandText = "SELECT id, class_name, grade_level FROM Classes ORDER BY grade_level, class_name";
            using var reader = await cmd.ExecuteReaderAsync();
            while (await reader.ReadAsync())
                results.Add(new { id = reader.GetInt32(0), name = reader.GetString(1), grade = reader.GetInt32(2) });
        }
        finally { await conn.CloseAsync(); }
        return Ok(results);
    }

    // -------------------------------------------------------
    // GET /api/manage/subjects
    // מחזיר רשימת כל המקצועות הפעילים לבחירה בטופס הוספת שיעור
    // -------------------------------------------------------
    [HttpGet("subjects")]
    public async Task<IActionResult> GetSubjects()
    {
        var conn = _db.Database.GetDbConnection();
        var results = new List<object>();
        await conn.OpenAsync();
        try
        {
            using var cmd = conn.CreateCommand();
            // שאילתה: מקצועות פעילים בלבד (is_active=1), מסודרים לפי שם
            cmd.CommandText = "SELECT id, subject_name FROM Subjects WHERE is_active = 1 ORDER BY subject_name";
            using var reader = await cmd.ExecuteReaderAsync();
            while (await reader.ReadAsync())
                results.Add(new { id = reader.GetInt32(0), name = reader.GetString(1) });
        }
        finally { await conn.CloseAsync(); }
        return Ok(results);
    }

    // -------------------------------------------------------
    // GET /api/manage/teachers
    // מחזיר רשימת כל המורות לבחירה בטופס הוספת שיעור
    // -------------------------------------------------------
    [HttpGet("teachers")]
    public async Task<IActionResult> GetTeachers()
    {
        var conn = _db.Database.GetDbConnection();
        var results = new List<object>();
        await conn.OpenAsync();
        try
        {
            using var cmd = conn.CreateCommand();
            // רק מורות פעילות (is_active=1)
            cmd.CommandText = "SELECT id, full_name FROM Teachers WHERE is_active = 1 ORDER BY full_name";
            using var reader = await cmd.ExecuteReaderAsync();
            while (await reader.ReadAsync())
                results.Add(new { id = reader.GetInt32(0), name = reader.GetString(1) });
        }
        finally { await conn.CloseAsync(); }
        return Ok(results);
    }

    // -------------------------------------------------------
    // POST /api/manage/teacher
    // הוספת מורה חדשה לטבלת Teachers
    //
    // שדות נדרשים (multipart form או JSON):
    //   fullName    — שם מלא
    //   email       — אימייל (חייב להיות ייחודי)
    //   username    — שם משתמש להתחברות (חייב להיות ייחודי)
    //   password    — סיסמה גולמית (תיהפך ל-SHA256)
    //   phone       — טלפון (אופציונלי)
    // -------------------------------------------------------
    [HttpPost("teacher")]
    public async Task<IActionResult> AddTeacher([FromBody] AddTeacherRequest req)
    {
        // ולידציה — שדות חובה
        if (string.IsNullOrWhiteSpace(req.FullName) ||
            string.IsNullOrWhiteSpace(req.Email) ||
            string.IsNullOrWhiteSpace(req.Username) ||
            string.IsNullOrWhiteSpace(req.Password))
            return BadRequest(new { error = "שם מלא, אימייל, שם משתמש וסיסמה הם שדות חובה" });

        // ולידציה בסיסית של פורמט אימייל — בדיקה שיש @ ונקודה
        if (!req.Email.Contains('@') || !req.Email.Contains('.'))
            return BadRequest(new { error = "פורמט האימייל אינו תקין" });

        // הצפנת הסיסמה ל-SHA256 hex lowercase
        // לעולם לא שומרים סיסמה כטקסט גלוי במסד נתונים!
        var hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(req.Password))).ToLower();

        var conn = _db.Database.GetDbConnection();
        await conn.OpenAsync();
        try
        {
            using var cmd = conn.CreateCommand();
            // INSERT עם בדיקת ייחודיות — email ו-username הם UNIQUE בטבלה
            // אם יש כפילות, SQL Server יזרוק שגיאה שנתפוס ב-catch
            cmd.CommandText = @"
                INSERT INTO Teachers (full_name, email, username, password_hash, phone, role, is_active, created_at)
                VALUES (@name, @email, @username, @hash, @phone, 'teacher', 1, GETDATE());
                SELECT SCOPE_IDENTITY();";
            // SCOPE_IDENTITY() מחזיר את ה-id שנוצר ע"י AUTO INCREMENT

            // הוספת פרמטרים — מגינים על SQL Injection
            AddParam(cmd, "@name",     req.FullName.Trim());
            AddParam(cmd, "@email",    req.Email.Trim().ToLower());
            AddParam(cmd, "@username", req.Username.Trim());
            AddParam(cmd, "@hash",     hash);
            AddParam(cmd, "@phone",    string.IsNullOrWhiteSpace(req.Phone) ? DBNull.Value : (object)req.Phone.Trim());

            // ExecuteScalarAsync — מחזיר ערך בודד (ה-id החדש)
            var newId = Convert.ToInt32(await cmd.ExecuteScalarAsync());
            return Ok(new { id = newId, message = $"המורה {req.FullName} נוספה בהצלחה" });
        }
        catch (Exception ex) when (ex.Message.Contains("UNIQUE") || ex.Message.Contains("duplicate"))
        {
            // SQL Server זורק שגיאת UNIQUE constraint אם אימייל או שם משתמש כבר קיים
            return Conflict(new { error = "אימייל או שם משתמש כבר קיים במערכת" });
        }
        finally { await conn.CloseAsync(); }
    }

    // -------------------------------------------------------
    // POST /api/manage/lesson
    // הוספת שיעור חדש ללוח הזמנים (ScheduledLessons)
    //
    // שדות נדרשים:
    //   teacherId   — id המורה (חייב להיות קיים בטבלת Teachers)
    //   classId     — id הכיתה (חייב להיות קיים בטבלת Classes)
    //   subjectId   — id המקצוע (חייב להיות קיים בטבלת Subjects)
    //   dayOfWeek   — יום בשבוע (1=ראשון ... 6=שישי)
    //   lessonHour  — מספר שעת שיעור ביום (1–12)
    //   startTime   — שעת התחלה (HH:MM)
    //   endTime     — שעת סיום (HH:MM)
    //   roomNumber  — חדר (אופציונלי)
    // -------------------------------------------------------
    [HttpPost("lesson")]
    public async Task<IActionResult> AddLesson([FromBody] AddLessonRequest req)
    {
        // ולידציה — בדיקת שדות חובה
        if (req.TeacherId <= 0 || req.ClassId <= 0 || req.SubjectId <= 0)
            return BadRequest(new { error = "יש לבחור מורה, כיתה ומקצוע" });
        if (req.DayOfWeek < 1 || req.DayOfWeek > 6)
            return BadRequest(new { error = "יום בשבוע חייב להיות בין 1 (ראשון) ל-6 (שישי)" });
        if (req.LessonHour < 1 || req.LessonHour > 12)
            return BadRequest(new { error = "שעת שיעור חייבת להיות בין 1 ל-12" });
        if (string.IsNullOrWhiteSpace(req.StartTime) || string.IsNullOrWhiteSpace(req.EndTime))
            return BadRequest(new { error = "שעת התחלה וסיום הן שדות חובה" });

        var conn = _db.Database.GetDbConnection();
        await conn.OpenAsync();
        try
        {
            using var cmd = conn.CreateCommand();
            // INSERT לטבלת ScheduledLessons
            // valid_from = היום — השיעור תקף מהיום
            // valid_until = NULL — אין תאריך סיום (שיעור קבוע)
            cmd.CommandText = @"
                INSERT INTO ScheduledLessons
                    (teacher_id, class_id, subject_id, day_of_week, lesson_hour,
                     start_time, end_time, room_number, is_active, valid_from)
                VALUES
                    (@tid, @cid, @sid, @dow, @hour,
                     @start, @end, @room, 1, CAST(GETDATE() AS DATE));
                SELECT SCOPE_IDENTITY();";

            // פרמטרים מוגנים
            AddParam(cmd, "@tid",   req.TeacherId);
            AddParam(cmd, "@cid",   req.ClassId);
            AddParam(cmd, "@sid",   req.SubjectId);
            AddParam(cmd, "@dow",   req.DayOfWeek);
            AddParam(cmd, "@hour",  req.LessonHour);
            AddParam(cmd, "@start", req.StartTime);
            AddParam(cmd, "@end",   req.EndTime);
            AddParam(cmd, "@room",  string.IsNullOrWhiteSpace(req.RoomNumber) ? DBNull.Value : (object)req.RoomNumber.Trim());

            var newId = Convert.ToInt32(await cmd.ExecuteScalarAsync());
            return Ok(new { id = newId, message = "השיעור נוסף בהצלחה ללוח הזמנים" });
        }
        catch (Exception ex) when (ex.Message.Contains("UNIQUE") || ex.Message.Contains("duplicate") || ex.Message.Contains("uq_"))
        {
            // SQL Server יזרוק שגיאת UNIQUE אם:
            // - אותה כיתה, אותו יום, אותה שעה (uq_class_day_hour)
            // - אותה מורה, אותו יום, אותה שעה (uq_teacher_day_hour)
            return Conflict(new { error = "כבר קיים שיעור באותו יום/שעה לכיתה זו או למורה זו" });
        }
        finally { await conn.CloseAsync(); }
    }

    // -------------------------------------------------------
    // פונקציית עזר פנימית — יצירת פרמטר SQL מוגן
    // מונעת SQL Injection ע"י שימוש ב-parameterized queries
    // -------------------------------------------------------
    private static void AddParam(System.Data.Common.DbCommand cmd, string name, object value)
    {
        var p = cmd.CreateParameter();
        p.ParameterName = name;
        p.Value = value;
        cmd.Parameters.Add(p);
    }
}

// -------------------------------------------------------
// AddTeacherRequest — מבנה הנתונים לבקשת הוספת מורה
// record = מחלקה immutable קומפקטית (C# 9+)
// -------------------------------------------------------
public record AddTeacherRequest(
    string FullName,
    string Email,
    string Username,
    string Password,
    string? Phone
);

// -------------------------------------------------------
// AddLessonRequest — מבנה הנתונים לבקשת הוספת שיעור
// -------------------------------------------------------
public record AddLessonRequest(
    int TeacherId,     // id המורה מטבלת Teachers
    int ClassId,       // id הכיתה מטבלת Classes
    int SubjectId,     // id המקצוע מטבלת Subjects
    int DayOfWeek,     // 1=ראשון, 2=שני, ... 6=שישי
    int LessonHour,    // 1–12 (מספר שעת שיעור ביום)
    string StartTime,  // פורמט HH:MM, לדוגמה "08:00"
    string EndTime,    // פורמט HH:MM, לדוגמה "08:45"
    string? RoomNumber // חדר (אופציונלי)
);
