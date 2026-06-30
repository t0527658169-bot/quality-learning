// ===================================================
// RecordingsController.cs — בקר API לניהול הקלטות
// ===================================================
// בקר זה הוא "שומר הסף" של המערכת:
//   1. מקבל קובץ שמע מהדפדפן (POST /upload)
//   2. שומר אותו לדיסק עם שם ייחודי
//   3. יוצר רשומה ב-DB עם סטטוס "ממתין"
//   4. דוחף את מזהה ההקלטה לתור → QueueProcessorService יאסוף ויישלח לפייתון
//
// נתיבי API:
//   POST   /api/recordings/upload          — העלאת הקלטה חדשה
//   GET    /api/recordings/{id}/status     — בדיקת מצב עיבוד
//   GET    /api/recordings/search          — חיפוש הקלטות לפי פרמטרים
//   POST   /api/recordings/{id}/reprocess  — שליחה מחדש לעיבוד
//   GET    /api/recordings/queue-status    — כמה הקלטות בתור
//   GET    /api/recordings/{id}/windows    — חלוני ניתוח מפורטים
//   GET    /api/recordings/lessons         — רשימת שיעורים לבחירה בטופס
// ===================================================

// ייבוא AppDbContext — גשר ל-SQL Server דרך Entity Framework
using AudioReceiver.Data;
// ייבוא המודל Recording — מבנה הנתונים של הקלטה בודדת
using AudioReceiver.Models;
// ייבוא IAudioQueueService — ממשק התור שאליו דוחפים הקלטות
using AudioReceiver.Services;
// ייבוא מחלקות ASP.NET Core לבניית API: ControllerBase, IActionResult, HttpGet/Post וכו'
using Microsoft.AspNetCore.Mvc;
// ייבוא Entity Framework Core — AsNoTracking, FirstOrDefaultAsync, SaveChangesAsync
using Microsoft.EntityFrameworkCore;

// הגדרת ה-namespace — מארגן את הקוד לפי שכבה (Controllers)
namespace AudioReceiver.Controllers;

// [ApiController] — מפעיל התנהגויות אוטומטיות:
//   - ולידציה אוטומטית של [FromForm]/[FromBody]
//   - תגובות 400 Bad Request אוטומטיות כשהמודל לא תקין
//   - Binding אוטומטי של פרמטרים
[ApiController]
// [Route("api/[controller]")] — [controller] = שם המחלקה ללא "Controller"
// → כל הנתיבים מתחילים ב-/api/recordings
[Route("api/[controller]")]
public class RecordingsController : ControllerBase
{
    // _db — גישה ל-SQL Server דרך Entity Framework (ORM)
    // AppDbContext מכיל את כל הטבלאות: Recordings, TimeWindows וכו'
    private readonly AppDbContext _db;

    // _queue — תור ההקלטות הממתינות לעיבוד (נמצא בזיכרון כ-ConcurrentQueue)
    // IAudioQueueService הוא ממשק (interface) — מאפשר בדיקות יחידה (unit tests)
    private readonly IAudioQueueService _queue;

    // _logger — מערכת לוגים מובנית של ASP.NET Core
    // כותב הודעות לקונסול + קבצי לוג
    private readonly ILogger<RecordingsController> _logger;

    // _config — גישה לקובץ appsettings.json ולמשתני סביבה
    // משמש לקריאת Storage:AudioPath ולהגדרות נוספות
    private readonly IConfiguration _config;

    // AudioStoragePath — נתיב התיקייה שבה נשמרים קבצי השמע
    // קורא מ-appsettings.json; אם לא מוגדר → fallback לנתיב קבוע
    // זהו Property (לא שדה) — מחושב בכל קריאה
    private string AudioStoragePath =>
        _config["Storage:AudioPath"] ?? "C:\\audio_files";

    // Constructor — נקרא ע"י ASP.NET Core כשמגיעה בקשה 
    // Dependency Injection: ASP.NET Core "יודע" שצריך ליצור את כל הפרמטרים
    // ומזריק אותם אוטומטית בהתאם לרישום ב-Program.cs
    public RecordingsController(AppDbContext db, IAudioQueueService queue,
        ILogger<RecordingsController> logger, IConfiguration config)
    {
        // שמירת ההזרקות בשדות פרטיים לשימוש בכל המתודות
        _db = db; _queue = queue; _logger = logger; _config = config;
    }

    // -------------------------------------------------------
    // GET /api/recordings/lessons
    // מחזיר רשימת שיעורים פעילים מה-DB — לתפריט הבחירה בטופס ההעלאה
    // [FromQuery] teacherId — אם מועבר, מסנן רק שיעורים של המורה הזו
    // -------------------------------------------------------
    [HttpGet("lessons")]
    public async Task<IActionResult> GetLessons([FromQuery] int? teacherId)
    {
        // פתיחת חיבור ישיר ל-SQL Server (במקום EF) לשאילתות קריאה מורכבות
        // GetDbConnection() מחזיר את חיבור ה-ADO.NET הבסיסי מתוך ה-DbContext
        var conn = _db.Database.GetDbConnection();
        // רשימה ריקה שתתמלא בתוצאות הקריאה
        var results = new List<object>();
        // פתיחת החיבור — Async כי ה-API צריך להיות לא חוסם (non-blocking)
        await conn.OpenAsync();
        try
        {
            // יצירת פקודת SQL — conn.CreateCommand() בטוח יותר משרשור מחרוזות (מונע SQL Injection)
            using var cmd = conn.CreateCommand();
            // שאילתה: JOIN של 4 טבלאות לקבל שם מורה, כיתה, מקצוע, וזמן השיעור
            // WHERE is_active = 1 — רק שיעורים פעילים (לא שיעורים שבוטלו)
            // @tid IS NULL OR — אם לא הועבר מסנן מורה, מחזיר הכל
            cmd.CommandText = @"
                SELECT sl.id,
                       t.full_name  AS teacher_name,
                       c.class_name,
                       s.subject_name,
                       sl.day_of_week,
                       sl.start_time,
                       sl.end_time,
                       t.id         AS teacher_id
                FROM   ScheduledLessons sl
                JOIN   Teachers t ON t.id = sl.teacher_id
                JOIN   Classes  c ON c.id = sl.class_id
                JOIN   Subjects s ON s.id = sl.subject_id
                WHERE  sl.is_active = 1
                  AND  (@tid IS NULL OR sl.teacher_id = @tid)
                ORDER  BY t.full_name, c.class_name, sl.day_of_week, sl.start_time";
            // הגדרת פרמטר @tid — אם teacherId הוא null, שולחים DBNull.Value (= NULL ב-SQL)
            // שימוש בפרמטרים (ולא שרשור) מונע SQL Injection (OWASP Top 10 #3)
            var pt = cmd.CreateParameter();
            pt.ParameterName = "@tid";
            pt.Value = teacherId.HasValue ? (object)teacherId.Value : DBNull.Value;
            cmd.Parameters.Add(pt);

            // הרצת הקריאה — Async כדי לא לחסום ת'שרד
            using var reader = await cmd.ExecuteReaderAsync();
            // מערך שמות ימי שבוע בעברית לפי אינדקס (1=ראשון, 2=שני וכו')
            var days = new[] { "", "ראשון","שני","שלישי","רביעי","חמישי","שישי","שבת" };
            // קריאת כל שורה מהתוצאה — reader.ReadAsync() מחזיר true כל עוד יש שורות
            while (await reader.ReadAsync())
            {
                // קריאת ערך יום השבוע (byte = מספר קטן 1-7)
                var day   = reader.GetByte(4);
                // המרת TimeSpan לפורמט שעות:דקות (hh:mm) לתצוגה
                var start = ((TimeSpan)reader.GetValue(5)).ToString(@"hh\:mm");
                var end   = ((TimeSpan)reader.GetValue(6)).ToString(@"hh\:mm");
                var teacher = reader.GetString(1);
                var cls     = reader.GetString(2);
                var subj    = reader.GetString(3);
                // בניית אובייקט אנונימי — יומר ל-JSON אוטומטית ע"י ASP.NET Core
                results.Add(new
                {
                    id          = reader.GetInt32(0),
                    teacherId   = reader.GetInt32(7),
                    teacherName = teacher,
                    className   = cls,
                    subjectName = subj,
                    dayOfWeek   = day,
                    startTime   = start,
                    endTime     = end,
                    // label — מחרוזת תצוגה מוכנה לשימוש ב-dropdown בדפדפן
                    label       = $"{teacher} | {cls} | {subj} | יום {days[day]} {start}–{end}"
                });
            }
        }
        // finally — מובטח שייסגר החיבור גם אם הייתה שגיאה
        // חשוב: אי-סגירת חיבורים גוction Pool Exhaustion
        finally { await conn.CloseAsync(); }

        // Ok(200) + JSON של הרשימה
        return Ok(results);
    }

    // -------------------------------------------------------
    // POST /api/recordings/upload
    // -------------------------------------------------------
    // זו הפונקציה המרכזית — מכאן מתחיל כל תהליך העיבוד:
    //
    //   דפדפן שולח multipart/form-data עם:
    //     - scheduledLessonId: מזהה השיעור שנבחר
    //     - lessonDate: תאריך השיעור
    //     - file: קובץ השמע (WAV/MP3 וכו')
    //     - teacherNotes: הערות (אופציונלי)
    //
    //   מה שקורה כאן:
    //     1. ולידציות (קובץ קיים? סיומת מותרת? שיעור קיים?)
    //     2. שמירת הקובץ לדיסק עם שם ייחודי (GUID)
    //     3. יצירת רשומה ב-DB (status=Pending)
    //     4. דחיפה לתור → QueueProcessorService מאסף ושולח לפייתון
    // -------------------------------------------------------
    [HttpPost("upload")]
    // הגבלת גודל בקשה ל-500MB — ASP.NET Core ברירת מחדל = 28MB בלבד
    // 500MB = מספיק לשיעור של שעתיים ב-WAV איכות גבוהה
    [RequestSizeLimit(500 * 1024 * 1024)]
    public async Task<IActionResult> Upload(
        [FromForm] int scheduledLessonId,    // מזהה השיעור שנבחר בטופס (ScheduledLessons.id)
        [FromForm] DateOnly lessonDate,       // תאריך השיעור — DateOnly = ללא שעה
        [FromForm] IFormFile file,            // הקובץ עצמו — IFormFile = stream לקריאה בזיכרון
        [FromForm] string? teacherNotes)      // ? = אופציונלי, יכול להיות null
    {
        // --- ולידציה 1: שיעור תקין נבחר ---
        // scheduledLessonId <= 0 קורה כשהדפדפן שולח 0 (לא נבחר שיעור)
        if (scheduledLessonId <= 0)
            return BadRequest(new { error = "יש לבחור שיעור מהרשימה" });

        // --- ולידציה 2: קובץ קיים ולא ריק ---
        // file?.Length == 0 = הקובץ הועלה אבל ריק (0 bytes)
        if (file is null || file.Length == 0)
            return BadRequest(new { error = "לא נבחר קובץ שמע" });

        // --- ולידציה 3: סיומת קובץ מותרת (OWASP: Unrestricted File Upload) ---
        // Path.GetExtension מחלץ ".wav" / ".mp3" וכו' מהשם המקורי
        // ToLowerInvariant() = ממיר לאותיות קטנות כדי שגם ".WAV" יתקבל
        var allowedExtensions = new[] { ".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm" };
        var extension = Path.GetExtension(file.FileName).ToLowerInvariant();
        if (!allowedExtensions.Contains(extension))
            return BadRequest(new { error = $"סוג קובץ לא נתמך: {extension}" });

        // --- ולידציה 4: השיעור קיים ופעיל ב-DB ---
        // מבוצעת בשאילתת COUNT — מהיר יותר מאשר לטעון אובייקט שלם
        var conn = _db.Database.GetDbConnection();
        await conn.OpenAsync();
        bool lessonExists;
        try
        {
            using var cmd = conn.CreateCommand();
            // COUNT(1) > 0 = השיעור קיים; is_active = 1 = השיעור לא בוטל
            cmd.CommandText = "SELECT COUNT(1) FROM ScheduledLessons WHERE id = @id AND is_active = 1";
            var p = cmd.CreateParameter();
            p.ParameterName = "@id"; p.Value = scheduledLessonId;
            cmd.Parameters.Add(p);
            // ExecuteScalarAsync = מחזיר ערך בודד (מספר השורות)
            lessonExists = Convert.ToInt32(await cmd.ExecuteScalarAsync()) > 0;
        }
        finally { await conn.CloseAsync(); }

        if (!lessonExists)
            return BadRequest(new { error = "השיעור שנבחר לא נמצא במערכת" });

        // --- שמירת הקובץ לדיסק ---
        // יצירת תיקייה אם לא קיימת — CreateDirectory בטוח לקרוא גם אם התיקייה קיימת
        Directory.CreateDirectory(AudioStoragePath);

        // Guid.NewGuid() = מזהה ייחודי גלובלי בן 36 תווים (e.g. "3f2504e0-4f89-...")
        // הוספת סיומת המקורית — מאפשרת Python לדעת את הפורמט
        // שם ייחודי מונע: (1) התנגשויות, (2) Path Traversal Attack
        var safeFileName = $"{Guid.NewGuid()}{extension}";

        // Path.Combine = בונה נתיב בצורה בטוחה בלי לשרשר / ידנית
        // example: "C:\audio_files\3f2504e0-4f89-11d3-9a0c-0305e82c3301.wav"
        var fullPath = Path.Combine(AudioStoragePath, safeFileName);

        // יצירת הקובץ וכתיבת תוכן הבקשה אליו
        // await using = מבטיח סגירת הStream גם אם תהיה שגיאה
        await using (var stream = System.IO.File.Create(fullPath))
            // CopyToAsync = מעתיק את תוכן הקובץ שהועלה לקובץ בדיסק ב-streaming (לא טוען לזיכרון)
            await file.CopyToAsync(stream);

        // --- יצירת רשומה ב-DB ---
        var recording = new Recording
        {
            ScheduledLessonId           = scheduledLessonId,   // איזה שיעור
            LessonDate                  = lessonDate,           // מתי (תאריך)
            FileNameNew                 = safeFileName,         // שם הקובץ החדש (GUID)
            FilePath                    = fullPath,             // נתיב מלא לדיסק
            FileSizeBytes               = file.Length,          // גודל בbytes לתצוגה
            OriginalFileName            = Path.GetFileName(file.FileName), // שם המקורי שהמורה העלתה
            FormatFile                  = extension.TrimStart('.'),        // "wav" / "mp3" (ללא נקודה)
            // ProcessingStatus.Pending = 0 — ממתין לתורו לעיבוד
            StatusProcessing            = ProcessingStatus.Pending,
            TeacherNotes                = teacherNotes?.Trim()  // ?.Trim() = null אם teacherNotes=null
        };

        // _db.Recordings.Add() — רושם את האובייקט ל-EF Core tracking (עדיין לא ב-DB)
        _db.Recordings.Add(recording);
        // SaveChangesAsync() — מריץ INSERT ב-DB; EF Core מאכלס recording.Id אוטומטית
        await _db.SaveChangesAsync();

        // --- דחיפה לתור ---
        // _queue הוא ConcurrentQueue<int> (בטוח לשימוש ממספר threads)
        // QueueProcessorService (עובד רקע) מאזין לתור ויאסוף את המזהה הזה
        // מרגע זה — Python יתחיל לעבד כשיגיע תורו
        _queue.Enqueue(recording.Id);

        // לוג ה-debug: מאפשר לעקוב אחר הזרימה בקונסול
        _logger.LogInformation("Recording {Id} enqueued — lesson {LessonId}, queue: {Q}",
            recording.Id, scheduledLessonId, _queue.PendingCount);

        // 202 Accepted = "קיבלתי, אבל לא עשיתי עדיין" — מתאים כי העיבוד אסינכרוני
        return Accepted(new
        {
            message        = "ההקלטה הועלתה! המערכת תנתח אותה בקרוב.",
            recordingId    = recording.Id,       // המזהה שהמשתמש יוכל לשאול עליו
            queuePosition  = _queue.PendingCount, // כמה לפניו בתור
            uploadedAt     = recording.UploadedAt
        });
    }

    // -------------------------------------------------------
    // GET /api/recordings/{id}/status
    // בדיקת מצב עיבוד הקלטה — הדפדפן קורא לזה כל כמה שניות (polling)
    // מחזיר: סטטוס, ציון, אחוז חיובי, הודעת שגיאה אם כשל
    // -------------------------------------------------------
    [HttpGet("{id:int}/status")]
    public async Task<IActionResult> GetStatus(int id)
    {
        // AsNoTracking() = EF Core לא "עוקב" אחר האובייקט (מהיר יותר לקריאה בלבד)
        // FirstOrDefaultAsync = מחזיר null אם לא נמצא (לא זורק שגיאה)
        var r = await _db.Recordings.AsNoTracking().FirstOrDefaultAsync(x => x.Id == id);
        // 404 אם ההקלטה לא קיימת
        if (r is null) return NotFound(new { error = $"הקלטה {id} לא נמצאה" });

        return Ok(new
        {
            id           = r.Id,
            // ToString() = שם ה-enum: "Pending" / "Processing" / "Done" / "Failed"
            status       = r.StatusProcessing.ToString(),
            // switch expression — תרגום ל-עברית לתצוגה בדפדפן
            statusHebrew = r.StatusProcessing switch
            {
                ProcessingStatus.Pending    => "ממתין בתור",    // עדיין לא התחיל
                ProcessingStatus.Processing => "בעיבוד...",      // Python מעבד עכשיו
                ProcessingStatus.Done       => "הושלם",          // יש תוצאות
                ProcessingStatus.Failed     => "נכשל",           // שגיאה בעיבוד
                _                           => "לא ידוע"         // מקרה קצה
            },
            grade                  = r.Grade,              // "מצוין" / "טוב" / "בינוני" / "נמוך"
            positivePct            = r.PositivePct,        // אחוז הזמן שהשיעור היה חיובי (0-100)
            durationSec            = r.DurationSec,        // משך ההקלטה בשניות
            errorMessage           = r.StatusProcessingErrorMessage, // הודעת שגיאה אם כשל
            uploadedAt             = r.UploadedAt,
            processingFinishedAt   = r.ProcessingFinishedAt
        });
    }

    // -------------------------------------------------------
    // GET /api/recordings/search
    // חיפוש הקלטות לפי פרמטרים — כל הפרמטרים אופציונליים
    // מחזיר עד 100 תוצאות (OFFSET 0 FETCH NEXT 100)
    // -------------------------------------------------------
    [HttpGet("search")]
    public async Task<IActionResult> Search(
        [FromQuery] int?      lessonId,     // מזהה שיעור ספציפי
        [FromQuery] DateOnly? date,         // תאריך ספציפי
        [FromQuery] int?      teacherId,    // מסנן לפי מורה
        [FromQuery] string?   className,    // מסנן לפי כיתה ("י'2" וכו')
        [FromQuery] string?   subjectName)  // מסנן לפי מקצוע ("מתמטיקה" וכו')
    {
        var conn = _db.Database.GetDbConnection();
        var results = new List<object>();
        await conn.OpenAsync();
        try
        {
            using var cmd = conn.CreateCommand();
            // שאילתה עם פרמטרים אופציונליים: @X IS NULL OR = אם לא הועבר פרמטר, לא מסנן
            // ORDER BY lesson_date DESC = הכי חדשות ראשונות
            // OFFSET 0 FETCH NEXT 100 = עמוד ראשון, עד 100 תוצאות (מניעת overhead)
            cmd.CommandText = @"
                SELECT r.id, r.lesson_date, r.original_file_name, r.format_file,
                       r.status_Processing, r.grade, r.positive_pct, r.duration_sec, r.uploaded_at,
                       t.full_name AS teacher_name, c.class_name, s.subject_name
                FROM   Recordings r
                JOIN   ScheduledLessons sl ON sl.id = r.scheduled_lesson_id
                JOIN   Teachers t ON t.id = sl.teacher_id
                JOIN   Classes  c ON c.id = sl.class_id
                JOIN   Subjects s ON s.id = sl.subject_id
                WHERE  (@lid  IS NULL OR r.scheduled_lesson_id = @lid)
                  AND  (@date IS NULL OR r.lesson_date = @date)
                  AND  (@tid  IS NULL OR sl.teacher_id = @tid)
                  AND  (@cls  IS NULL OR c.class_name  = @cls)
                  AND  (@sub  IS NULL OR s.subject_name = @sub)
                ORDER BY r.lesson_date DESC, r.uploaded_at DESC
                OFFSET 0 ROWS FETCH NEXT 100 ROWS ONLY";

            // הגדרת כל הפרמטרים — null → DBNull.Value (= NULL ב-SQL)
            var p1 = cmd.CreateParameter(); p1.ParameterName = "@lid";  p1.Value = lessonId.HasValue    ? (object)lessonId.Value                    : DBNull.Value; cmd.Parameters.Add(p1);
            var p2 = cmd.CreateParameter(); p2.ParameterName = "@date"; p2.Value = date.HasValue         ? (object)date.Value.ToString("yyyy-MM-dd") : DBNull.Value; cmd.Parameters.Add(p2);
            var p3 = cmd.CreateParameter(); p3.ParameterName = "@tid";  p3.Value = teacherId.HasValue    ? (object)teacherId.Value                   : DBNull.Value; cmd.Parameters.Add(p3);
            var p4 = cmd.CreateParameter(); p4.ParameterName = "@cls";  p4.Value = string.IsNullOrWhiteSpace(className)   ? (object)DBNull.Value : className;    cmd.Parameters.Add(p4);
            var p5 = cmd.CreateParameter(); p5.ParameterName = "@sub";  p5.Value = string.IsNullOrWhiteSpace(subjectName) ? (object)DBNull.Value : subjectName;  cmd.Parameters.Add(p5);

            using var reader = await cmd.ExecuteReaderAsync();
            // מיפוי מספרי סטטוס לשמות קריאים (0=Pending, 1=Processing, 2=Done, 3=Failed)
            var statusMap = new[] { "Pending", "Processing", "Done", "Failed" };
            while (await reader.ReadAsync())
            {
                // GetByte — כי status_Processing מוגדר כ-tinyint ב-SQL Server
                var si = reader.GetByte(4);
                results.Add(new
                {
                    id               = reader.GetInt32(0),
                    // ToString("yyyy-MM-dd") = פורמט ISO 8601 — אוניברסלי לכל דפדפן
                    lessonDate       = reader.GetFieldValue<DateOnly>(1).ToString("yyyy-MM-dd"),
                    // IsDBNull = בדיקת NULL לפני קריאה — מונע NullReferenceException
                    originalFileName = reader.IsDBNull(2) ? null : reader.GetString(2),
                    formatFile       = reader.IsDBNull(3) ? null : reader.GetString(3),
                    status           = si < statusMap.Length ? statusMap[si] : "Unknown",
                    grade            = reader.IsDBNull(5)  ? null              : reader.GetString(5),
                    positivePct      = reader.IsDBNull(6)  ? (double?)null     : reader.GetDouble(6),
                    durationSec      = reader.IsDBNull(7)  ? (double?)null     : reader.GetDouble(7),
                    uploadedAt       = reader.GetDateTime(8),
                    teacherName      = reader.GetString(9),
                    className        = reader.GetString(10),
                    subjectName      = reader.GetString(11)
                });
            }
        }
        finally { await conn.CloseAsync(); }

        return Ok(results);
    }

    // -------------------------------------------------------
    // POST /api/recordings/{id}/reprocess
    // שליחה מחדש של הקלטה לעיבוד — שימושי כשהעיבוד נכשל
    // מאפס את הסטטוס ל-Pending, מוחק TimeWindows ישנים, ודוחף לתור
    // -------------------------------------------------------
    [HttpPost("{id:int}/reprocess")]
    public async Task<IActionResult> Reprocess(int id)
    {
        // בדיקה שההקלטה קיימת לפני הניסיון לאפס
        var conn = _db.Database.GetDbConnection();
        await conn.OpenAsync();
        bool exists;
        try
        {
            using var cmd = conn.CreateCommand();
            cmd.CommandText = "SELECT COUNT(1) FROM Recordings WHERE id = @id";
            var p = cmd.CreateParameter(); p.ParameterName = "@id"; p.Value = id;
            cmd.Parameters.Add(p);
            exists = Convert.ToInt32(await cmd.ExecuteScalarAsync()) > 0;
        }
        finally { await conn.CloseAsync(); }

        if (!exists) return NotFound(new { error = $"הקלטה {id} לא נמצאה" });

        // איפוס סטטוס ל-Pending (0) ב-DB + מחיקת TimeWindows ישנים
        // DELETE TimeWindows — נדרש כדי ש-Python יכניס נתונים חדשים ריקים
        await conn.OpenAsync();
        try
        {
            using var cmd = conn.CreateCommand();
            cmd.CommandText = @"
                UPDATE Recordings
                SET    status_Processing = 0,
                       processing_started_at  = NULL,
                       processing_finished_at = NULL,
                       status_Processing_error_message = NULL
                WHERE  id = @id;
                DELETE FROM TimeWindows WHERE recording_id = @id2;";
            var p1 = cmd.CreateParameter(); p1.ParameterName = "@id";  p1.Value = id; cmd.Parameters.Add(p1);
            var p2 = cmd.CreateParameter(); p2.ParameterName = "@id2"; p2.Value = id; cmd.Parameters.Add(p2);
            await cmd.ExecuteNonQueryAsync();
        }
        finally { await conn.CloseAsync(); }

        // דחיפה לתור בזיכרון — QueueProcessorService יאסוף ויישלח ל-Python
        _queue.Enqueue(id);

        _logger.LogInformation("Recording {Id} re-queued for reprocessing.", id);
        return Accepted(new { message = $"הקלטה {id} הוכנסה מחדש לתור", queuePosition = _queue.PendingCount });
    }

    // -------------------------------------------------------
    // GET /api/recordings/queue-status
    // כמה הקלטות ממתינות בתור — לתצוגה בדאשבורד
    // -------------------------------------------------------
    [HttpGet("queue-status")]
    public IActionResult GetQueueStatus()
    {
        // PendingCount = גודל ה-ConcurrentQueue הנוכחי
        var count = _queue.PendingCount;
        return Ok(new
        {
            pending = count,
            // הודעה ידידותית לממשק המשתמש
            message = count == 0
                ? "התור ריק — כל ההקלטות עובדו"
                : $"בתור: {count} הקלטות ממתינות"
        });
    }

    // -------------------------------------------------------
    // GET /api/recordings/{id}/windows
    // כל חלוני ה-3 שניות של הקלטה — לתצוגה מפורטת אצל מנהלת
    // כל חלון = שורה אחת מטבלת TimeWindows ב-DB
    // -------------------------------------------------------
    [HttpGet("{id:int}/windows")]
    public async Task<IActionResult> GetWindows(int id)
    {
        var conn = _db.Database.GetDbConnection();
        var results = new List<object>();
        await conn.OpenAsync();
        try
        {
            using var cmd = conn.CreateCommand();
            // שאילתה: כל העמודות של TimeWindows לפי recording_id
            // ORDER BY window_index = בסדר כרונולוגי (חלון 0, 1, 2...)
            cmd.CommandText = @"
                SELECT window_index, start_sec, end_sec,
                       speech_ratio, has_speech,
                       rms_value, rms_db, rms_level,
                       audio_type, speaker_type, overlap_score,
                       context_category, context_confidence,
                       transcribed_text, state_machine, attention_label
                FROM   TimeWindows
                WHERE  recording_id = @id
                ORDER  BY window_index";
            var p = cmd.CreateParameter(); p.ParameterName = "@id"; p.Value = id;
            cmd.Parameters.Add(p);

            using var reader = await cmd.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                // בניית אובייקט עבור כל חלון — כולל ברירות מחדל null לשדות אופציונליים
                results.Add(new
                {
                    windowIndex        = reader.GetInt32(0),    // מספר החלון (0, 1, 2...)
                    startSec           = reader.GetDouble(1),   // שנייה התחלה (0.0, 3.0, 6.0...)
                    endSec             = reader.GetDouble(2),   // שנייה סיום (3.0, 6.0, 9.0...)
                    speechRatio        = reader.GetDouble(3),   // 0.0-1.0 כמה מהחלון היה דיבור
                    hasSpeech          = reader.GetBoolean(4),  // האם בכלל זוהה דיבור
                    rmsValue           = reader.IsDBNull(5)  ? (double?)null : reader.GetDouble(5),   // עוצמת קול מספרית
                    rmsDb              = reader.IsDBNull(6)  ? (double?)null : reader.GetDouble(6),   // עוצמה בדציבלים
                    rmsLevel           = reader.IsDBNull(7)  ? null : reader.GetString(7),            // "שקט"/"רגיל"/"רועש"
                    audioType          = reader.IsDBNull(8)  ? null : reader.GetString(8),            // "שקט"/"פטפטת_חלשה"/"דיבור_ברור"
                    speakerType        = reader.IsDBNull(9)  ? null : reader.GetString(9),            // "דובר_יחיד"/"ריבוי_דוברים"/"רעש"
                    overlapScore       = reader.IsDBNull(10) ? (double?)null : reader.GetDouble(10),  // ציון חפיפה 0-1
                    contextCategory    = reader.IsDBNull(11) ? null : reader.GetString(11),           // "למידה_פעילה"/"הפרעה"/"דובר_יחיד"
                    contextConfidence  = reader.IsDBNull(12) ? (double?)null : reader.GetDouble(12),  // רמת ביטחון
                    transcribedText    = reader.IsDBNull(13) ? null : reader.GetString(13),           // טקסט מזוהה (ASR)
                    stateMachine       = reader.IsDBNull(14) ? null : reader.GetString(14),           // מצב מכונת המצבים
                    attentionLabel     = reader.IsDBNull(15) ? null : reader.GetString(15)            // "חיובי"/"הפרעה"/null
                });
            }
        }
        finally { await conn.CloseAsync(); }

        // 404 אם אין חלונות — כנראה ההקלטה עדיין לא עובדה
        if (!results.Any()) return NotFound(new { error = $"אין חלוני זמן להקלטה {id}" });
        return Ok(results);
    }
}
