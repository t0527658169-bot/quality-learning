// ===================================================
// QueueProcessorService.cs — עובד הרקע של עיבוד ההקלטות
// ===================================================

using AudioReceiver.Data;
using AudioReceiver.Models;
using Microsoft.EntityFrameworkCore;
using System.Net.Http.Headers;
using System.Text.Json.Serialization;

namespace AudioReceiver.Services;

// QueueProcessorService יורש מ-BackgroundService
// BackgroundService = חוזה שמחייב לממש ExecuteAsync()
// ASP.NET Core מפעיל אותו בthread נפרד ברקע
public class QueueProcessorService : BackgroundService
{
    // _queue — התור המשותף (ConcurrentQueue) שמחזיק מזהי הקלטות ממתינות
    // RecordingsController דוחף כאן, QueueProcessorService מושך מכאן
    private readonly IAudioQueueService _queue;

    // _scopeFactory — יוצר Scope חדש לכל הקלטה (נדרש כי DbContext הוא Scoped)
    // לא ניתן להשתמש ב-DbContext ישירות ב-Singleton — לכן Scope factory
    private readonly IServiceScopeFactory _scopeFactory;

    // _httpClientFactory — יוצר HttpClient מוגדר (שם "python") לשליחה ל-Python
    // IHttpClientFactory עדיף על new HttpClient() — מנהל connection pooling
    private readonly IHttpClientFactory _httpClientFactory;

    // _logger — לוגים לקונסול ולקבצי לוג
    private readonly ILogger<QueueProcessorService> _logger;

    // _config — גישה ל-appsettings.json ומשתני סביבה
    private readonly IConfiguration _config;

    // EmptyQueuePollIntervalMs — כשהתור ריק, מחכים 2 שניות לפני בדיקה חוזרת
    // מניעת busy-wait (לולאה שמכלה CPU בלי לעשות כלום)
    private const int EmptyQueuePollIntervalMs = 2000;

    // Constructor — ASP.NET Core מזריק את כל השירותים אוטומטית
    public QueueProcessorService(
        IAudioQueueService queue,
        IServiceScopeFactory scopeFactory,
        IHttpClientFactory httpClientFactory,
        ILogger<QueueProcessorService> logger,
        IConfiguration config)
    {
        // שמירת ההזרקות בשדות פרטיים
        _queue = queue;
        _scopeFactory = scopeFactory;
        _httpClientFactory = httpClientFactory;
        _logger = logger;
        _config = config;
    }

    // -------------------------------------------------------
    // ExecuteAsync — הפונקציה הראשית שמופעלת בthread נפרד
    // מופעלת אוטומטית ע"י ASP.NET Core כשהשרת עולה
    // -------------------------------------------------------
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Queue Processor started — loading pending recordings...");

        // שלב 1: בסטארטאפ — טעינת כל ההקלטות הממתינות לתור
        // זה חשוב למקרה שהשרת קרס באמצע עיבוד — ההקלטות לא יאבדו
        await LoadPendingRecordingsAsync(stoppingToken);
        _logger.LogInformation("Queue Processor ready — waiting for recordings...");

        // שלב 2: לולאה אינסופית — רצה עד שהשרת עוצר
        while (!stoppingToken.IsCancellationRequested)
        {
            // TryDequeue — מנסה לשלוף מזהה מהתור (לא חוסם — מחזיר false אם ריק)
            if (!_queue.TryDequeue(out int recordingId))
            {
                // התור ריק — מחכים 2 שניות ומנסים שוב
                // Task.Delay Async = לא חוסם thread 
                await Task.Delay(EmptyQueuePollIntervalMs, stoppingToken);
            }
            else
            {
                // מצאנו הקלטה ממתינה — מעבדים אותה
                // עוטפים ב-try/catch כדי שאם ProcessRecordingAsync זורק Exception
                // הלולאה תמשיך לרוץ (השרת לא יקרוס)
                try
                {
                    await ProcessRecordingAsync(recordingId, stoppingToken);
                }
                catch (Exception ex)
                {
                    // כל Exception שלא נתפס בתוך ProcessRecordingAsync מגיע לכאן
                    // ← לוגים אבל ממשיכים — השרת ישאר פעיל!
                    _logger.LogError(ex, "Unhandled exception processing recording {Id} — skipping and continuing.", recordingId);
                }
            }
        }
    }

    // -------------------------------------------------------
    // LoadPendingRecordingsAsync — טעינת הקלטות ממתינות בסטארטאפ
    // נקרא פעם אחת בלבד כשהשרת עולה
    // -------------------------------------------------------
    /// <summary>
    /// בעלייה של השרת: מחזיר Processing→Pending וטוען הכל לתור לפי סדר העלאה.
    /// </summary>
    private async Task LoadPendingRecordingsAsync(CancellationToken stoppingToken)
    {
        // יצירת Scope חדש — נדרש כי AppDbContext הוא Scoped (לא Singleton)
        // כלומר: יש ליצור DbContext חדש לכל "פעולה" ולסגור אחר כך
        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        // שחזור הקלטות "תקועות" — היו בסטטוס Processing כשהשרת קרס
        // אם status=Processing ואין תוצאות → השרת קרס באמצע → מחזירים ל-Pending
        var stuck = await db.Recordings
            .Where(r => r.StatusProcessing == ProcessingStatus.Processing)
            .ToListAsync(stoppingToken);
        // מחזירים כל תקועה ל-Pending כדי שתעובד מחדש
        foreach (var rec in stuck)
            rec.StatusProcessing = ProcessingStatus.Pending;
        // שמירה רק אם יש שינויים (מניעת SaveChanges מיותר)
        if (stuck.Count > 0)
            await db.SaveChangesAsync(stoppingToken);

        // טעינת כל הקלטות ה-Pending לתור לפי סדר העלאה (ראשון נכנס, ראשון יוצא)
        var pending = await db.Recordings
            .Where(r => r.StatusProcessing == ProcessingStatus.Pending)
            .OrderBy(r => r.UploadedAt)    // הכי ישנה תעובד ראשונה
            .Select(r => r.Id)             // Select(r => r.Id) = לוקח רק את המזהה (לא את כל האובייקט)
            .ToListAsync(stoppingToken);

        // דחיפת כל המזהים לתור — QueueProcessorService יאסוף בלולאה
        foreach (var id in pending)
            _queue.Enqueue(id);

        _logger.LogInformation(
            "Loaded {Count} pending recordings into queue on startup (including {Stuck} recovered).",
            pending.Count, stuck.Count);
    }

    // -------------------------------------------------------
    // ProcessRecordingAsync — עיבוד הקלטה בודדת מתחילה לסוף
    // שלבים: מצא בDB → סמן Processing → שלח ל-Python → שמור תוצאות → סמן Done/Failed
    // -------------------------------------------------------
    private async Task ProcessRecordingAsync(int recordingId, CancellationToken stoppingToken)
    {
        // Scope חדש לכל הקלטה — מבטיח שה-DbContext נסגר כשהעיבוד נגמר
        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        // שליפת ההקלטה מה-DB לפי מזהה
        var recording = await db.Recordings.FindAsync(new object[] { recordingId }, stoppingToken);
        if (recording is null)
        {
            // ייתכן שנמחקה בינתיים — מדלגים
            _logger.LogWarning("Recording {Id} not found — skipping.", recordingId);
            return;
        }

        _logger.LogInformation("Processing recording {Id} — lesson {LessonId} / {Date}",
            recording.Id, recording.ScheduledLessonId, recording.LessonDate);

        // סימון ב-DB שהעיבוד התחיל (Processing) + שמירת זמן התחלה
        // חשוב: אם השרת יקרוס עכשיו, LoadPendingRecordingsAsync ישחזר ל-Pending
        recording.StatusProcessing = ProcessingStatus.Processing;
        recording.ProcessingStartedAt = DateTime.UtcNow;
        try
        {
            await db.SaveChangesAsync(stoppingToken);
        }
        catch (DbUpdateConcurrencyException)
        {
            // ההקלטה נמחקה בין FindAsync לבין SaveChanges — מדלגים 
            _logger.LogWarning("Recording {Id} deleted before processing started — skipping.", recordingId);
            return;
        }

        try
        {
            // שליחת קובץ השמע ל-Python ולקבלת תוצאות
            var result = await SendToPythonAsync(recording, stoppingToken);

            // -------------------------------------------------------
            // טעינת ה-recording מחדש מה-DB לפני שמירת התוצאות
            // -------------------------------------------------------
            // הסיבה: Python עיבד 90+ דקות. ב-EF Core, ה-entity שנטען בתחילת
            // הפונקציה יכול להפוך ל-"stale" (לא מסונכרן עם DB).
            // אם בינתיים מישהו מחק את ההקלטה → UPDATE ל-0 שורות → DbUpdateConcurrencyException
            // הפתרון: טוענים מחדש, ואם לא קיים — מדלגים בשקט (הקלטה נמחקה)
            // db.Entry(recording).State = EntityState.Detached מנתק את ה-entity הישן
            db.Entry(recording).State = EntityState.Detached;
            var freshRecording = await db.Recordings.FindAsync(new object[] { recordingId }, stoppingToken);
            if (freshRecording is null)
            {
                _logger.LogWarning("Recording {Id} was deleted during processing — results discarded.", recordingId);
                return;
            }
            recording = freshRecording;

            // שמירת כל תוצאות ה-AI מ-Python לשדות ההקלטה
            recording.StatusProcessing      = ProcessingStatus.Done;         // עיבוד הצליח
            recording.DurationSec           = result.DurationSec;            // אורך ההקלטה בשניות
            recording.PositivePct           = result.PositivePct;            // אחוז חלונות "חיוביים" (למידה פעילה)
            recording.NegativePct           = result.NegativePct;            // אחוז חלונות "שליליים" (הפרעות)
            recording.NoiseWindowsCount     = result.NoiseCount;             // מספר חלונות רעש טהור
            recording.TotalWindowsCount     = result.TotalWindows;           // סה"כ חלונות (אורך / 3 שניות)
            recording.Grade                 = result.Grade;                  // ציון מילולי: "מצוין"/"טוב"/"גרוע"
            recording.ProcessingFinishedAt  = DateTime.UtcNow;              // זמן סיום עיבוד

            // שמירת חלוני הזמן (TimeWindows) לטבלה — כל 3 שניות = שורה אחת
            if (result.Windows is { Count: > 0 })
            {
                // מחיקת חלונות ישנים אם הוחזר לעיבוד מחדש (Reprocess)
                // מונע שכפול שורות בטבלת TimeWindows
                var existing = db.TimeWindows.Where(w => w.RecordingId == recording.Id);
                db.TimeWindows.RemoveRange(existing);

                // המרת תוצאות Python לאובייקטי TimeWindow ל-EF Core
                // idx = מספר החלון (0, 1, 2, ...) — window_index
                var windows = result.Windows.Select((w, idx) => new AudioReceiver.Models.TimeWindow
                {
                    RecordingId       = recording.Id,      // קישור לטבלת Recordings
                    WindowIndex       = idx,               // מספר סדרי של החלון
                    StartSec          = w.StartSec,        // שנייה התחלה (0, 3, 6, ...)
                    EndSec            = w.EndSec,          // שנייה סיום (3, 6, 9, ...)
                    SpeechRatio       = w.SpeechRatio,     // אחוז הדיבור בחלון (0.0–1.0)
                    HasSpeech         = w.HasSpeech,       // האם זוהה דיבור בכלל
                    RmsValue          = w.Rms,             // עוצמת קול (RMS) מספרית
                    RmsDb             = w.RmsDb,           // עוצמה בדציבלים
                    RmsLevel          = w.RmsLevel,        // "שקט"/"רגיל"/"רועש"
                    AudioType         = w.AudioType,       // "שקט"/"פטפטת_חלשה"/"דיבור_ברור"/"רעש_סביבתי"
                    SpeakerType       = w.SpeakerType,     // "דובר_יחיד"/"ריבוי_דוברים"/"רעש"
                    OverlapScore      = w.OverlapScore,    // ציון חפיפת דוברים (0.0–1.0)
                    ContextCategory   = w.ContextCategory,   // "למידה_פעילה"/"הפרעה"/"דובר_יחיד"
                    ContextConfidence = w.ContextConfidence, // רמת ביטחון ה-context (0.0–1.0)
                    StateMachine      = w.StateMachine,      // מצב מכונת המצבים: None/lesson/disruption
                    TranscribedText   = w.TranscribedText,   // טקסט שהומר מדיבור (ASR) — אם יש
                    AttentionLabel    = w.AttentionLabel     // "חיובי"/"הפרעה" — ציון סופי
                }).ToList();

                // הוספה מסיבית ל-DB (AddRange = INSERT אחד גדול במקום הרבה קטנים)
                await db.TimeWindows.AddRangeAsync(windows, stoppingToken);
            }

            _logger.LogInformation("Recording {Id} done — grade: {Grade}, score: {Score}%, windows: {W}",
                recording.Id, recording.Grade, recording.PositivePct, result.Windows?.Count ?? 0);
        }
        catch (Exception ex)
        {
            // שגיאה בשליחה ל-Python או בפענוח התשובה — סימון כ-Failed
            // טוענים מחדש כי ה-entity עשוי להיות stale לאחר זמן עיבוד ארוך
            db.Entry(recording).State = EntityState.Detached;
            var failedRec = await db.Recordings.FindAsync(new object[] { recordingId }, stoppingToken);
            if (failedRec is null)
            {
                _logger.LogWarning("Recording {Id} deleted during failed processing — ignoring.", recordingId);
                return;
            }
            recording = failedRec;
            recording.StatusProcessing               = ProcessingStatus.Failed;
            recording.StatusProcessingErrorMessage   = ex.Message;          // שמירת הודעת השגיאה לדיבוג
            recording.ProcessingFinishedAt           = DateTime.UtcNow;
            _logger.LogError(ex, "Recording {Id} failed.", recording.Id);
        }

        // שמירת כל השינויים ל-DB — סטטוס + תוצאות + חלונות
        await db.SaveChangesAsync(stoppingToken);
    }

   
    private async Task<PythonResult> SendToPythonAsync(Recording recording, CancellationToken ct)
    {
        // קבלת HttpClient מוגדר מראש עם BaseAddress של Python (http://localhost:5000)
        // "python" = שם הלקוח שהוגדר ב-Program.cs: builder.Services.AddHttpClient("python", ...)
        var client = _httpClientFactory.CreateClient("python");
     
        
        // זה בדיוק מה ש-Python FastAPI מצפה לקבל (File + Form fields)
        using var form = new MultipartFormDataContent();

        // פתיחת קובץ השמע לקריאה (streaming — לא טוען הכל לזיכרון)
        using var fileStream = File.OpenRead(recording.FilePath);
        using var fileContent = new StreamContent(fileStream);
        fileContent.Headers.ContentType = new MediaTypeHeaderValue("audio/wav");
        // הוספת הקובץ לטופס: שם השדה = "file", שם הקובץ כפי שיוצג ב-Python
        form.Add(fileContent, "file", recording.FileNameNew ?? "recording.wav");

     
        // Python משתמש בהם ל-logging ולקישור לשיעור הנכון
        form.Add(new StringContent(recording.LessonDate.ToString("yyyy-MM-dd")), "lesson_date");
        form.Add(new StringContent(recording.ScheduledLessonId.ToString()), "scheduled_lesson_id");

        // שליחת הבקשה ל-POST http://localhost:5000/process
        // EnsureSuccessStatusCode() = זורק Exception אם קוד תשובה >= 400
        var response = await client.PostAsync("process", form, ct);
        response.EnsureSuccessStatusCode();

        // פענוח JSON מהתשובה למחלקה PythonResult
        // ReadFromJsonAsync = קורא JSON ישירות מה-stream (יעיל יותר מ-ReadAsStringAsync + JsonSerializer.Deserialize)
        var json = await response.Content.ReadFromJsonAsync<PythonResult>(cancellationToken: ct);
        // אם Python החזיר גוף ריק — זה bug → זורקים Exception
        return json ?? throw new InvalidOperationException("Python returned empty response");
    }

    // -------------------------------------------------------
    // PythonResult — מבנה נתונים של תשובת ה-JSON מ-web_api.py
    // record = מחלקה immutable עם property לכל שדה
    // [JsonPropertyName] = שם השדה ב-JSON (snake_case ← camelCase)
    // -------------------------------------------------------
    // תואם לתשובת ה-JSON מ-web_api.py
    private record PythonResult(
        [property: JsonPropertyName("duration_sec")]    double              DurationSec,
        [property: JsonPropertyName("positive_pct")]    double              PositivePct,
        [property: JsonPropertyName("negative_pct")]    double              NegativePct,
        [property: JsonPropertyName("noise_count")]     int                 NoiseCount,
        [property: JsonPropertyName("total_windows")]   int                 TotalWindows,
        [property: JsonPropertyName("grade")]           string              Grade,
        [property: JsonPropertyName("windows")]         List<PythonWindow>? Windows);

    private record PythonWindow(
        [property: JsonPropertyName("start_sec")]          double  StartSec,
        [property: JsonPropertyName("end_sec")]            double  EndSec,
        [property: JsonPropertyName("speech_ratio")]       double  SpeechRatio,
        [property: JsonPropertyName("has_speech")]         bool    HasSpeech,
        [property: JsonPropertyName("rms")]                double? Rms,
        [property: JsonPropertyName("rms_db")]             double? RmsDb,
        [property: JsonPropertyName("rms_level")]          string? RmsLevel,
        [property: JsonPropertyName("audio_type")]         string? AudioType,
        [property: JsonPropertyName("speaker_type")]       string? SpeakerType,
        [property: JsonPropertyName("overlap_score")]      double? OverlapScore,
        [property: JsonPropertyName("context_category")]   string? ContextCategory,
        [property: JsonPropertyName("context_confidence")] double? ContextConfidence,
        [property: JsonPropertyName("state_machine")]      string? StateMachine,
        [property: JsonPropertyName("transcribed_text")]   string? TranscribedText,
        [property: JsonPropertyName("attention_label")]    string? AttentionLabel);
}
