// ===================================================
// Recording.cs — מודל נתוני הקלטה
// תואם במדויק לטבלת Recordings ב-SQL Server
// ===================================================
// מחלקה זו היא העריכות C# שמייצגת שורה אחת בטבלת Recordings.
// EF Core מתרגם אותה ל-INSERT/SELECT/UPDATE SQL.
// ===================================================
namespace AudioReceiver.Models;

public class Recording
{
    // מזהה ייחודי של ההקלטה (אותו-ינקרמנט על ידי SQL Server)
    public int Id { get; set; }
    // קשר לטבלת ScheduledLessons — איזה שיעור סדיר מדובר בהקלטה זו
    public int ScheduledLessonId { get; set; }
    // תאריך השיעור הספציפי (יום בלבד ללא שעה — DateOnly = .NET 6+)
    public DateOnly LessonDate { get; set; }
    // שם הקובץ הניתן בדיסק (GUID + סיומת, לדוגמה: "a1b2c3.wav")
    public string? FileNameNew { get; set; }
    // נתיב מלא לקובץ בדיסק, לדוגמא: "C:\audio_files\a1b2c3.wav"
    public string FilePath { get; set; } = string.Empty;
    // גודל הקובץ ב-bytes — לדיווח סטטיסטיקות
    public long FileSizeBytes { get; set; }
    // שם הקובץ המקורי שהעלותה המורה (לצורך הצגה בלבד)
    public string? OriginalFileName { get; set; }
    // סיומת הקובץ (ללא נקודה): "wav", "mp3", "ogg"
    public string? FormatFile { get; set; }
    // סטטוס עיבוד ההקלטה — ראה ProcessingStatus enum קדמה
    // ברירת מחדל = Pending (אפס = ממתין)
    public ProcessingStatus StatusProcessing { get; set; } = ProcessingStatus.Pending;
    // הודעת שגיאה אם העיבוד נכשל (null אם הצליח)
    public string? StatusProcessingErrorMessage { get; set; }
    // משך ההקלטה בשניות — מחושב על ידי Python (null לפני עיבוד)
    public double? DurationSec { get; set; }
    // אחוז הזמן החיובי (0-100) — Python מחשב אותו
    public double? PositivePct { get; set; }
    // אחוז הזמן השלילי (הפרעות)
    public double? NegativePct { get; set; }
    // ספירת חלונות שהיו רעש בלבד (לא נספרו בציון)
    public int? NoiseWindowsCount { get; set; }
    // ספירת חלונות עם דיבור אמיתי (רלוונטיים לציון)
    public int? TotalWindowsCount { get; set; }
    // דירוג מילולי: "מצוין" / "טוב" / "בינוני" / "נמוך" / או "N/A"
    public string? Grade { get; set; }
    // זמן העלאה (ב-UTC — סטנדרטי לאחסון בבסיס נתונים)
    public DateTime UploadedAt { get; set; } = DateTime.UtcNow;
    // מתי התחיל העיבוד (null = עדיין לא התחיל)
    public DateTime? ProcessingStartedAt { get; set; }
    // מתי הסתיים העיבוד (null = עדיין לא הסתיים)
    public DateTime? ProcessingFinishedAt { get; set; }
    // הערות חופשיות שכתבה המורה לפני העלאה
    public string? TeacherNotes { get; set; }
}

// ===================================================
// ProcessingStatus — מכונת המצבים של הקלטה
// ===================================================
// סדר המעברים:
//   Pending → Processing → Done
//                      ↘
//                      Failed
//
// הערך המספרי נשמר כ-byte ב-SQL Server
// ===================================================
public enum ProcessingStatus
{
    Pending    = 0,  // הקלטה נקלטה וממתינה בתור
    Processing = 1,  // QueueProcessorService שלח ל-Python ומצפה לתשובה
    Done       = 2,  // עיבוד הסתיים בהצלחה — TimeWindows נשמרו
    Failed     = 3   // עיבוד נכשל — ראה StatusProcessingErrorMessage
}
