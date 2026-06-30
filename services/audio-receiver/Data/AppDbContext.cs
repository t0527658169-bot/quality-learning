// ===================================================
// AppDbContext.cs — חיבור ל-SQL Server עם EF Core
// מיפוי עמודות תואם לסכמת Recordings
// ===================================================
// AppDbContext הוא ה"שער" בין קוד C# לבסיס הנתונים.
// EF Core (Entity Framework Core) מתרגם בין אובייקטים ל-SQL.
//
// מה נמצא כאן:
//   1. DbSet<Recording>  — ממפה לטבלת Recordings ב-SQL
//   2. DbSet<TimeWindow> — ממפה לטבלת TimeWindows ב-SQL
//   3. OnModelCreating   — הגדרת שמות עמודות מדויקים
//      (C# משתמש בPascalCase, SQL משתמש ב-snake_case)
// ===================================================
using AudioReceiver.Models;
using Microsoft.EntityFrameworkCore;

namespace AudioReceiver.Data;

// DbContext הוא המחלקה הבסיסית של EF Core לניהול חיבורים ושאילתות
public class AppDbContext : DbContext
{
    // קבלת הגדרות החיבור מ-DI (connection string מ-appsettings.json)
    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

    // DbSet = אוסף של שורות מטבלה — משמש לשאילתות LINQ ולשמירה
    public DbSet<Recording>   Recordings   => Set<Recording>();
    public DbSet<TimeWindow>  TimeWindows  => Set<TimeWindow>();

    // OnModelCreating — הגדרת מיפוי מפורש בין שדות C# לעמודות SQL
    // נדרש כי EF Core לא מנחש שמות snake_case אוטומטית
    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        // -------------------------------------------------------
        // מיפוי Recordings
        // -------------------------------------------------------
        modelBuilder.Entity<Recording>(entity =>
        {
            // ToTable = שם הטבלה ב-SQL Server
            entity.ToTable("Recordings");
            // HasKey = מפתח ראשי (Primary Key)
            entity.HasKey(r => r.Id);
            // UseIdentityColumn = SQL Server AUTO_INCREMENT (IDENTITY)
            entity.Property(r => r.Id).HasColumnName("id").UseIdentityColumn();

            entity.Property(r => r.ScheduledLessonId).HasColumnName("scheduled_lesson_id").IsRequired();
            entity.Property(r => r.LessonDate).HasColumnName("lesson_date").IsRequired();

            entity.Property(r => r.FileNameNew).HasColumnName("file_name_new").HasMaxLength(260);
            entity.Property(r => r.FilePath).HasColumnName("file_path").HasMaxLength(500).IsRequired();
            entity.Property(r => r.FileSizeBytes).HasColumnName("file_size_bytes").IsRequired();
            entity.Property(r => r.OriginalFileName).HasColumnName("original_file_name").HasMaxLength(260);
            entity.Property(r => r.FormatFile).HasColumnName("format_file").HasMaxLength(10);

            // HasConversion<byte> = שמירת ה-enum כמספר (0-3) ב-SQL
            entity.Property(r => r.StatusProcessing)
                  .HasColumnName("status_Processing")
                  .HasConversion<byte>()
                  .IsRequired();
            entity.Property(r => r.StatusProcessingErrorMessage)
                  .HasColumnName("status_Processing_error_message");

            entity.Property(r => r.DurationSec).HasColumnName("duration_sec");
            entity.Property(r => r.PositivePct).HasColumnName("positive_pct");
            entity.Property(r => r.NegativePct).HasColumnName("negative_pct");
            entity.Property(r => r.NoiseWindowsCount).HasColumnName("noise_windows_count");
            entity.Property(r => r.TotalWindowsCount).HasColumnName("total_windows_count");
            entity.Property(r => r.Grade).HasColumnName("grade").HasMaxLength(20);

            entity.Property(r => r.UploadedAt).HasColumnName("uploaded_at").IsRequired();
            entity.Property(r => r.ProcessingStartedAt).HasColumnName("processing_started_at");
            entity.Property(r => r.ProcessingFinishedAt).HasColumnName("processing_finished_at");

            entity.Property(r => r.TeacherNotes).HasColumnName("teacher_notes").HasMaxLength(500);

            // אינדקסים לאחסון יעיל — מאיצים שאילתות חיפוש לפי שיעור ותאריך
            entity.HasIndex(r => r.ScheduledLessonId).HasDatabaseName("ix_recordings_lesson");
            entity.HasIndex(r => r.LessonDate).HasDatabaseName("ix_recordings_date");
        });

        // -------------------------------------------------------
        // מיפוי TimeWindows
        // -------------------------------------------------------
        modelBuilder.Entity<TimeWindow>(entity =>
        {
            entity.ToTable("TimeWindows");
            entity.HasKey(w => w.Id);
            entity.Property(w => w.Id).HasColumnName("id").UseIdentityColumn();
            entity.Property(w => w.RecordingId).HasColumnName("recording_id").IsRequired();
            entity.Property(w => w.WindowIndex).HasColumnName("window_index").IsRequired();
            entity.Property(w => w.StartSec).HasColumnName("start_sec").IsRequired();
            entity.Property(w => w.EndSec).HasColumnName("end_sec").IsRequired();
            entity.Property(w => w.SpeechRatio).HasColumnName("speech_ratio").IsRequired();
            entity.Property(w => w.HasSpeech).HasColumnName("has_speech").IsRequired();
            entity.Property(w => w.RmsValue).HasColumnName("rms_value");
            entity.Property(w => w.RmsDb).HasColumnName("rms_db");
            entity.Property(w => w.RmsLevel).HasColumnName("rms_level").HasMaxLength(20);
            entity.Property(w => w.AudioType).HasColumnName("audio_type").HasMaxLength(30);
            entity.Property(w => w.SpeakerType).HasColumnName("speaker_type").HasMaxLength(30);
            entity.Property(w => w.OverlapScore).HasColumnName("overlap_score");
            entity.Property(w => w.ContextCategory).HasColumnName("context_category").HasMaxLength(30);
            entity.Property(w => w.ContextConfidence).HasColumnName("context_confidence");
            entity.Property(w => w.TranscribedText).HasColumnName("transcribed_text");
            entity.Property(w => w.StateMachine).HasColumnName("state_machine").HasMaxLength(20);
            entity.Property(w => w.AttentionLabel).HasColumnName("attention_label").HasMaxLength(20);

            // Cascade Delete — כשמוחקים Recordings, נמחקים גם כל ה-TimeWindows שלה
            entity.HasOne(w => w.Recording)
                  .WithMany()
                  .HasForeignKey(w => w.RecordingId)
                  .OnDelete(DeleteBehavior.Cascade);

            // אינדקס על recording_id — מאיץ שאילתות "תן לי את כל החלונות של הקלטה X"
            entity.HasIndex(w => w.RecordingId).HasDatabaseName("ix_timewindows_recording");
        });
    }
}
