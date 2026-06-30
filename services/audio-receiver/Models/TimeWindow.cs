// ===================================================
// TimeWindow.cs — מודל חלון זמן בתוך הקלטה
// תואם לטבלת TimeWindows ב-SQL Server
// ===================================================
namespace AudioReceiver.Models;

public class TimeWindow
{
    public int    Id           { get; set; }
    public int    RecordingId  { get; set; }
    public int    WindowIndex  { get; set; }

    public double StartSec     { get; set; }
    public double EndSec       { get; set; }

    // VAD
    public double SpeechRatio  { get; set; }
    public bool   HasSpeech    { get; set; }

    // RMS
    public double? RmsValue    { get; set; }
    public double? RmsDb       { get; set; }
    public string? RmsLevel    { get; set; }

    // סיווג אקוסטי
    public string? AudioType   { get; set; }

    // זיהוי דוברים
    public string? SpeakerType  { get; set; }
    public double? OverlapScore { get; set; }

    // ניתוח הקשרי
    public string? ContextCategory   { get; set; }
    public double? ContextConfidence { get; set; }
    public string? TranscribedText   { get; set; }
    public string? StateMachine      { get; set; }

    // ציון קשב
    public string? AttentionLabel    { get; set; }

    // ניווט (לא נמפה ל-DB בנפרד)
    public Recording? Recording { get; set; }
}
