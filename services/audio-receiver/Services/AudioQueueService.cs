// ===================================================
// AudioQueueService.cs — ניהול תור ההקלטות
// ===================================================
// שירות Singleton שמחזיק ConcurrentQueue עם מזהי ההקלטות.
// מבטיח שרק הקלטה אחת בכל פעם נשלחת לפייתון.
//
// זרימה:
//   RecordingsController.Enqueue(id) ← מורה מעלה הקלטה
//        ↓
//   ConcurrentQueue — מחזיק רשימת מזהים ממתינים בזיכרון
//        ↓
//   QueueProcessorService.TryDequeue() ← עובד רקע שולף ושולח לפייתון
// ===================================================

// ייבוא ConcurrentQueue — מבנה תור שבטוח לשימוש ממספר threads בו-זמנית
using System.Collections.Concurrent;

// Namespace = מרחב שמות — מקבץ יחד את כל הקוד של שכבת השירותים
namespace AudioReceiver.Services;

// הגדרת ממשק (interface) — חוזה שמגדיר מה כל מימוש חייב לספק
// שימוש בממשק מאפשר להחליף מימוש (למשל לצורך בדיקות)
public interface IAudioQueueService
{
    // הגדרת מתודה לדחיפת מזהה לתור
    void Enqueue(int recordingId);

    // הגדרת מתודה לשליפת מזהה מהתור
    // out = פרמטר פלט — מאפשר החזרת ערך נוסף בנוסף ל-bool
    bool TryDequeue(out int recordingId);

    // מאפיין (Property) לבדיקת כמה הקלטות ממתינות בתור
    int PendingCount { get; }
}

// מימוש קונקרטי של הממשק — זה השירות שמוזרק בפועל לכל הקוד
// ConcurrentQueue = תור מבוסס FIFO (ראשון נכנס, ראשון יוצא) שבטוח לריבוי threads
public class AudioQueueService : IAudioQueueService
{
    // יצירת תור ריק בזמן אתחול — נשמר בזיכרון לאורך כל חיי השרת
    // ConcurrentQueue<int> = תור של מספרים שלמים (מזהי הקלטות)
    private readonly ConcurrentQueue<int> _queue = new();

    // Enqueue — מוסיף מזהה לסוף התור
    // קורא לפונקציה המובנית של ConcurrentQueue → בטיחות thread מובנית
    public void Enqueue(int recordingId) => _queue.Enqueue(recordingId);

    // TryDequeue — מנסה לשלוף מהתחלה; מחזיר true אם הצליח, false אם התור ריק
    // out recordingId = מוחזר בפרמטר אם הצלחנו לשלוף
    public bool TryDequeue(out int recordingId) => _queue.TryDequeue(out recordingId);

    // PendingCount — כמה הקלטות ממתינות עכשיו בתור
    // Count הוא property מובנה של ConcurrentQueue
    public int PendingCount => _queue.Count;
}
