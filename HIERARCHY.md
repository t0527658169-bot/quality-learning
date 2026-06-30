# היררכיית המערכת — זרימת נתונים מהקלטה לתוצאה

מסמך זה מתאר את הסדר המדויק שבו עוברת כל הקלטה דרך המערכת.
לכל שלב: שם הקובץ, שם הפונקציה, ומה היא עושה.

---

## שלב 1 — קבלת ההקלטה
**קובץ:** `services/audio-receiver/Controllers/RecordingsController.cs`
**פונקציה:** `UploadRecording()` → נתיב: `POST /api/recordings/upload`

מה קורה כאן:
- הדפדפן שולח קובץ שמע (WAV/MP3) + מטא-דאטה (מורה, כיתה, תאריך)
- הקובץ נשמר לדיסק עם שם GUID ייחודי (לדוגמה: `a1b2c3.wav`)
- נוצרת רשומה חדשה בטבלת `Recordings` ב-SQL Server עם `StatusProcessing = Pending`
- מזהה ההקלטה (`recordingId`) נדחף לתור → ממתין לעיבוד

**קובץ הבא:** `AudioQueueService.cs`

---

## שלב 2 — ניהול התור
**קובץ:** `services/audio-receiver/Services/AudioQueueService.cs`
**ממשק:** `IAudioQueueService` → מימוש: `AudioQueueService`

מה קורה כאן:
- `Enqueue(recordingId)` — דוחף את מזהה ההקלטה לתור (ConcurrentQueue)
- ConcurrentQueue = תור בטוח ל-threads מרובים (הבקר כותב, עובד הרקע קורא)
- התור נמצא **בזיכרון** — לא ב-DB. אם השרת נכבה, הנתונים נשמרים ב-DB (Pending) ונטענים מחדש בהפעלה הבאה

**קובץ הבא:** `QueueProcessorService.cs`

---

## שלב 3 — עובד הרקע (שולח ל-Python)
**קובץ:** `services/audio-receiver/Services/QueueProcessorService.cs`
**פונקציה:** `ExecuteAsync()` → לולאה אינסופית ברקע

מה קורה כאן:
1. `ExecuteAsync()` — מופעל עם עליית השרת. טוען כל הקלטות Pending מה-DB לתור
2. `ProcessRecordingAsync(recordingId)` — שולף הקלטה מהתור ומטפל בה
3. `SendToPythonAsync(recording)` — שולח HTTP POST לכתובת `http://localhost:5000/process`:
   - body: multipart/form-data עם קובץ WAV + מטא-דאטה
4. מקבל תשובת JSON מ-Python עם כל תוצאות הניתוח
5. שומר `TimeWindow` לכל חלון זמן בטבלת `TimeWindows` ב-DB
6. מעדכן את `Recording.StatusProcessing = Done` + ציוני הסיכום

**קובץ הבא:** `web_api.py` (Python)

---

## שלב 4 — שרת Python מקבל את הקובץ
**קובץ:** `services/audio-processor/web_api.py`
**פונקציה:** `process_audio()` → נתיב: `POST /process`

מה קורה כאן:
- מקבל את הקובץ ומטא-דאטה מה-C#
- שומר את הקובץ לקובץ זמני בדיסק (נמחק אחרי העיבוד)
- קורא ל-`AudioPipeline.process_file(path)` — כל הניתוח קורה שם
- בנוי כ-**Singleton Pattern**: ה-pipeline נטען פעם אחת בלבד (מודלי AI כבדים)
- מחזיר JSON עם תוצאות הניתוח ל-C#

**קובץ הבא:** `main_pipeline.py`

---

## שלב 5 — ה-Pipeline המרכזי
**קובץ:** `services/audio-processor/main_pipeline.py`
**פונקציה:** `AudioPipeline.process_file(path)`

מה קורה כאן בסדר:

### א. טעינת האודיו
- קורא את קובץ WAV בעזרת `scipy.io.wavfile`
- ממיר לקצב דגימה אחיד: 16,000 דגימות לשנייה (`SAMPLE_RATE`)
- ממיר ל-float32 (ערכים בין -1 ל-1)

### ב. ניקוי רעשים
- קורא לפונקציה `WienerFilter.apply(audio)` ← **קובץ:** `wiener_filter.py`
- מנקה רעשי רקע (מזגן, כיסאות) תוך שמירה על הדיבור

### ג. כיול ה-Overlap Detector
- קורא ל-`OverlapDetector.calibrate(audio)` ← **קובץ:** `overlap_detector.py`
- מחשב את ה-`harm_threshold` הדינמי לפי ההקלטה הספציפית
- (calibrate בוחן את ה-harmonicity של כל ההקלטה כדי לקבוע מה "גבוה" ומה "נמוך")

### ד. לולאת חלונות (כל 3 שניות)
כל 3 שניות נקראות הפונקציות הבאות בסדר:

**1. VAD — זיהוי דיבור**
- `VADDetector.get_speech_ratio(chunk)` ← **קובץ:** `vad_detector.py`
- בודק: האם יש דיבור בחלון הזה בכלל? (חישוב אנרגיה + ZCR + ספקטרום)
- אם `speech_ratio < 0.1` → חלון נחשב כשקט, דילוג

**2. RMS — עוצמת קול**
- `RMSAnalyzer.analyze_chunk(chunk)` ← **קובץ:** `rms_analyzer.py`
- מחשב אנרגיה (RMS) ומסווג: שקט / רגיל / רועש
- אם רעש חזק מדי → מסמן כ-`רעש`, לא נספר בציון

**3. סיווג אקוסטי**
- `AudioClassifier.classify(chunk)` ← **קובץ:** `audio_classifier.py`
- מחשב מקדמי MFCC ומסווג לאחת מ-4 קטגוריות:
  שקט / רעש_סביבתי / פטפטת_חלשה / דיבור_ברור

**4. זיהוי ריבוי דוברים**
- `OverlapDetector.detect(chunk)` ← **קובץ:** `overlap_detector.py`
- מחשב harmonicity (ACF) + overlap_score (אנרגיה+ZCR+שטחיות+רוחב_סרט)
- מסווג לאחת מ-3 תוצאות: `דובר_יחיד` / `ריבוי_דוברים` / `רעש`

**5. ניתוח הקשר עברי (HEBERT + Whisper)**
- `HEBERTContextAnalyzer.check_opening(chunk)` ← **קובץ:** `hebert_context_analyzer.py`
- `HEBERTContextAnalyzer.check_silencing(chunk)` ← **קובץ:** `hebert_context_analyzer.py`
- Whisper מתמלל את הדיבור לטקסט עברי
- HEBERT (avichr/heBERT) ממיר את הטקסט לוקטור 768 ממדי
- Cosine Similarity בין הוקטור לבין ביטויי-מפתח → מזהה פתיחת דיון / השתקה

**6. ציון חלון**
- `AttentionScorer.score_window(window_result)` ← **קובץ:** `attention_scorer.py`
- מסווג את החלון: `חיובי` / `הפרעה` / `None` (רעש)

### ה. ציון הסיכום
- `AttentionScorer.score_lesson(window_scores)` ← **קובץ:** `attention_scorer.py`
- מחשב אחוז חיובי מתוך הרלוונטיים
- קובע דירוג: מצוין (≥85%) / טוב (≥70%) / בינוני (≥50%) / נמוך (<50%)

**קובץ הבא:** חזרה ל-`web_api.py` → חזרה ל-`QueueProcessorService.cs`

---

## מודולי עזר (אין להם "תור" — נקראים לפי הצורך)

| קובץ | תפקיד |
|------|--------|
| `config.py` | כל הקבועים המספריים של המערכת (ספים, פרמטרים) |
| `audio_utils.py` | פונקציות חישוב: frame_audio, compute_rms, compute_mfcc, compute_zcr |
| `wiener_filter.py` | פילטר וינר — ניקוי רעשים |

---

## מבנה בסיס הנתונים (SQL Server — QualityLearning)

```
Recordings (טבלה ראשית)
  id, scheduled_lesson_id, lesson_date
  file_path, file_size_bytes, original_file_name
  status_Processing: 0=Pending, 1=Processing, 2=Done, 3=Error
  duration_sec, positive_pct, negative_pct, grade
  uploaded_at, processing_started_at, processing_finished_at

TimeWindows (תוצאות לכל חלון זמן)
  id, recording_id (FK → Recordings.id)
  start_sec, end_sec
  speaker_type: דובר_יחיד / ריבוי_דוברים / רעש
  rms_level, classification, score_label
  overlap_score, harmonicity, speech_ratio
  is_opening_discussion, is_silencing
```

---

## סדר עיון מומלץ לבוחן (לפי הזרימה)

1. `Recording.cs` — מהו מבנה הנתונים
2. `AppDbContext.cs` — איך C# מתחבר ל-SQL
3. `Program.cs` — איך עולה השרת
4. `AuthController.cs` — כניסת מורה
5. `RecordingsController.cs` — קבלת הקלטה חדשה
6. `AudioQueueService.cs` — התור הפנימי
7. `QueueProcessorService.cs` — שליחה ל-Python
8. `web_api.py` — שרת Python מקבל
9. `config.py` — כל הפרמטרים
10. `main_pipeline.py` — תזמורת הניתוח
11. `wiener_filter.py` — ניקוי רעשים
12. `vad_detector.py` — זיהוי דיבור
13. `rms_analyzer.py` — עוצמת קול
14. `audio_classifier.py` — סיווג MFCC
15. `overlap_detector.py` — זיהוי ריבוי דוברים
16. `hebert_context_analyzer.py` — HEBERT + Whisper
17. `attention_scorer.py` — ציון סופי
18. `audio_utils.py` — פונקציות עזר
