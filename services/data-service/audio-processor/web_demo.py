# ===================================================
# web_demo.py — DISABLED (data-service — גרסה ישנה)
# *** קובץ זה שייך לגרסה הישנה לפני HEBERT ***
# *** ניתן למחוק את תיקיית data-service/audio-processor כולה ***
# ===================================================

# ===================================================
# web_demo.py — שרת דמו עם למידה מקוונת (נפרד לחלוטין!)
# ===================================================
# קובץ עזר לבדיקה נוחה של האלגוריתמים דרך דפדפן.
# *** ניתן למחוק קובץ זה + demo.html בלי לפגוע בפרויקט ***
#
# תכונות:
#   - העלאת קובץ שמע וקבלת ניתוח
#   - תיקון תיוגי דוברים (יחיד/ריבוי) בממשק
#   - אימון מחדש של המודל מהתיוגים שנצברו
#
# שימוש:
#   python web_demo.py
#   ואז לפתוח בדפדפן: http://localhost:8080

import os
import sys
import json
import tempfile
import warnings
import pickle
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import cgi

warnings.filterwarnings('ignore')

from main_pipeline import AudioPipeline
from overlap_detector import OverlapDetector
from train_overlap_model import extract_features

PIPELINE = None
PORT = 8080

# קובץ לשמירת נתוני אימון שנצברו מהמשתמש
LABELED_DATA_PATH = os.path.join(os.path.dirname(__file__), "models", "labeled_data.pkl")
# נתוני אימון להקשר (ריבוי דוברים: לצורך השיעור או הפרעה)
CONTEXT_LABELED_PATH = os.path.join(os.path.dirname(__file__), "models", "context_labeled_data.pkl")
# נתוני אימון trigger — features מהחלון שלפני רצף ריבוי (מה שהמורה אמרה)
TRIGGER_LABELED_PATH = os.path.join(os.path.dirname(__file__), "models", "trigger_labeled_data.pkl")
# שמירת חלונות אודיו של הניתוח האחרון (לצורך חילוץ features בעת תיוג)
LAST_ANALYSIS_CHUNKS = []
# שמירת האודיו הגולמי + נקודות ההתחלה — לצורך חילוץ הקשר (5 שניות לפני)
LAST_RAW_AUDIO = None
LAST_CHUNK_OFFSETS = []
# שמירת נתיב הקובץ האחרון — לצורך ניתוח מחדש
LAST_AUDIO_PATH = None


def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = AudioPipeline()
    return PIPELINE


def load_labeled_data():
    """טעינת נתוני אימון שנצברו."""
    if os.path.exists(LABELED_DATA_PATH):
        with open(LABELED_DATA_PATH, 'rb') as f:
            return pickle.load(f)
    return {'features': [], 'labels': []}


def save_labeled_data(data):
    """שמירת נתוני אימון."""
    os.makedirs(os.path.dirname(LABELED_DATA_PATH), exist_ok=True)
    with open(LABELED_DATA_PATH, 'wb') as f:
        pickle.dump(data, f)


def load_context_labeled():
    """טעינת נתוני אימון הקשריים (ריבוי: שיעור vs הפרעה)."""
    if os.path.exists(CONTEXT_LABELED_PATH):
        with open(CONTEXT_LABELED_PATH, 'rb') as f:
            return pickle.load(f)
    return {'features': [], 'labels': []}


def save_context_labeled(data):
    """שמירת נתוני אימון הקשריים."""
    os.makedirs(os.path.dirname(CONTEXT_LABELED_PATH), exist_ok=True)
    with open(CONTEXT_LABELED_PATH, 'wb') as f:
        pickle.dump(data, f)


def retrain_model(labeled_data):
    """אימון מחדש של מודל ה-overlap מנתונים שנצברו."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X = np.array(labeled_data['features'])
    y = np.array(labeled_data['labels'])

    n_samples = len(y)
    n_single = int(np.sum(y == 0))
    n_multi = int(np.sum(y == 1))
    n_noise = int(np.sum(y == 2))

    # צריך לפחות 2 דוגמאות מכל סוג קיים
    unique = set(y.tolist())
    for cls_val in unique:
        if int(np.sum(y == cls_val)) < 2:
            cls_name = {0: 'יחיד', 1: 'ריבוי', 2: 'רעש'}[cls_val]
            return None, f"צריך לפחות 2 דוגמאות מסוג '{cls_name}'. יש כרגע: {int(np.sum(y == cls_val))}"

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X, y)
    accuracy = accuracy_score(y, model.predict(X))

    # שמירת המודל
    model_path = os.path.join(os.path.dirname(__file__), "models", "overlap_rf_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'accuracy': accuracy}, f)

    # טעינה מחדש ב-pipeline
    global PIPELINE
    PIPELINE = None
    get_pipeline()

    return accuracy, f"מודל דוברים אומן על {n_samples} דוגמאות ({n_single} יחיד, {n_multi} ריבוי, {n_noise} רעש). דיוק: {accuracy:.1%}"


def load_trigger_labeled():
    """טעינת נתוני אימון trigger (חלון קודם לרצף ריבוי)."""
    if os.path.exists(TRIGGER_LABELED_PATH):
        with open(TRIGGER_LABELED_PATH, 'rb') as f:
            return pickle.load(f)
    return {'features': [], 'labels': []}


def save_trigger_labeled(data):
    """שמירת נתוני אימון trigger."""
    os.makedirs(os.path.dirname(TRIGGER_LABELED_PATH), exist_ok=True)
    with open(TRIGGER_LABELED_PATH, 'wb') as f:
        pickle.dump(data, f)


def retrain_trigger_model(trigger_data):
    """אימון מודל trigger — לומד מהחלון שלפני רצף ריבוי מה הכוונה."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X = np.array(trigger_data['features'])
    y = np.array(trigger_data['labels'])
    n_total = len(y)

    if n_total < 2:
        return None, f"מודל trigger: צריך לפחות 2 דוגמאות. יש כרגע: {n_total}"

    unique_classes = list(set(y.tolist()))
    if len(unique_classes) == 1:
        only_class = unique_classes[0]
        class_name = 'הפרעה' if only_class == 0 else 'שיעור'
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy='constant', constant=only_class)
        model.fit(X, y)
        accuracy = 1.0
    else:
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
        )
        model.fit(X, y)
        accuracy = accuracy_score(y, model.predict(X))

    model_path = os.path.join(os.path.dirname(__file__), "models", "trigger_rf_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'accuracy': accuracy}, f)

    n_lesson = int(np.sum(y == 1))
    n_disrupt = int(np.sum(y == 0))
    return accuracy, f"מודל trigger אומן על {n_total} דוגמאות ({n_lesson} שיעור, {n_disrupt} הפרעה). דיוק: {accuracy:.1%}"


def retrain_context_model(ctx_data):
    """אימון מחדש של מודל ההקשר (לצורך השיעור vs הפרעה) מנתונים שנצברו."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X = np.array(ctx_data['features'])
    y = np.array(ctx_data['labels'])

    n_lesson = int(np.sum(y == 1))
    n_disrupt = int(np.sum(y == 0))
    n_total = n_lesson + n_disrupt

    if n_total < 2:
        return None, f"צריך לפחות 2 דוגמאות. יש כרגע: {n_total}"

    # אם יש רק סוג אחד — שמירת מודל שתמיד מחזיר את הסוג הזה
    unique_classes = list(set(y.tolist()))
    if len(unique_classes) == 1:
        only_class = unique_classes[0]
        class_name = 'הפרעה' if only_class == 0 else 'שיעור'

        # יצירת מודל פשוט שמחזיר תמיד את המחלקה היחידה
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy='constant', constant=only_class)
        model.fit(X, y)
        accuracy = 1.0

        ctx_model_path = os.path.join(os.path.dirname(__file__), "models", "context_rf_model.pkl")
        with open(ctx_model_path, 'wb') as f:
            pickle.dump({'model': model, 'accuracy': accuracy}, f)

        return accuracy, f"מודל הקשר: כל {n_total} הדוגמאות הן '{class_name}' — המודל ילמד להחזיר '{class_name}'. הוסיפי דוגמאות מהסוג השני לדיוק טוב יותר."

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X, y)
    accuracy = accuracy_score(y, model.predict(X))

    # שמירת המודל (מחליף את מודל ההקשר הישן)
    ctx_model_path = os.path.join(os.path.dirname(__file__), "models", "context_rf_model.pkl")
    with open(ctx_model_path, 'wb') as f:
        pickle.dump({'model': model, 'accuracy': accuracy}, f)

    return accuracy, f"מודל הקשר אומן על {n_total} דוגמאות ({n_lesson} שיעור, {n_disrupt} הפרעה). דיוק: {accuracy:.1%}"


class DemoHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self._serve_html()
        elif self.path == '/stats':
            self._handle_stats()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/analyze':
            self._handle_analyze()
        elif self.path == '/label':
            self._handle_label()
        elif self.path == '/retrain':
            self._handle_retrain()
        elif self.path == '/train':
            self._handle_save_and_train()
        elif self.path == '/reanalyze':
            self._handle_reanalyze()
        else:
            self.send_error(404)

    # --- ניתוח קובץ שמע ---
    def _handle_analyze(self):
        global LAST_ANALYSIS_CHUNKS, LAST_RAW_AUDIO, LAST_CHUNK_OFFSETS
        try:
            content_type = self.headers.get('Content-Type', '')
            if 'multipart/form-data' not in content_type:
                self._send_json(400, {'error': 'Content-Type must be multipart/form-data'})
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': content_type}
            )

            file_item = form['audio']
            if not file_item.filename:
                self._send_json(400, {'error': 'No file uploaded'})
                return

            global LAST_AUDIO_PATH
            # מחיקת קובץ קודם
            if LAST_AUDIO_PATH and os.path.exists(LAST_AUDIO_PATH):
                os.remove(LAST_AUDIO_PATH)

            suffix = os.path.splitext(file_item.filename)[1] or '.wav'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_item.file.read())
                tmp_path = tmp.name
            LAST_AUDIO_PATH = tmp_path

            pipeline = get_pipeline()

            # לכידת הפלט
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                results = pipeline.process_file(tmp_path)
            finally:
                sys.stdout = old_stdout

            # שמירת חלונות האודיו לצורך חילוץ features בעת תיוג
            audio = pipeline.load_audio(tmp_path)
            clean = pipeline.wiener.apply(audio)
            window_samples = int(pipeline.sr * 3.0)
            LAST_ANALYSIS_CHUNKS = []
            LAST_RAW_AUDIO = audio
            LAST_CHUNK_OFFSETS = []
            for i in range(0, len(clean), window_samples):
                chunk = clean[i:i + window_samples]
                if len(chunk) >= pipeline.sr:
                    LAST_ANALYSIS_CHUNKS.append(chunk)
                    LAST_CHUNK_OFFSETS.append(i)

            # לא מוחקים — נשמר ל-reanalyze

            windows_data = []
            for idx, w in enumerate(results['windows']):
                label = w['attention_label']  # "חיובי" / "הפרעה" / None
                windows_data.append({
                    'idx': idx,
                    'start': w['start_sec'],
                    'end': w['end_sec'],
                    'speech_ratio': round(w['speech_ratio'] * 100, 1),
                    'rms_level': w['rms_level'],
                    'speaker_type': w['speaker_type'],
                    'context': w['context_category'],
                    'label': label if label else 'רעש',
                    'is_noise': label is None,
                })

            # ספירת נתוני אימון קיימים
            labeled = load_labeled_data()
            ls = results['lesson_score']

            # פירוט סוגי חלונות
            n_single = sum(1 for w in windows_data if w['speaker_type'] == 'דובר_יחיד' and w['label'] == 'חיובי')
            n_multi_lesson = sum(1 for w in windows_data if not w['is_noise'] and w['speaker_type'] == 'ריבוי_דוברים' and w['label'] == 'חיובי')
            n_disruption = sum(1 for w in windows_data if w['label'] == 'הפרעה')
            n_noise = sum(1 for w in windows_data if w['is_noise'])
            total = len(windows_data)

            response = {
                'duration_sec': round(results['duration_sec'], 1),
                'duration_min': round(results['duration_sec'] / 60, 1),
                'windows': windows_data,
                'positive_pct': ls['positive_pct'],
                'negative_pct': ls['negative_pct'],
                'noise_count': ls['noise_count'],
                'total_relevant': ls['total_relevant'],
                'grade': ls['grade'],
                'labeled_count': len(labeled['labels']),
                'breakdown': {
                    'single': n_single,
                    'multi_lesson': n_multi_lesson,
                    'disruption': n_disruption,
                    'noise': n_noise,
                    'total': total,
                },
            }
            # DEBUG
            print(f"  [DEBUG] positive_pct={ls['positive_pct']}, negative_pct={ls['negative_pct']}, grade={ls['grade']}")
            print(f"  [DEBUG] total_relevant={ls['total_relevant']}, noise_count={ls['noise_count']}")
            for w in windows_data:
                print(f"    win {w['idx']}: speaker={w['speaker_type']}, label={w['label']}, rms={w['rms_level']}, noise={w['is_noise']}, speech={w['speech_ratio']}%")
            self._send_json(200, response)

        except Exception as e:
            self._send_json(500, {'error': str(e)})

    # --- שמירת תיוגים ---
    def _handle_label(self):
        global LAST_ANALYSIS_CHUNKS, LAST_RAW_AUDIO, LAST_CHUNK_OFFSETS
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8')
            data = json.loads(body)

            corrections = data.get('corrections', [])
            if not corrections:
                self._send_json(400, {'error': 'No corrections provided'})
                return

            if not LAST_ANALYSIS_CHUNKS:
                self._send_json(400, {'error': 'No analysis data in memory. Please re-analyze first.'})
                return

            labeled = load_labeled_data()
            ctx_labeled = load_context_labeled()
            added = 0
            ctx_added = 0

            pipeline = get_pipeline()

            for c in corrections:
                idx = c.get('idx')
                label = c.get('label')
                # label: 'single', 'noise', 'multiple_lesson', 'multiple_disruption'
                if idx is None or label is None:
                    continue
                if idx < 0 or idx >= len(LAST_ANALYSIS_CHUNKS):
                    continue

                # --- אימון מודל דוברים ---
                features = extract_features(LAST_ANALYSIS_CHUNKS[idx])
                if label == 'single':
                    y = 0
                elif label == 'noise':
                    y = 2
                else:
                    y = 1  # multiple_lesson / multiple_disruption -> ריבוי
                labeled['features'].append(features.tolist())
                labeled['labels'].append(y)
                added += 1

                # --- אימון מודל הקשר (רק לריבוי דוברים) ---
                if label in ('multiple_lesson', 'multiple_disruption'):
                    # חילוץ features מהחלון עצמו (זהה למה שהמודל מקבל בזמן חיזוי)
                    ctx_features = extract_features(LAST_ANALYSIS_CHUNKS[idx])
                    # 0 = הפרעה, 1 = לצורך השיעור
                    ctx_y = 1 if label == 'multiple_lesson' else 0
                    ctx_labeled['features'].append(ctx_features.tolist())
                    ctx_labeled['labels'].append(ctx_y)
                    ctx_added += 1

            save_labeled_data(labeled)
            save_context_labeled(ctx_labeled)

            n_single = sum(1 for l in labeled['labels'] if l == 0)
            n_multi = sum(1 for l in labeled['labels'] if l == 1)
            n_noise = sum(1 for l in labeled['labels'] if l == 2)
            n_ctx_lesson = sum(1 for l in ctx_labeled['labels'] if l == 1)
            n_ctx_disrupt = sum(1 for l in ctx_labeled['labels'] if l == 0)

            self._send_json(200, {
                'added': added,
                'ctx_added': ctx_added,
                'total': len(labeled['labels']),
                'single_count': n_single,
                'multi_count': n_multi,
                'noise_count': n_noise,
                'ctx_lesson': n_ctx_lesson,
                'ctx_disruption': n_ctx_disrupt,
            })

        except Exception as e:
            self._send_json(500, {'error': str(e)})

    # --- אימון מחדש ---
    def _handle_retrain(self):
        try:
            messages = []

            # אימון מודל דוברים
            labeled = load_labeled_data()
            if len(labeled['labels']) >= 4:
                accuracy, msg = retrain_model(labeled)
                if accuracy is None:
                    messages.append(msg)
                else:
                    messages.append(msg)
            else:
                messages.append(f"מודל דוברים: צריך לפחות 4 דוגמאות (יש {len(labeled['labels'])})")

            # אימון מודל הקשר
            ctx_labeled = load_context_labeled()
            if len(ctx_labeled['labels']) >= 2:
                ctx_acc, ctx_msg = retrain_context_model(ctx_labeled)
                if ctx_acc is None:
                    messages.append(ctx_msg)
                else:
                    messages.append(ctx_msg)
            else:
                messages.append(f"מודל הקשר: צריך לפחות 2 דוגמאות (יש {len(ctx_labeled['labels'])})")

            # אימון מודל trigger
            trigger_data = load_trigger_labeled()
            if len(trigger_data['labels']) >= 2:
                trig_acc, trig_msg = retrain_trigger_model(trigger_data)
                if trig_acc is None:
                    messages.append(trig_msg)
                else:
                    messages.append(trig_msg)
            else:
                messages.append(f"מודל trigger: צריך לפחות 2 דוגמאות (יש {len(trigger_data['labels'])})")

            # טעינה מחדש
            global PIPELINE
            PIPELINE = None
            get_pipeline()

            self._send_json(200, {'message': ' | '.join(messages)})

        except Exception as e:
            self._send_json(500, {'error': str(e)})

    # --- שמירה + אימון בפעולה אחת ---
    def _handle_save_and_train(self):
        """שומר תיוגים ומאמן מחדש בלחיצה אחת."""
        global LAST_ANALYSIS_CHUNKS, LAST_RAW_AUDIO, LAST_CHUNK_OFFSETS
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8')
            data = json.loads(body)
            corrections = data.get('corrections', [])

            # --- שלב 1: שמירת תיוגים ---
            save_msg = ""
            if corrections and LAST_ANALYSIS_CHUNKS:
                labeled = load_labeled_data()
                ctx_labeled = load_context_labeled()
                added = 0
                ctx_added = 0
                pipeline = get_pipeline()

                for c in corrections:
                    idx = c.get('idx')
                    label = c.get('label')
                    if idx is None or label is None:
                        continue
                    if idx < 0 or idx >= len(LAST_ANALYSIS_CHUNKS):
                        continue

                    features = extract_features(LAST_ANALYSIS_CHUNKS[idx])
                    if label == 'single':
                        y = 0
                    elif label == 'noise':
                        y = 2
                    else:
                        y = 1
                    labeled['features'].append(features)
                    labeled['labels'].append(y)
                    added += 1

                    if label in ('multiple_lesson', 'multiple_disruption'):
                        ctx_feat = extract_features(LAST_ANALYSIS_CHUNKS[idx])
                        ctx_y = 1 if label == 'multiple_lesson' else 0
                        ctx_labeled['features'].append(ctx_feat)
                        ctx_labeled['labels'].append(ctx_y)
                        ctx_added += 1

                save_labeled_data(labeled)
                if ctx_added > 0:
                    save_context_labeled(ctx_labeled)

                # --- שלב 1.5: למידה מהחלון שלפני רצף ריבוי (trigger) ---
                # כשמשתמש מתקן רצף רצוף של ריבוי דוברים, החלון שלפני הרצף
                # כנראה מכיל את מה שהמורה אמרה (שאלה/הנחיה) שגרם לריבוי.
                multi_corrections = [
                    (c['idx'], c['label']) for c in corrections
                    if c.get('label') in ('multiple_lesson', 'multiple_disruption')
                    and 0 <= c.get('idx', -1) < len(LAST_ANALYSIS_CHUNKS)
                ]
                trigger_added = 0
                if multi_corrections:
                    multi_corrections.sort(key=lambda x: x[0])
                    # קיבוץ לרצפים רצופים עם אותו תיוג
                    groups = []
                    current_group = [multi_corrections[0]]
                    for mi in range(1, len(multi_corrections)):
                        m_idx, m_label = multi_corrections[mi]
                        prev_idx, prev_label = current_group[-1]
                        if m_idx == prev_idx + 1 and m_label == prev_label:
                            current_group.append((m_idx, m_label))
                        else:
                            groups.append(current_group)
                            current_group = [(m_idx, m_label)]
                    groups.append(current_group)

                    # לכל רצף — שמירת features מהחלון שלפניו
                    trigger_data = load_trigger_labeled()
                    for group in groups:
                        first_idx = group[0][0]
                        group_label = group[0][1]
                        preceding_idx = first_idx - 1
                        if preceding_idx >= 0 and preceding_idx < len(LAST_ANALYSIS_CHUNKS):
                            trigger_feat = extract_features(LAST_ANALYSIS_CHUNKS[preceding_idx])
                            trigger_y = 1 if group_label == 'multiple_lesson' else 0
                            trigger_data['features'].append(trigger_feat.tolist())
                            trigger_data['labels'].append(trigger_y)
                            trigger_added += 1
                    if trigger_added > 0:
                        save_trigger_labeled(trigger_data)

                save_msg = f"נשמרו {added} דוגמאות ({ctx_added} הקשר, {trigger_added} trigger)"
            else:
                save_msg = "אין תיקונים חדשים"

            # --- שלב 2: אימון מחדש ---
            train_messages = []
            labeled = load_labeled_data()
            if len(labeled['labels']) >= 4:
                accuracy, msg = retrain_model(labeled)
                train_messages.append(msg)
            else:
                train_messages.append(f"מודל דוברים: צריך לפחות 4 דוגמאות (יש {len(labeled['labels'])})")

            ctx_labeled = load_context_labeled()
            if len(ctx_labeled['labels']) >= 2:
                ctx_acc, ctx_msg = retrain_context_model(ctx_labeled)
                train_messages.append(ctx_msg)
            else:
                train_messages.append(f"מודל הקשר: צריך לפחות 2 דוגמאות (יש {len(ctx_labeled['labels'])})")

            # אימון מודל trigger (חלון קודם → חיזוי הקשר ריבוי)
            trigger_data = load_trigger_labeled()
            if len(trigger_data['labels']) >= 2:
                trig_acc, trig_msg = retrain_trigger_model(trigger_data)
                train_messages.append(trig_msg)
            else:
                train_messages.append(f"מודל trigger: צריך לפחות 2 דוגמאות (יש {len(trigger_data['labels'])})")

            global PIPELINE
            PIPELINE = None
            get_pipeline()

            self._send_json(200, {
                'save_msg': save_msg,
                'train_msg': ' | '.join(train_messages),
            })

        except Exception as e:
            self._send_json(500, {'error': str(e)})

    # --- ניתוח מחדש של אותו קובץ ---
    def _handle_reanalyze(self):
        global LAST_ANALYSIS_CHUNKS, LAST_RAW_AUDIO, LAST_CHUNK_OFFSETS, LAST_AUDIO_PATH
        try:
            if not LAST_AUDIO_PATH or not os.path.exists(LAST_AUDIO_PATH):
                self._send_json(400, {'error': 'אין קובץ לנתח מחדש. העלי קובץ קודם.'})
                return

            pipeline = get_pipeline()

            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                results = pipeline.process_file(LAST_AUDIO_PATH)
            finally:
                sys.stdout = old_stdout

            # עדכון chunks
            audio = pipeline.load_audio(LAST_AUDIO_PATH)
            clean = pipeline.wiener.apply(audio)
            window_samples = int(pipeline.sr * 3.0)
            LAST_ANALYSIS_CHUNKS = []
            LAST_RAW_AUDIO = audio
            LAST_CHUNK_OFFSETS = []
            for i in range(0, len(clean), window_samples):
                chunk = clean[i:i + window_samples]
                if len(chunk) >= pipeline.sr:
                    LAST_ANALYSIS_CHUNKS.append(chunk)
                    LAST_CHUNK_OFFSETS.append(i)

            windows_data = []
            for idx, w in enumerate(results['windows']):
                label = w['attention_label']
                windows_data.append({
                    'idx': idx,
                    'start': w['start_sec'],
                    'end': w['end_sec'],
                    'speech_ratio': round(w['speech_ratio'] * 100, 1),
                    'rms_level': w['rms_level'],
                    'speaker_type': w['speaker_type'],
                    'context': w['context_category'],
                    'label': label if label else '\u05e8\u05e2\u05e9',
                    'is_noise': label is None,
                })

            labeled = load_labeled_data()
            ls = results['lesson_score']

            n_single = sum(1 for w in windows_data if w['speaker_type'] == 'דובר_יחיד' and w['label'] == 'חיובי')
            n_multi_lesson = sum(1 for w in windows_data if not w['is_noise'] and w['speaker_type'] == 'ריבוי_דוברים' and w['label'] == 'חיובי')
            n_disruption = sum(1 for w in windows_data if w['label'] == 'הפרעה')
            n_noise = sum(1 for w in windows_data if w['is_noise'])
            total = len(windows_data)

            response = {
                'duration_sec': round(results['duration_sec'], 1),
                'duration_min': round(results['duration_sec'] / 60, 1),
                'windows': windows_data,
                'positive_pct': ls['positive_pct'],
                'negative_pct': ls['negative_pct'],
                'noise_count': ls['noise_count'],
                'total_relevant': ls['total_relevant'],
                'grade': ls['grade'],
                'labeled_count': len(labeled['labels']),
                'breakdown': {
                    'single': n_single,
                    'multi_lesson': n_multi_lesson,
                    'disruption': n_disruption,
                    'noise': n_noise,
                    'total': total,
                },
            }
            self._send_json(200, response)

        except Exception as e:
            self._send_json(500, {'error': str(e)})

    # --- סטטיסטיקות נתוני אימון ---
    def _handle_stats(self):
        labeled = load_labeled_data()
        n_single = sum(1 for l in labeled['labels'] if l == 0)
        n_multi = sum(1 for l in labeled['labels'] if l == 1)
        n_noise = sum(1 for l in labeled['labels'] if l == 2)
        ctx = load_context_labeled()
        n_ctx_lesson = sum(1 for l in ctx['labels'] if l == 1)
        n_ctx_disrupt = sum(1 for l in ctx['labels'] if l == 0)
        self._send_json(200, {
            'total': len(labeled['labels']),
            'single_count': n_single,
            'multi_count': n_multi,
            'noise_count': n_noise,
            'ctx_lesson': n_ctx_lesson,
            'ctx_disruption': n_ctx_disrupt,
        })

    def _serve_html(self):
        html_path = os.path.join(os.path.dirname(__file__), 'demo.html')
        if not os.path.exists(html_path):
            self.send_error(404, 'demo.html not found')
            return
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))

    def _send_json(self, code, data):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def log_message(self, format, *args):
        print(f"  [{args[0]}] {args[1]}")


def main():
    print(f"\n{'='*50}")
    print(f"  Quality Learning — דמו + למידה מקוונת")
    print(f"{'='*50}")
    print(f"  טוען מודלים...")
    get_pipeline()
    labeled = load_labeled_data()
    print(f"  [V] נתוני אימון: {len(labeled['labels'])} דוגמאות")
    print(f"  [V] מוכן!")
    print(f"\n  פתחי בדפדפן: http://localhost:{PORT}")
    print(f"  להפסקה: Ctrl+C")
    print(f"{'='*50}\n")

    server = HTTPServer(('0.0.0.0', PORT), DemoHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  [X] השרת נעצר.")
        server.server_close()


if __name__ == '__main__':
    main()


# ===================================================
# DISABLED: הקוד מעל אינו פעיל
# ===================================================
raise SystemExit(f"[DISABLED] web_demo.py — גרסה ישנה, אינו פעיל")
