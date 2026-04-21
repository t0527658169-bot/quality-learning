# ===================================================
# train_context_model.py — אימון מודל לסיווג הקשר פדגוגי
# ===================================================
# מודל: Logistic Regression + TF-IDF
#
# למה Logistic Regression?
#   - שקוף לחלוטין: ניתן לראות את המשקל של כל מילה
#   - מתמטית פשוט: P(y=1|x) = sigmoid(w·x + b)
#   - יעיל: מאומן ב-שניות גם על אלפי דגימות
#   - פרשני: מילים עם משקל חיובי = מעידות על למידה, שלילי = הפרעה
#
# TF-IDF (Term Frequency × Inverse Document Frequency):
#   - ממיר טקסט למספרים — כל מילה מקבלת ציון חשיבות
#   - מילה שמופיעה הרבה במסמך אחד אבל לא בכל המסמכים = חשובה יותר
#
# קטגוריות:
#   0 = "הפרעה"       — רעש, בלגן ללא הקשר לימודי
#   1 = "למידה_פעילה" — דיון, שאלות, עבודה משותפת
#   2 = "פתיחה_לדיון" — המורה פותח/ת דיון מכוון
#
# הרצה: python train_context_model.py

import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- נתיב שמירה ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "context_lr_model.pkl")


# ===================================================
# שלב 1: דאטהסט אימון — משפטים לדוגמה
# ===================================================
# כל משפט מתויג עם הקטגוריה שלו.
# בפרויקט אמיתי — נאסוף נתונים מכיתות אמיתיות.
# כאן יצרנו דגימות מציאותיות ידנית לאימון ראשוני.

TRAINING_DATA = [
    # --- קטגוריה 0: הפרעה ---
    ("מספיק שקט בבקשה", 0),
    ("תפסיקו לדבר", 0),
    ("אני לא שומע כלום", 0),
    ("יש פה יותר מדי רעש", 0),
    ("שקט שקט שקט", 0),
    ("מי זרק את זה", 0),
    ("תחזיר לי את העט", 0),
    ("למה אתה צועק", 0),
    ("תפסיקו להפריע", 0),
    ("אני רוצה הפסקה", 0),
    ("מתי ההפסקה", 0),
    ("אני רעב", 0),
    ("בואו נלך החוצה", 0),
    ("זה משעמם", 0),
    ("לא מעניין אותי", 0),
    ("תני לי את הטלפון", 0),
    ("אני רוצה הביתה", 0),
    ("מי דיבר עכשיו בלי רשות", 0),
    ("אל תיגע בזה", 0),
    ("למה הוא מכה אותי", 0),
    ("אני לא רוצה לעבוד", 0),
    ("חחחח צחוק", 0),
    ("בלה בלה בלה", 0),
    ("נו באמת", 0),
    ("כבר נמאס לי", 0),
    ("אפשר ללכת לשירותים", 0),
    ("מה קורה", 0),
    ("אני לא מבין כלום", 0),
    ("תעזוב אותי", 0),
    ("זה לא פייר", 0),
    ("שקט כולם", 0),
    ("ילדים הרגעו", 0),
    ("בחזרה למקומות", 0),
    ("מי שמדבר יוצא החוצה", 0),
    ("אסור לדבר עכשיו", 0),
    ("זה לא הזמן לשיחות", 0),
    ("תירגעו כולכם", 0),
    ("צריך שקט בכיתה", 0),
    ("אני לא שומעת את עצמי", 0),
    ("רגע אחד בלי רעש", 0),

    # --- קטגוריה 1: למידה פעילה ---
    ("בואו נעבוד בקבוצות על התרגיל", 1),
    ("כל קבוצה תדון בשאלה", 1),
    ("תענו על השאלות בזוגות", 1),
    ("עבדו יחד על המשימה", 1),
    ("שתפו את הקבוצה שלכם", 1),
    ("כתבו את התשובה בצוות", 1),
    ("תתרגלו את זה ביחד", 1),
    ("עבודת קבוצה עכשיו", 1),
    ("חשבו ביחד על הפתרון", 1),
    ("דברו בינכם על התוצאה", 1),
    ("נעשה עבודת צוותים", 1),
    ("כל זוג יכתוב תשובה", 1),
    ("שתפו מה מצאתם", 1),
    ("הסבירו אחד לשני", 1),
    ("עזרו אחד לשני בתרגיל", 1),
    ("תנסו לפתור את זה ביחד", 1),
    ("כל קבוצה מציגה את התשובה", 1),
    ("עבדו על הדף ביחד", 1),
    ("תרגול בזוגות", 1),
    ("נחלק לקבוצות של ארבע", 1),
    ("המשימה היא משותפת", 1),
    ("תעבדו בצמדים", 1),
    ("תדונו בנושא ותכתבו מסקנות", 1),
    ("כל קבוצה מכינה תשובה", 1),
    ("פתרו את התרגיל בצוות", 1),
    ("הקבוצות ישבו ביחד", 1),
    ("כתבו סיכום ביחד", 1),
    ("חשבו על דוגמאות", 1),
    ("נעשה פעילות משותפת", 1),
    ("מי רוצה לענות", 1),

    # --- קטגוריה 2: פתיחה לדיון ---
    ("בואו נפתח דיון בנושא", 2),
    ("מה דעתכם על הנושא הזה", 2),
    ("אני רוצה לשמוע מה אתם חושבים", 2),
    ("מי יכול להסביר למה זה קורה", 2),
    ("יש למישהו רעיון", 2),
    ("בואו נחשוב ביחד על זה", 2),
    ("מה אתם חושבים על מה שקראנו", 2),
    ("למה לדעתכם זה המצב", 2),
    ("מי מסכים ומי לא מסכים", 2),
    ("בואו נדון בשאלה הבאה", 2),
    ("נפתח את הדיון", 2),
    ("אני שואל שאלה פתוחה", 2),
    ("מה הייתם עושים במקרה כזה", 2),
    ("חשבו על היתרונות והחסרונות", 2),
    ("מי רוצה לחלוק את דעתו", 2),
    ("בואו נדבר על מה שלמדנו", 2),
    ("איזה מסקנות אפשר להסיק", 2),
    ("מה דעתכם האם זה נכון", 2),
    ("נשמע דעות שונות", 2),
    ("מה למדנו מהניסוי הזה", 2),
    ("הגיע הזמן לדיון בכיתה", 2),
    ("פתחו דיון על הנושא", 2),
    ("מה הייתם משנים פה", 2),
    ("למה בחר המחבר במילים האלה", 2),
    ("מה אפשר ללמוד מהסיפור", 2),
    ("נשמע עוד דעות בנושא", 2),
    ("בואו נעלה רעיונות", 2),
    ("האם יש גישה אחרת", 2),
    ("אני מזמין דיון בנושא", 2),
    ("מה דעתכם על ההצעה הזאת", 2),
]


# ===================================================
# שלב 2: הרחבת הדאטהסט (Data Augmentation)
# ===================================================
def _augment_data(data: list, multiplier: int = 5) -> list:
    """
    הרחבת הנתונים — יצירת גרסאות שונות של כל משפט.
    שיטות:
      - הסרת מילה אקראית
      - שכפול מילה אקראית
      - שינוי סדר המילים
    זה עוזר למודל ללמוד לזהות תבניות גם כשהניסוח מעט שונה.
    """
    augmented = list(data)  # מתחילים עם המקור

    for text, label in data:
        words = text.split()
        if len(words) < 2:
            continue

        for _ in range(multiplier):
            method = np.random.choice(['drop', 'duplicate', 'shuffle'])

            if method == 'drop' and len(words) > 2:
                # הסרת מילה אקראית
                idx = np.random.randint(0, len(words))
                new_words = words[:idx] + words[idx+1:]
            elif method == 'duplicate':
                # שכפול מילה אקראית
                idx = np.random.randint(0, len(words))
                new_words = words[:idx] + [words[idx]] + words[idx:]
            else:
                # ערבוב חלקי (החלפת 2 מילים)
                new_words = list(words)
                if len(new_words) > 2:
                    i, j = np.random.choice(len(new_words), 2, replace=False)
                    new_words[i], new_words[j] = new_words[j], new_words[i]

            augmented.append((" ".join(new_words), label))

    return augmented


# ===================================================
# שלב 3: אימון המודל
# ===================================================
def train_model():
    """
    אימון מודל Logistic Regression + TF-IDF לסיווג הקשר פדגוגי.

    TF-IDF ממיר טקסט למספרים → Logistic Regression מסווג.
    """
    print(f"\n{'='*50}")
    print(f"  אימון מודל סיווג הקשרי (Logistic Regression)")
    print(f"{'='*50}")

    # --- הכנת דאטהסט ---
    print(f"\n  [1/5] דאטהסט מקורי: {len(TRAINING_DATA)} דגימות")
    augmented = _augment_data(TRAINING_DATA, multiplier=8)
    print(f"  [1/5] לאחר הרחבה:  {len(augmented)} דגימות")

    texts = [t for t, _ in augmented]
    labels = np.array([l for _, l in augmented])

    # --- TF-IDF: המרת טקסט למספרים ---
    # כל מילה מקבלת ציון חשיבות לפי:
    #   TF (שכיחות במשפט) × IDF (נדירות בכלל המשפטים)
    print(f"\n  [2/5] חילוץ מאפיינים (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),    # מילים בודדות + צמדי מילים ("עבודת קבוצות")
        max_features=500,       # עד 500 מאפיינים (מילים/צמדים)
        sublinear_tf=True       # log(TF) — החלקה לוגריתמית
    )
    X = vectorizer.fit_transform(texts)
    print(f"  [2/5] מאפיינים: {X.shape[1]} מילים/צמדי-מילים")

    # --- חלוקה לאימון ומבחן ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\n  [3/5] אימון: {X_train.shape[0]} | מבחן: {X_test.shape[0]}")

    # --- אימון Logistic Regression ---
    print(f"\n  [4/5] אימון Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,         # מקסימום 1000 איטרציות
        C=1.0,                 # פרמטר רגולריזציה (מונע overfitting)
        class_weight='balanced',  # איזון — נותן משקל שווה לכל מחלקה
        random_state=42
    )
    model.fit(X_train, y_train)

    # --- הערכה ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    target_names = ['הפרעה', 'למידה_פעילה', 'פתיחה_לדיון']
    print(f"\n  דיוק כולל: {acc:.1%}")
    print(f"\n{classification_report(y_test, y_pred, target_names=target_names)}")

    # --- מילות המפתח של המודל (שקיפות מלאה) ---
    print(f"  המילים הכי משפיעות לכל קטגוריה:")
    feature_names = vectorizer.get_feature_names_out()
    for i, cat in enumerate(target_names):
        coefs = model.coef_[i]
        top_idx = np.argsort(coefs)[-8:][::-1]  # 8 המילים הכי חזקות
        top_words = [(feature_names[j], coefs[j]) for j in top_idx]
        print(f"\n    {cat}:")
        for word, score in top_words:
            bar = "█" * int(abs(score) * 10)
            print(f"      {word:<20s} {score:+.3f} {bar}")

    # --- שמירה ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            'model': model,
            'vectorizer': vectorizer,
            'label_names': target_names,
            'accuracy': acc,
            'description': 'Logistic Regression + TF-IDF for pedagogical context classification'
        }, f)
    print(f"\n  [5/5] מודל נשמר: {MODEL_PATH}")
    print(f"{'='*50}\n")

    return model, vectorizer


if __name__ == "__main__":
    train_model()
