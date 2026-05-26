# view_training_data.py - הצגת נתוני האימון שנצברו

import pickle
import os
import numpy as np

# נתיבים
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
LABELED_DATA_PATH = os.path.join(MODELS_DIR, "labeled_data.pkl")
CONTEXT_LABELED_PATH = os.path.join(MODELS_DIR, "context_labeled_data.pkl")
TRIGGER_LABELED_PATH = os.path.join(MODELS_DIR, "trigger_labeled_data.pkl")

def load_pkl(path):
    """טעינת קובץ pickle"""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def main():
    print(f"\n{'='*60}")
    print(f"  נתוני אימון שנצברו מתיקוני המשתמש")
    print(f"{'='*60}\n")
    
    # 1. נתוני דוברים
    labeled = load_pkl(LABELED_DATA_PATH)
    if labeled:
        features = np.array(labeled['features'])
        labels = np.array(labeled['labels'])
        
        n_single = int(np.sum(labels == 0))
        n_multi = int(np.sum(labels == 1))
        n_noise = int(np.sum(labels == 2))
        
        print(f"📊 1. נתוני דוברים (labeled_data.pkl)")
        print(f"   סה\"כ דגימות: {len(labels)}")
        print(f"   - דובר יחיד (0): {n_single} דגימות")
        print(f"   - ריבוי דוברים (1): {n_multi} דגימות")
        print(f"   - רעש (2): {n_noise} דגימות")
        print(f"   גודל features: {features.shape}")
        
        # הצגת 3 דגימות ראשונות
        print(f"\n   דוגמאות:")
        for i in range(min(3, len(labels))):
            label_name = {0: 'דובר יחיד', 1: 'ריבוי', 2: 'רעש'}[labels[i]]
            print(f"   [{i+1}] {label_name}: {features[i]}")
    else:
        print(f"❌ 1. labeled_data.pkl לא נמצא")
    
    print(f"\n{'-'*60}\n")
    
    # 2. נתוני הקשר
    ctx_labeled = load_pkl(CONTEXT_LABELED_PATH)
    if ctx_labeled:
        features = np.array(ctx_labeled['features'])
        labels = np.array(ctx_labeled['labels'])
        
        n_lesson = int(np.sum(labels == 1))
        n_disrupt = int(np.sum(labels == 0))
        
        print(f"📊 2. נתוני הקשר (context_labeled_data.pkl)")
        print(f"   סה\"כ דגימות: {len(labels)}")
        print(f"   - לצורך השיעור (1): {n_lesson} דגימות")
        print(f"   - הפרעה (0): {n_disrupt} דגימות")
        print(f"   גודל features: {features.shape}")
        
        # הצגת 3 דגימות ראשונות
        print(f"\n   דוגמאות:")
        for i in range(min(3, len(labels))):
            label_name = {0: 'הפרעה', 1: 'לצורך השיעור'}[labels[i]]
            print(f"   [{i+1}] {label_name}: {features[i]}")
    else:
        print(f"❌ 2. context_labeled_data.pkl לא נמצא")
    
    print(f"\n{'-'*60}\n")
    
    # 3. נתוני trigger
    trigger_labeled = load_pkl(TRIGGER_LABELED_PATH)
    if trigger_labeled:
        features = np.array(trigger_labeled['features'])
        labels = np.array(trigger_labeled['labels'])
        
        n_lesson = int(np.sum(labels == 1))
        n_disrupt = int(np.sum(labels == 0))
        
        print(f"📊 3. נתוני trigger (trigger_labeled_data.pkl)")
        print(f"   סה\"כ דגימות: {len(labels)}")
        print(f"   - מוביל ללצורך השיעור (1): {n_lesson} דגימות")
        print(f"   - מוביל להפרעה (0): {n_disrupt} דגימות")
        print(f"   גודל features: {features.shape}")
        
        # הצגת 3 דגימות ראשונות
        print(f"\n   דוגמאות:")
        for i in range(min(3, len(labels))):
            label_name = {0: 'מוביל להפרעה', 1: 'מוביל ללצורך השיעור'}[labels[i]]
            print(f"   [{i+1}] {label_name}: {features[i]}")
    else:
        print(f"❌ 3. trigger_labeled_data.pkl לא נמצא")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()


