import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from main_pipeline import AudioPipeline

test_file = r"C:\Users\school\AppData\Local\Temp\tmp6oxbae9p.wav"
print("Testing:", test_file)

p = AudioPipeline()
r = p.process_file(test_file)

print("\n=== WINDOW DETAILS ===")
for w in r['windows']:
    sr = w['speech_ratio']
    sp = w['speaker_type']
    rm = w['rms_level']
    lb = w['attention_label']
    print(f"  speech={sr:.2f}  speaker={sp}  rms={rm}  label={lb}")

ls = r['lesson_score']
print(f"\npos={ls['positive_pct']}%  neg={ls['negative_pct']}%  noise={ls['noise_count']}  relevant={ls['total_relevant']}")
