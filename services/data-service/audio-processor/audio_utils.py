# ===================================================
# audio_utils.py — DISABLED (data-service — גרסה ישנה)
# *** קובץ זה שייך לגרסה הישנה לפני HEBERT ***
# *** ניתן למחוק את תיקיית data-service/audio-processor כולה ***
# ===================================================

# ===================================================
# audio_utils.py — פונקציות עזר לעיבוד שמע
# ===================================================
# מודול זה מספק את כל חישובי האודיו הבסיסיים שצריך,
# ללא תלות בספריות חיצוניות כבדות (כמו librosa).
# משתמש רק ב-numpy ו-scipy שהם יציבים ואמינים.

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window
from config import SAMPLE_RATE


# ===============================================
# חלוקה לפריימים
# ===============================================
def frame_audio(audio: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """
    חיתוך האות למטריצת פריימים (חלונות קצרים).
    כל שורה = פריים אחד של frame_len דגימות.
    המרחק בין פריימים = hop_len דגימות.
    """
    num_frames = 1 + (len(audio) - frame_len) // hop_len
    indices = np.arange(frame_len)[None, :] + hop_len * np.arange(num_frames)[:, None]
    return audio[indices]


# ===============================================
# חישוב RMS (אנרגיה) לכל פריים
# ===============================================
def compute_rms(frames: np.ndarray) -> np.ndarray:
    """RMS = √(ממוצע(x²)) — עוצמת הצליל בכל פריים."""
    return np.sqrt(np.mean(frames ** 2, axis=1))


# ===============================================
# חישוב קצב חציית אפס (ZCR)
# ===============================================
def compute_zcr(frames: np.ndarray) -> np.ndarray:
    """
    ZCR — כמה פעמים הגל חוצה את ציר ה-0 בכל פריים.
    ערך גבוה = רעש אקראי, ערך נמוך = צליל מובנה (דיבור).
    """
    signs = np.sign(frames)
    # ספירת שינויי סימן (+ ↔ -)
    crossings = np.abs(np.diff(signs, axis=1))
    return np.mean(crossings > 0, axis=1)


# ===============================================
# חישוב ספקטרום (FFT) לפריימים
# ===============================================
def compute_spectrum(frames: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """
    חישוב ספקטרום הספק לכל פריים באמצעות FFT.
    מחזיר: (magnitude, frequencies)
    """
    # הפעלת חלון Hann למניעת דליפה ספקטרלית
    window = get_window('hann', frames.shape[1])
    windowed = frames * window
    # FFT — מעבר מתחום הזמן לתחום התדר
    spectrum = np.abs(rfft(windowed, axis=1))
    freqs = rfftfreq(frames.shape[1], d=1.0 / sample_rate)
    return spectrum, freqs


# ===============================================
# מרכז ספקטרלי (Spectral Centroid)
# ===============================================
def compute_spectral_centroid(spectrum: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    מרכז הכובד של התדרים — "בהירות" הצליל.
    נוסחה: centroid = Σ(f × |X(f)|) / Σ(|X(f)|)
    """
    total_energy = np.sum(spectrum, axis=1) + 1e-10
    centroid = np.sum(spectrum * freqs, axis=1) / total_energy
    return centroid


# ===============================================
# שטחיות ספקטרלית (Spectral Flatness)
# ===============================================
def compute_spectral_flatness(spectrum: np.ndarray) -> np.ndarray:
    """
    מדד "שטחיות" הספקטרום.
    ממוצע גיאומטרי / ממוצע אריתמטי.
    1.0 = רעש לבן (שטוח), 0.0 = צליל טהור (שיא בודד).
    """
    # ממוצע גיאומטרי — משתמשים ב-log כדי להימנע מ-overflow
    log_spectrum = np.log(spectrum + 1e-10)
    geo_mean = np.exp(np.mean(log_spectrum, axis=1))
    arith_mean = np.mean(spectrum, axis=1) + 1e-10
    return geo_mean / arith_mean


# ===============================================
# רוחב פס ספקטרלי (Spectral Bandwidth)
# ===============================================
def compute_spectral_bandwidth(spectrum: np.ndarray, freqs: np.ndarray,
                                centroid: np.ndarray) -> np.ndarray:
    """
    רוחב הפס — כמה "רחב" הצליל בתדרים.
    נוסחה: bw = √( Σ(|X(f)| × (f - centroid)²) / Σ(|X(f)|) )
    """
    total = np.sum(spectrum, axis=1) + 1e-10
    deviation = (freqs - centroid[:, None]) ** 2
    bw = np.sqrt(np.sum(spectrum * deviation, axis=1) / total)
    return bw


# ===============================================
# MFCC — Mel-Frequency Cepstral Coefficients
# ===============================================
def _hz_to_mel(hz):
    """המרה מ-Hz ל-Mel (סולם לוגריתמי שמחקה שמיעה אנושית)."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    """המרה מ-Mel ל-Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(num_filters: int, fft_size: int, sample_rate: int) -> np.ndarray:
    """
    בניית בנק מסננים בסולם Mel.
    יוצר מסננים משולשים בתדרים שמחקים את האוזן האנושית.
    """
    low_mel = _hz_to_mel(0)
    high_mel = _hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = _mel_to_hz(mel_points)

    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    n_fft_bins = fft_size // 2 + 1
    filterbank = np.zeros((num_filters, n_fft_bins))

    for i in range(num_filters):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        # משולש עולה
        for j in range(left, center):
            if j < n_fft_bins:
                filterbank[i, j] = (j - left) / (center - left + 1e-10)
        # משולש יורד
        for j in range(center, right):
            if j < n_fft_bins:
                filterbank[i, j] = (right - j) / (right - center + 1e-10)

    return filterbank


def compute_mfcc(audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
                  n_mfcc: int = 13, n_filters: int = 26) -> np.ndarray:
    """
    חישוב 13 מקדמי MFCC (Mel-Frequency Cepstral Coefficients).
    שלבים:
      1. FFT — מעבר לתחום התדר
      2. בנק מסננים Mel — מחקה שמיעה אנושית
      3. Log — דחיסה לוגריתמית
      4. DCT — מעבר קוסינוס בדיד → המקדמים הסופיים

    מחזיר ממוצע MFCC על פני כל הפריימים = וקטור אחד לכל קטע.
    """
    from scipy.fft import dct

    # חלוקה לפריימים
    frame_len = int(0.025 * sample_rate)
    hop_len = int(0.010 * sample_rate)
    frames = frame_audio(audio, frame_len, hop_len)

    # FFT עם חלון Hann
    window = get_window('hann', frame_len)
    windowed = frames * window
    power_spectrum = np.abs(rfft(windowed, axis=1)) ** 2

    # בנק מסננים Mel
    filterbank = _mel_filterbank(n_filters, frame_len, sample_rate)
    mel_spectrum = np.dot(power_spectrum, filterbank.T)

    # Log
    mel_log = np.log(mel_spectrum + 1e-10)

    # DCT — שמירה על n_mfcc מקדמים ראשונים
    mfcc_frames = dct(mel_log, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    # ממוצע על פני הזמן
    return np.mean(mfcc_frames, axis=0)


# ===================================================
# DISABLED: הקוד מעל אינו פעיל
# ===================================================
raise SystemExit(f"[DISABLED] audio_utils.py — גרסה ישנה, אינו פעיל")
