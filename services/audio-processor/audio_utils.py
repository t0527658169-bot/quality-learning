# ===================================================
# audio_utils.py — פונקציות עזר לעיבוד שמע
# ===================================================
# מודול זה מספק את כל חישובי האודיו הבסיסיים שצריך,
# ללא תלות בספריות חיצוניות כבדות (כמו librosa).
# משתמש רק ב-numpy ו-scipy שהם יציבים ואמינים.

# יייבוא numpy לחישובים על מערכי דגימות
import numpy as np
# rfft = Fast Fourier Transform לצד החיובי בלבד (מספיק לאותות אמיתיים)
# rfftfreq = חישוב התדרים המתאימים לתוצאת rfft
from scipy.fft import rfft, rfftfreq
# get_window = יצירת חלוני חלקה (Hann, Hamming וכו') למניעת דליפה ספקטרלית
from scipy.signal import get_window
# קצב הדגימה הגלובלי של המערכת
# from config import SAMPLE_RATE
from config import SAMPLE_RATE, MFCC_N_COEFFICIENTS, MFCC_N_FILTERS


# ===============================================
# חלוקה לפריימים
# ===============================================
def frame_audio(audio: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """
    חיתוך האות למטריצת פריימים (חלונות קצרים).
    כל שורה = פריים אחד של frame_len דגימות.
    המרחק בין פריימים = hop_len דגימות.
    """
    # חישוב כמה פריימים יהיו בסך הכל
    # 1 + ... = פריים ראשון תמיד קיים; // = חלוקה שלמה (ללא שארית)
    num_frames = 1 + (len(audio) - frame_len) // hop_len
    # בניית מטריצת אינדקסים: כל שורה = אינדקסי פריים אחד
    # np.arange(frame_len)[None, :] = וקטור 0..frame_len-1 כשורה יחידה
    # hop_len * np.arange(num_frames)[:, None] = הזזה של כל פריים
    # הסכום נותן מטריצה num_frames × frame_len של אינדקסים
    indices = np.arange(frame_len)[None, :] + hop_len * np.arange(num_frames)[:, None]
    # שליפת הדגימות עצמן בעזרת מטריצת האינדקסים
    return audio[indices]


# ===============================================
# חישוב RMS (אנרגיה) לכל פריים
# ===============================================
def compute_rms(frames: np.ndarray) -> np.ndarray:
    """RMS = √(ממוצע(x²)) — עוצמת הצליל בכל פריים."""
    # frames**2 = ריבוע כל דגימה, np.mean(..., axis=1) = ממוצע לכל שורה (פריים)
    # np.sqrt = שורש — נותן RMS לכל פריים
    return np.sqrt(np.mean(frames ** 2, axis=1))


# ===============================================
# חישוב קצב חציית אפס (ZCR)
# ===============================================
def compute_zcr(frames: np.ndarray) -> np.ndarray:
    """
    ZCR — כמה פעמים הגל חוצה את ציר ה-0 בכל פריים.
    ערך גבוה = רעש אקראי, ערך נמוך = צליל מובנה (דיבור).
    """
    # np.sign(frames) = +1 לדגימות חיוביות, -1 לשליליות, 0 לאפס
    signs = np.sign(frames)
    # np.diff(signs, axis=1) = הפרש בין דגימות סמוכות לאורך הפריים
    # np.abs() > 0 = True בכל מקום שיש שינוי סימן (חציית אפס)
    # ספירת שינויי סימן (+ ↔ -)
    crossings = np.abs(np.diff(signs, axis=1))
    # np.mean(..., axis=1) = אחוז החציות בכל פריים (0.0 עד 1.0)
    return np.mean(crossings > 0, axis=1)


# ===============================================
# חישוב ספקטרום (FFT) לפריימים
# ===============================================
def compute_spectrum(frames: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """
    חישוב ספקטרום הספק לכל פריים באמצעות FFT.
    מחזיר: (magnitude, frequencies)
    """
    # הפעלת חלון Hann — פונקציה שמאפסת את קצות הפריים כדי למנוע
    # דליפה ספקטרלית (artifacts מחיתוך חד של האות)
    window = get_window('hann', frames.shape[1])
    # הכפלת כל פריים בחלון Hann
    windowed = frames * window
    # FFT — מעבר מתחום הזמן לתחום התדר
    # rfft מחזיר רק תדרים חיוביים (מספיק לאותות אמיתיים)
    # np.abs() לוקח גודל (מודולוס) של מספרים מרוכבים
    spectrum = np.abs(rfft(windowed, axis=1))
    # rfftfreq מחשב את התדרים (בHz) המתאימים לכל בין FFT
    # d=1/sample_rate = מרחק בין דגימות בשניות
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
    # סכום האנרגיה בכל התדרים לכל פריים (+1e-10 מונע חלוקה באפס)
    total_energy = np.sum(spectrum, axis=1) + 1e-10
    # ממוצע משוקלל של התדרים — תדרים חזקים יותר משפיעים יותר
    # spectrum * freqs = כל בין מוכפל בתדר שלו
    # np.sum(..., axis=1) = סכום על כל התדרים לכל פריים
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
    # log(spectrum+1e-10) מונע log(0)
    log_spectrum = np.log(spectrum + 1e-10)
    # ממוצע הלוגריתמים ואז exp = ממוצע גיאומטרי
    geo_mean = np.exp(np.mean(log_spectrum, axis=1))
    # ממוצע אריתמטי רגיל (+1e-10 מונע חלוקה באפס)
    arith_mean = np.mean(spectrum, axis=1) + 1e-10
    # יחס גיאומטרי/אריתמטי: 1.0=שטוח לחלוטין, קרוב ל-0=שיא בודד
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
    # סכום האנרגיה הכולל בכל פריים
    total = np.sum(spectrum, axis=1) + 1e-10
    # ריבוע המרחק של כל תדר ממרכז הספקטרום
    # centroid[:, None] מרחיב וקטור 1D לעמודה כדי לאפשר חיסור מטריצה
    deviation = (freqs - centroid[:, None]) ** 2
    # שורש של ממוצע משוקלל — דומה לסטיית תקן של התדרים
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


# def compute_mfcc(audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
#                   n_mfcc: int = 13, n_filters: int = 26) -> np.ndarray:
def compute_mfcc(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, 
                 n_mfcc: int = MFCC_N_COEFFICIENTS, n_filters: int = MFCC_N_FILTERS) -> np.ndarray:
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
