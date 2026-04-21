# ===================================================
# wiener_filter.py — אלגוריתם פילטר וינר לניקוי רעשים
# ===================================================
# פילטר וינר מפחית רעש מאות השמע תוך שמירה על הדיבור.
# הוא עובד בתחום התדר: מעריך את "טביעת האצבע" של הרעש
# ואז מפחית את הרעש מכל תדר בנפרד.
#
# נוסחה: H(f) = max(0, 1 - Noise(f) / Signal(f))
# כאשר H(f) = מקדם הסינון בכל תדר

import numpy as np
from scipy.signal import stft, istft
from config import SAMPLE_RATE


class WienerFilter:
    """מסנן וינר לניקוי רעשי רקע מהקלטות כיתתיות."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

    # ------- הערכת פרופיל הרעש -------
    def _estimate_noise(self, audio: np.ndarray, noise_duration: float = 0.5) -> np.ndarray:
        """
        הערכת הרעש מתוך החצי שנייה הראשונה של ההקלטה.
        ההנחה: בתחילת ההקלטה עדיין לא מדברים, אז מה שנשמע הוא רעש רקע.
        """
        # חיתוך קטע הרעש מתחילת ההקלטה
        noise_samples = int(noise_duration * self.sample_rate)
        noise_segment = audio[:max(noise_samples, 1)]

        # חישוב ספקטרוגרמה (מעבר לתחום התדר) של קטע הרעש
        _, _, noise_stft = stft(noise_segment, fs=self.sample_rate, nperseg=512)

        # ממוצע ההספק על פני הזמן — זה "טביעת האצבע" של הרעש
        noise_profile = np.mean(np.abs(noise_stft) ** 2, axis=1)
        return noise_profile

    # ------- הפעלת הפילטר -------
    def apply(self, audio: np.ndarray) -> np.ndarray:
        """
        הפעלת פילטר וינר על אות השמע המלא.
        שלבים:
          1. מעריך את פרופיל הרעש מתחילת ההקלטה
          2. מחשב ספקטרוגרמה של כל האות
          3. מחשב מקדם סינון לכל תדר (כמה לדכא)
          4. מפעיל את המקדם וממיר חזרה לאות זמן
        """
        # שלב 1: הערכת רעש הרקע
        noise_profile = self._estimate_noise(audio)

        # שלב 2: ספקטרוגרמה של האות המלא (מעבר מזמן לתדר)
        freqs, times, audio_stft = stft(audio, fs=self.sample_rate, nperseg=512)
        audio_power = np.abs(audio_stft) ** 2

        # שלב 3: חישוב מקדם הסינון של וינר בכל תדר
        # אם ההספק באות גדול מהרעש — שומרים; אחרת — מדכאים
        noise_expanded = noise_profile[:, np.newaxis]
        wiener_gain = np.maximum(0.0, 1.0 - noise_expanded / (audio_power + 1e-10))

        # שלב 4: הפעלת המסנן על הספקטרוגרמה והמרה חזרה לאות זמן
        cleaned_stft = audio_stft * wiener_gain
        _, cleaned_audio = istft(cleaned_stft, fs=self.sample_rate, nperseg=512)

        # התאמת אורך למקור (istft עלול להוסיף דגימות)
        return cleaned_audio[:len(audio)]
