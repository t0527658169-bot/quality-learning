# ===================================================
# Dockerfile — Audio Processor (Python)
# ===================================================
FROM python:3.11-slim

WORKDIR /app

# התקנת ספריות מערכת לעיבוד שמע
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# התקנת dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# העתקת קוד המקור
COPY . .

EXPOSE 5000

# הפעלת FastAPI דרך uvicorn
CMD ["python", "-m", "uvicorn", "web_api:app", "--host", "0.0.0.0", "--port", "5000"]
