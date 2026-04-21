# Quality Learning - Classroom Attention Monitoring System

## Project Overview

Quality Learning is an advanced microservices-based system that performs **automatic analysis of classroom audio recordings** to assess student engagement and classroom dynamics. The system integrates acoustic signal processing, machine learning, and natural language processing to provide objective, data-driven insights into teaching and learning quality.

### Key Features

✅ **Real-time Audio Analysis** - Processes audio streams from classroom microphones
✅ **Speaker Detection** - Distinguishes between single speaker and multiple speaker scenarios
✅ **Attention Level Classification** - Categorizes classroom states as:
   - *1. Single Speaker* (typically high attention)
   - *2. Multiple Speakers (Noise)* (potential disruption)
   - *3. Multiple Speakers (Active Learning)* (discussion/group work)

✅ **Contextual Analysis** - Uses NLP to determine if multiple speakers indicate pedagogically appropriate activity
✅ **Statistical Reporting** - Tracks engagement metrics over time with trend analysis
✅ **Teacher Dashboard** - Real-time visualization with historical reports and comparative analytics

---

## System Architecture

The system follows a **microservices architecture** with the following components:

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend Dashboard                  │
│              (Real-time visualization & Reports)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│              API Gateway (Node.js/Express)                   │
│         (Authentication, Routing, Request Handling)          │
└────────┬─────────────┬──────────────┬──────────────┬────────┘
         │             │              │              │
    ┌────▼─┐     ┌────▼──────┐  ┌────▼────┐  ┌──────▼─────┐
    │Audio │     │ Audio      │  │Data     │  │PostgreSQL  │
    │Receiver   │ Processor  │  │Service  │  │Database    │
    │(C#)  │   │(Python)    │  │(Node.js)│  │            │
    └──────┘     └───────────┘  └─────────┘  └────────────┘
```

### Service Details

| Service | Technology | Port | Responsibility |
|---------|-----------|------|-----------------|
| **API Gateway** | Node.js/Express | 3000 | Route management, auth, logging |
| **Audio Receiver** | C#/.NET | 5002 | Capture audio, buffer, queue |
| **Audio Processor** | Python/FastAPI | 5000 | Core algorithms, signal processing |
| **Data Service** | Node.js/Express | 5001 | Database operations, reports |
| **Frontend** | React/TypeScript | 3001 | Teacher dashboard UI |
| **Database** | PostgreSQL | 5432 | Persistent data storage |

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Git
- 4GB+ RAM available
- Windows/MacOS/Linux with command line access

### Setup Instructions

1. **Clone and navigate to project:**
   ```bash
   cd pro
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings if needed
   ```

3. **Start all services:**
   ```bash
   docker-compose up -d
   ```

4. **Wait for services to initialize (30-60 seconds):**
   ```bash
   docker-compose logs -f
   ```

5. **Access the system:**
   - **Frontend Dashboard:** http://localhost:3001
   - **API Gateway:** http://localhost:3000
   - **API Documentation:** http://localhost:3000/api-docs (if Swagger enabled)

### Stop Services

```bash
docker-compose down
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f audio-processor
```

---

## Project Structure

```
pro/
├── services/
│   ├── api-gateway/              # Node.js API Gateway service
│   │   ├── src/
│   │   ├── package.json
│   │   └── Dockerfile
│   ├── audio-processor/          # Python audio processing service
│   │   ├── src/
│   │   │   ├── algorithms/       # Core algorithms (Wiener, VAD, RMS, etc)
│   │   │   ├── models/           # ML models (classifiers, ASR integration)
│   │   │   ├── utils/            # Helper functions
│   │   │   └── app.py            # FastAPI application
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── audio-receiver/           # C# audio capture service
│   │   ├── src/
│   │   ├── AudioReceiver.csproj
│   │   └── Dockerfile
│   └── data-service/             # Node.js data management service
│       ├── src/
│       ├── package.json
│       └── Dockerfile
├── frontend/
│   └── quality-learning-dashboard/  # React TypeScript dashboard
│       ├── src/
│       │   ├── components/
│       │   ├── pages/
│       │   ├── services/
│       │   └── App.tsx
│       ├── package.json
│       └── Dockerfile
├── database/
│   ├── migrations/
│   │   └── 001_initial_schema.sql
│   └── seeds/
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Core Algorithms Implemented

### 1. **Wiener Filter** (Noise Reduction)
- Reduces environmental noise while preserving speech
- Adapts to signal and noise characteristics
- Foundation for reliable subsequent processing

### 2. **Voice Activity Detection (VAD)**
- Distinguishes between silence, speech, and non-speech noise
- Uses energy thresholding and spectral analysis
- Enables focus on relevant audio periods

### 3. **RMS Energy Calculation**
- Measures average sound intensity over time windows
- Provides objective, stable noise level measurements
- Used for real-time visualization

### 4. **Speaker Diarization**
- Identifies single vs. multiple speaker scenarios
- Detects overlapping speech patterns
- Foundation for context-aware classification

### 5. **Spectral Analysis**
- Analyzes frequency distribution (Spectral Centroid, Flatness)
- Helps distinguish speech from mechanical noise
- Improves acoustic classifier accuracy

### 6. **Contextual NLP Analysis**
- Converts speech to text (ASR - Whisper API)
- Analyzes semantic content to identify pedagogical intent
- Determines if multiple speakers indicate:
  - Appropriate discussion/group work
  - Unwanted disruption/noise

### 7. **Decision Tree Classifier**
- Integrates acoustic and contextual signals
- Produces final attention level classification
- Learns patterns from historical data

---

## Database Schema

### Main Tables

**teachers** - Teacher records
```sql
id, email, password_hash, full_name, created_at
```

**classes** - Classroom information
```sql
id, teacher_id, class_name, grade_level, created_at
```

**sessions** - Recording sessions
```sql
id, class_id, start_time, end_time, file_path, status
```

**audio_segments** - Time-windowed analysis results
```sql
id, session_id, start_time, end_time, 
rms_energy, classification, confidence, speaker_count
```

**daily_reports** - Aggregated daily statistics
```sql
id, class_id, date, avg_attention_score, 
noise_ratio, discussion_ratio
```

---

## Development Guidelines

### Code Standards

- **Python:** PEP 8 style, type hints, docstrings on all functions
- **JavaScript/TypeScript:** ESLint config, TSC strict mode
- **C#:** Microsoft C# coding guidelines, XML documentation comments
- **All:** Inline comments explaining algorithm steps and business logic

### Important Implementation Notes

⚠️ **Contextual Analysis Lookback:**
When multiple speakers are detected, the system performs **retrospective analysis** of preceding audio windows (configurable, default: 2-3 minutes) to determine if the activity was initiated by legitimate pedagogical direction (teacher announcement of discussion/group work).

**Algorithm Flow:**
1. Detect multiple speakers (acoustic VAD + diarization)
2. Flag potential disruption
3. Convert preceding audio to text (ASR)
4. Analyze for keywords: "discuss", "group work", "collaborate", etc.
5. Classify as either "Active Learning" or "Noise" based on linguistic evidence
6. Update classification for the entire window

---

## API Endpoints

### Audio Processing
```
POST /api/audio/process
  - Upload audio file or stream for analysis
  - Returns: Segment classifications

GET /api/audio/status/{sessionId}
  - Check processing status
```

### Reporting
```
GET /api/reports/daily/{classId}/{date}
  - Retrieve daily attention metrics

GET /api/reports/trends/{classId}?days=30
  - Get trend analysis over time
```

### Authentication
```
POST /api/auth/login
  - Teacher login with credentials

POST /api/auth/logout
  - End session
```

---

## Configuration

### Audio Processing Parameters

Edit in `services/audio-processor/src/config.py`:

```python
# Window duration for analysis (seconds)
SEGMENT_DURATION = 2

# RMS energy threshold for speech detection
RMS_THRESHOLD = 50

# Lookback window for contextual analysis (seconds)
CONTEXT_LOOKBACK = 180  # 3 minutes

# Confidence threshold for classifications
CONFIDENCE_MIN = 0.7
```

### Database Configuration

Edit `.env`:
```
DB_USER=qualitylearning
DB_PASSWORD=<secure-password>
DB_NAME=quality_learning
```

---

## Troubleshooting

### Services won't start
```bash
# Check Docker daemon is running
docker ps

# View detailed logs
docker-compose logs audio-processor

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Database connection errors
```bash
# Check PostgreSQL is healthy
docker-compose ps

# View database logs
docker-compose logs postgres

# Reset database
docker-compose exec postgres psql -U qualitylearning -d quality_learning -c "\dt"
```

### Audio not processing
- Verify microphone/audio file accessible to audio-receiver service
- Check audio format is supported (WAV, MP3)
- View audio-processor logs for algorithm errors

---

## Future Enhancements

🚀 **Planned Features:**
- Real-time WebSocket updates for live dashboards
- Mobile app for teachers
- Predictive models for at-risk classes
- Integration with Learning Management Systems (LMS)
- Advanced emotion detection from speech tone
- Integration with classroom IoT devices (doors, windows)
- Automatic teacher feedback generation
- Anonymous student participation metrics

---

## Security Considerations

🔐 **Implemented:**
- HTTPS/TLS for all inter-service communication
- Password hashing (bcrypt) for teacher credentials
- Role-based access control (Teacher, Admin)
- Database encryption for sensitive fields
- API rate limiting and validation

⚠️ **Production Checklist:**
- [ ] Change default database password
- [ ] Enable HTTPS with valid certificates
- [ ] Configure firewall rules
- [ ] Enable database backups
- [ ] Set up monitoring and alerting
- [ ] Review security logs regularly

---

## Support & Documentation

- **API Documentation:** Check `/api-docs` endpoint
- **Algorithm Details:** See `.md` files in each service directory
- **Database Schema:** View `database/migrations/` directory

---

## License

This project is developed for educational purposes.

---

## Contact

For questions or issues, please refer to the project documentation in each service directory.

**Happy Teaching and Learning! 📚**
