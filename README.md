# 🧠 MindGuard — Mental Health Early Warning System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

**MindGuard** is an AI-powered Mental Health Early Warning System designed to detect psychological distress and provide timely interventions. Built for the **IIT BHU Hackathon 2025**, it leverages advanced NLP, machine learning, and real-time monitoring to identify individuals at risk of depression, anxiety, and crisis situations.

## 🎯 Key Features

### 🤖 AI-Powered Risk Detection
- **Multi-dimensional analysis**: Detects depression, anxiety, and crisis risk scores
- **NLP-driven insights**: Contextual understanding of user emotional state
- **Real-time scoring**: Instant risk assessment with detailed breakdowns

### 🌍 Multi-Language Support
- **Indian language support**: English, Hindi, Tamil, Telugu
- **Language auto-detection**: Automatic identification of input language
- **Localized lexicon analysis**: Language-specific emotional keyword detection

### 🏥 Smart Hospital Locator
- **Location-based search**: Find nearby mental health facilities using OpenStreetMap
- **Clinic details**: Hospital names, addresses, contact information, and doctor profiles
- **Distance calculation**: Haversine formula for accurate distance estimation
- **Fallback mechanism**: Gemini API integration for enhanced accuracy

### 📊 Patient Dashboard
- **Historical tracking**: Visualize emotional trends over time
- **Session management**: Track multiple check-in sessions per patient
- **Analytics**: Risk score trends, feature analysis, and progress monitoring

### 🔔 Real-Time Alerts
- **WebSocket support**: Live alert system for critical risk detection
- **Crisis management**: Immediate intervention alerts for high-risk scenarios
- **Notification system**: Alerts to healthcare providers and support networks

### 🔐 Privacy-First Design
- **In-memory data storage**: No persistent user data storage by default
- **HIPAA-compliant architecture**: Designed with healthcare privacy standards
- **Secure WebSocket connections**: Encrypted real-time communications

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.9+)
- **AI Engine**: Google Generative AI (Gemini 2.5 Flash)
- **APIs**:
  - OpenStreetMap Nominatim (Geocoding)
  - Overpass API (Location-based amenity search)
- **WebSockets**: Real-time bidirectional communication
- **Server**: Uvicorn ASGI

### Frontend
- **Markup**: HTML5
- **Styling**: CSS3 (Clay Design System)
- **Visualization**: 
  - Three.js (3D backgrounds)
  - Leaflet.js (Interactive maps)
  - GSAP (Animations)
- **HTTP Client**: JavaScript Fetch API

## 📦 Installation

### Prerequisites
```bash
Python 3.9+
pip (Python package manager)
```

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mindguard.git
cd mindguard
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install fastapi uvicorn google-generativeai httpx pydantic python-multipart
```

4. **Set up environment variables**
Create a `.env` file in the root directory:
```
GOOGLE_API_KEY=your_google_api_key_here
```

**Note**: You can obtain a free Google API key from [Google AI Studio](https://aistudio.google.com/)

5. **Start the backend server**
```bash
python backend_bhu.py
```

The server will prompt you to enter a port (default: 8000). Once running:
- **API Docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8000/api/v1/dashboard

### Frontend Setup

1. **Open the HTML file**
Simply open `index_bhu.html` in your web browser:
```bash
# macOS
open index_bhu.html

# Windows
start index_bhu.html

# Linux
xdg-open index_bhu.html
```

2. **Connect to backend**
When prompted, enter the backend port (default: 8000)

## 📡 API Endpoints

### Patient Management

#### Get All Patients
```
GET /api/v1/patients
```
Returns list of all registered patients with basic information.

#### Get Patient Details
```
GET /api/v1/patients/{patient_id}
```
Retrieves comprehensive patient data including historical sessions.

### Session Management

#### Create New Session
```
POST /api/v1/session/{patient_id}
Content-Type: application/json

{
  "text": "User's emotional input",
  "language": "en"
}
```

Response:
```json
{
  "session_id": "P001_S01",
  "patient_id": "P001",
  "timestamp": "2025-01-10T10:30:00",
  "scores": {
    "depression": 0.45,
    "anxiety": 0.32,
    "crisis": 0.08
  },
  "analysis": "Detailed NLP analysis...",
  "recommendations": "Suggested interventions..."
}
```

#### Get Patient History
```
GET /api/v1/patients/{patient_id}/history?limit=10
```
Retrieves session history with pagination support.

### Analytics

#### Dashboard Data
```
GET /api/v1/dashboard
```
Returns aggregated statistics and patient overview.

#### Risk Trends
```
GET /api/v1/trends/{patient_id}?days=30
```
Analyzes risk score trends over specified period.

### Location Services

#### Find Hospitals
```
GET /api/v1/hospitals?location=Varanasi
```

Response:
```json
{
  "location": "Varanasi, Uttar Pradesh, India",
  "hospitals": [
    {
      "name": "City Hospital",
      "address": "123 Medical Road",
      "distance_km": 2.5,
      "contact": "+91 9876543210",
      "lat": 25.32,
      "lng": 82.98,
      "amenity_type": "Hospital",
      "doctors": [
        {
          "name": "Dr. Sharma",
          "specialty": "Psychiatrist",
          "experience": "12 yrs",
          "availability": "Available Today"
        }
      ]
    }
  ]
}
```

### Real-Time Alerts

#### WebSocket Connection
```
WS /ws/alerts
```

Messages:
- **Connected**: `{"type": "connected", "message": "..."}`
- **Heartbeat**: Send `ping` to receive `{"type": "pong"}`
- **Alert**: `{"type": "alert", "severity": "high", "message": "..."}`

## 🧪 Risk Scoring Algorithm

### Depression Score
Calculated based on:
- Hopelessness indicators (weight: 0.4)
- Sleep disturbances (weight: 0.2)
- Negation density (weight: 0.2)
- Positive sentiment ratio (weight: 0.2)

**Range**: 0.0 - 1.0
- **0.0-0.3**: Low risk
- **0.3-0.6**: Moderate risk
- **0.6-1.0**: High risk

### Anxiety Score
Calculated based on:
- Anxiety lexicon matches (weight: 0.4)
- Excessive worry indicators (weight: 0.3)
- Sleep disturbances (weight: 0.2)
- Negation patterns (weight: 0.1)

**Range**: 0.0 - 1.0

### Crisis Score
Calculated based on:
- Suicidal ideation keywords (weight: 0.5)
- Self-harm indicators (weight: 0.3)
- Hopelessness + isolation (weight: 0.2)

**Range**: 0.0 - 1.0
- **> 0.15**: Critical - Immediate intervention required

## 🌐 Multi-Language Support

Supported languages with native lexicon analysis:
- **English** (en): Full feature set
- **Hindi** (hi): Native Devanagari script support
- **Tamil** (ta): Native Tamil script support
- **Telugu** (te): Native Telugu script support

Language detection uses Unicode ranges:
```python
Hindi:    U+0900 to U+097F
Tamil:    U+0B80 to U+0BFF
Telugu:   U+0C00 to U+0C7F
```

## 🗂️ Project Structure

```
mindguard/
├── backend_bhu.py           # Main FastAPI backend
├── index_bhu.html           # Frontend application
├── models.txt               # Supported AI models reference
├── response.json            # Sample API responses
├── response_new.json        # Updated response format
├── response_utf8.json       # UTF-8 encoded responses
└── .vscode/
    └── settings.json        # VS Code configuration
```

## 🔄 Workflow

1. **User Input**: Patient enters emotional text in native language
2. **Language Detection**: System identifies input language
3. **Feature Extraction**: NLP extracts risk indicators
4. **Risk Scoring**: Algorithm computes depression, anxiety, crisis scores
5. **AI Insights**: Gemini API provides contextual analysis
6. **Recommendations**: System suggests interventions and resources
7. **Hospital Locator**: Finds nearby mental health facilities
8. **Alert System**: WebSocket notifies healthcare providers of critical cases

## 📊 Sample Use Cases

### Case 1: Depression Detection
```
Input: "I feel so hopeless lately. Nothing seems to have meaning anymore. 
        I can't sleep and just feel empty all the time."

Output:
- Depression Score: 0.72 (High Risk)
- Anxiety Score: 0.35 (Moderate)
- Crisis Score: 0.08 (Low)
- Recommendation: "Please reach out to a mental health professional"
```

### Case 2: Anxiety Management
```
Input: "मुझे हमेशा चिंता रहती है। काम पर ध्यान नहीं लगता।
        दिल तेज़ी से धड़कता है।"

Output:
- Depression Score: 0.28 (Low)
- Anxiety Score: 0.68 (High)
- Crisis Score: 0.02 (Low)
- Recommendation: "Breathing exercises and professional therapy recommended"
```

### Case 3: Crisis Detection
```
Input: "I can't take it anymore. Nobody cares. I just want to disappear."

Output:
- Depression Score: 0.81 (Very High)
- Anxiety Score: 0.45 (Moderate)
- Crisis Score: 0.28 (Critical)
- Action: ⚠️ CRITICAL ALERT - Immediate intervention required
```

## ⚙️ Configuration

### Customize Risk Thresholds
Modify in `backend_bhu.py`:
```python
DEPRESSION_THRESHOLD = 0.60  # High risk threshold
ANXIETY_THRESHOLD = 0.65
CRISIS_THRESHOLD = 0.15      # Critical threshold
```

### Adjust Lexicon Weights
Update sentiment weight distributions:
```python
hopelessness_weight = 0.4
anxiety_weight = 0.4
positive_sentiment_weight = 0.2
```

### Configure Hospital Search Radius
Default: 5000 meters (5 km)
```python
search_radius = 10000  # 10 km radius
```

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Write/update tests**
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

## 📋 Future Enhancements

- [ ] Database integration (PostgreSQL/MongoDB)
- [ ] User authentication and authorization
- [ ] Advanced visualization dashboards
- [ ] Mobile app (iOS/Android)
- [ ] Video-based emotional analysis
- [ ] Integration with wearable devices
- [ ] Predictive modeling with historical data
- [ ] Multi-provider therapy matching
- [ ] Community support features
- [ ] Appointment scheduling system

## 🐛 Known Limitations

- **No persistent storage**: Uses in-memory database (resets on restart)
- **API dependencies**: Requires active internet for Gemini API and OpenStreetMap
- **Rate limits**: Google API has usage quotas
- **Language coverage**: Limited to 4 Indian languages + English
- **Mock data**: Some hospital information is randomly generated

## 🔒 Security Considerations

⚠️ **Production Deployment Warning**:
- Replace hardcoded API keys with environment variables
- Implement proper authentication (JWT, OAuth 2.0)
- Use HTTPS/TLS for all connections
- Add rate limiting and DDoS protection
- Implement HIPAA compliance
- Set up proper logging and monitoring
- Use database encryption for sensitive data

## 📝 Environment Setup (Production)

```bash
# Install with production dependencies
pip install -r requirements.txt

# Set environment variables
export GOOGLE_API_KEY="your_production_key"
export DATABASE_URL="postgresql://user:pass@host/db"
export SECRET_KEY="your_secret_key"

# Run with Gunicorn (production server)
gunicorn -w 4 -b 0.0.0.0:8000 backend_bhu:app
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IIT BHU Hackathon 2025** for the opportunity
- **Google Generative AI** for Gemini API
- **OpenStreetMap** community for map data
- **FastAPI** framework for backend architecture
- **Leaflet.js** and **Three.js** for frontend visualization

## 📞 Support

For issues, questions, or suggestions:
- **GitHub Issues**: [Report a bug](https://github.com/yourusername/mindguard/issues)
- **Discussions**: [Start a discussion](https://github.com/yourusername/mindguard/discussions)
- **Email**: mindguard.support@example.com

## 🚀 Quick Start (30 seconds)

```bash
# 1. Install dependencies
pip install fastapi uvicorn google-generativeai httpx

# 2. Get API key
# Visit https://aistudio.google.com/ and copy your API key

# 3. Update backend_bhu.py with your API key (line 20)
genai.configure(api_key="YOUR_API_KEY_HERE")

# 4. Run backend
python backend_bhu.py
# Choose port (press Enter for 8000)

# 5. Open frontend
# Open index_bhu.html in your browser
# Enter port 8000 when prompted

# 6. Start using MindGuard!
```

---

**Made with ❤️ for mental health awareness and early intervention**

*Remember: MindGuard is a supportive tool, not a replacement for professional mental health care. Always consult qualified healthcare providers for serious concerns.*
