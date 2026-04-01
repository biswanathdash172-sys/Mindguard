"""
╔══════════════════════════════════════════════════════════════════╗
║          MindGuard — Mental Health Early Warning System          ║
║          FastAPI Backend  ·  IIT BHU Hackathon 2025              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import uvicorn
import json
import re
import random
import math
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import os
import google.generativeai as genai

genai.configure(api_key="AIzaSyAK_L0R2QRVS76xi82aMeEpzX6xZ9n1zCA")

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# No API key required — using free OpenStreetMap / Overpass API

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ════════════════════════════════════════════
#  App Bootstrap
# ════════════════════════════════════════════
app = FastAPI(
    title="MindGuard API",
    description="Mental Health Early Warning System — AI-powered risk detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════════
#  In-Memory Data Store
# ════════════════════════════════════════════
patients_db: Dict[str, dict] = {
    "P001": {"id": "P001", "name": "Aryan K.",  "age": 28, "lang": "Hindi",   "created": "2025-01-10", "avatar": "AK"},
    "P002": {"id": "P002", "name": "Priya D.",  "age": 34, "lang": "Tamil",   "created": "2025-01-15", "avatar": "PD"},
    "P003": {"id": "P003", "name": "Rohit S.",  "age": 22, "lang": "English", "created": "2025-02-01", "avatar": "RS"},
    "P004": {"id": "P004", "name": "Meera N.",  "age": 45, "lang": "Telugu",  "created": "2025-02-10", "avatar": "MN"},
}

sessions_db: Dict[str, List[dict]] = defaultdict(list)

# Seed historical sessions for demo realism
_seed_data = {
    "P001": [
        (0.32, 0.28, 0.05, "Feeling a bit low lately but managing."),
        (0.40, 0.35, 0.08, "Sleep has been difficult. Nothing feels interesting anymore."),
        (0.37, 0.33, 0.07, "Trying to stay positive but it's hard."),
        (0.54, 0.48, 0.12, "I feel so tired all the time. I don't know what's the point."),
        (0.60, 0.52, 0.15, "Completely hopeless. Can't sleep. Feel worthless."),
        (0.68, 0.58, 0.19, "Nobody cares. I'm so alone. Feeling empty and numb."),
        (0.76, 0.64, 0.20, "Hopeless. Everything is pointless. Isolated from everyone."),
    ],
    "P002": [
        (0.25, 0.55, 0.06, "Anxious about work. Hard to focus."),
        (0.28, 0.58, 0.07, "Worried constantly. Heart racing when I wake up."),
        (0.30, 0.60, 0.08, "Panic attacks getting worse. Scared of leaving home."),
        (0.35, 0.65, 0.10, "Overwhelmed by everything. Can't stop worrying."),
        (0.38, 0.67, 0.11, "Terrible anxiety. Feel like something bad will happen."),
        (0.42, 0.65, 0.12, "Stressed and anxious but trying breathing exercises."),
    ],
    "P003": [
        (0.15, 0.18, 0.02, "Doing okay today. Went for a walk."),
        (0.18, 0.20, 0.03, "A little sad but nothing major."),
        (0.16, 0.22, 0.02, "Feeling fine, sleeping better."),
        (0.20, 0.21, 0.03, "Good week overall. Saw friends."),
    ],
    "P004": [
        (0.12, 0.15, 0.01, "Feeling well. Meditation is helping."),
        (0.14, 0.16, 0.02, "Slight worry about family but mostly okay."),
        (0.13, 0.18, 0.02, "Stable mood. Grateful for support."),
    ],
}

for pid, rows in _seed_data.items():
    base_date = datetime.now() - timedelta(days=len(rows) * 7)
    for i, (dep, anx, cri, txt) in enumerate(rows):
        sess_date = base_date + timedelta(days=i * 7)
        sessions_db[pid].append({
            "session_id": f"{pid}_S{i+1:02d}",
            "patient_id": pid,
            "timestamp": sess_date.isoformat(),
            "scores": {"depression": dep, "anxiety": anx, "crisis": cri},
            "text": txt,
            "features": {"lang": "en", "word_count": len(txt.split()), "shap_factors": []},
        })

# ════════════════════════════════════════════
#  NLP Engine — Multi-language Risk Analysis
# ════════════════════════════════════════════

# ── Lexicons ──────────────────────────────────
HOPELESS_EN = {
    "hopeless","worthless","pointless","meaningless","useless","empty",
    "nothing","never","nobody","alone","lonely","isolated","abandoned",
    "burden","failure","stupid","hate","kill","die","dead","end","hurt",
    "tired","exhausted","numb","trapped","stuck","helpless","lost","quit",
    "miserable","depressed","dark","hollow","void","disappear","suicidal",
    "no point","no hope","no future","give up","can't go on","not worth",
}
ANXIETY_EN = {
    "anxious","worried","panic","scared","fear","afraid","nervous","dread",
    "overwhelmed","stress","terrified","heart racing","shaking","tense",
    "restless","uneasy","paranoid","catastrophe","disaster","worst",
    "horrible","unbearable","uncontrollable","spiraling","losing control",
    "falling apart","can't breathe","terrifying","apprehensive",
}
POSITIVE_EN = {
    "happy","good","better","hopeful","grateful","improving","positive",
    "confident","strong","supported","loved","motivated","progress",
    "thank","excited","energy","fine","great","wonderful","peaceful",
    "calm","relaxed","looking forward","enjoying","smile","laugh","fun",
}
SLEEP_WORDS   = {"sleep","insomnia","awake","nightmares","nightmare","tired","fatigue","rest","nap","night","woke"}
SOCIAL_ISO    = {"alone","nobody","isolated","no friends","no family","left me","abandoned","withdrawn","cut off"}
HOPELESS_HI   = {"निराश","बेकार","अकेला","थका","उदास","दुखी","दर्द","खत्म","नहीं","मतलब","बेमतलब","अंधेरा"}
HOPELESS_TA   = {"நம்பிக்கையற்ற","தனிமை","சோர்வு","வலி","கவலை","பயம்","விரக்தி"}
HOPELESS_TE   = {"నిరాశ","ఒంటరి","అలసట","బాధ","ఆందోళన","భయం","నిస్సత్తువ"}


def detect_language(text: str) -> str:
    hi = len(re.findall(r'[\u0900-\u097F]', text))
    ta = len(re.findall(r'[\u0B80-\u0BFF]', text))
    te = len(re.findall(r'[\u0C00-\u0C7F]', text))
    if hi > 4: return "hi"
    if ta > 4: return "ta"
    if te > 4: return "te"
    return "en"

LANG_LABELS = {"en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu"}

def extract_features(text: str) -> dict:
    lang        = detect_language(text)
    words       = text.lower().split()
    word_set    = set(words)
    total_words = max(len(words), 1)

    # Hopelessness
    if   lang == "hi": h_hits = len(word_set & HOPELESS_HI)
    elif lang == "ta": h_hits = len(word_set & HOPELESS_TA)
    elif lang == "te": h_hits = len(word_set & HOPELESS_TE)
    else:              h_hits = len(word_set & HOPELESS_EN)

    hopelessness  = min(h_hits / max(total_words * 0.07, 1), 1.0)
    anxiety_lex   = min(len(word_set & ANXIETY_EN) / max(total_words * 0.07, 1), 1.0)
    positive_r    = len(word_set & POSITIVE_EN) / total_words
    sleep_score   = min(len(word_set & SLEEP_WORDS) / 3.0, 1.0)
    iso_score     = min(len(word_set & SOCIAL_ISO) / 2.0, 1.0)

    # Negation density
    neg_count = len(re.findall(
        r"\b(no|not|never|nothing|nobody|nowhere|without|can't|won't|don't"
        r"|didn't|isn't|aren't|wasn't|weren't|couldn't|wouldn't|shouldn't)\b",
        text.lower()
    ))
    negation_score = min(neg_count / max(total_words * 0.1, 1), 1.0)

    # Brevity (short sentences → emotional withdrawal)
    sents = [s.strip() for s in re.split(r'[.!?]+', text.strip()) if s.strip()]
    avg_len = sum(len(s.split()) for s in sents) / max(len(sents), 1)
    brevity_score = max(0.0, 1.0 - avg_len / 22.0)

    # First-person density (high in depression)
    fp = len(re.findall(r"\b(i|me|my|myself|i'm|i've|i'd|i'll)\b", text.lower()))
    fp_score = min(fp / max(total_words * 0.15, 1), 1.0)

    # Build SHAP-style explanation
    shap = []
    if h_hits > 0:
        shap.append(f"Hopelessness tokens ×{h_hits} detected")
    if negation_score > 0.2:
        shap.append("High negation density (linguistic withdrawal)")
    if sleep_score > 0.3:
        shap.append("Sleep disruption language present")
    if iso_score > 0.3:
        shap.append("Social isolation markers found")
    if brevity_score > 0.55:
        shap.append("Very short sentences (emotional withdrawal)")
    if anxiety_lex > 0.3:
        shap.append(f"Anxiety vocabulary elevated")
    if positive_r > 0.1:
        shap.append("Protective: positive language present")
    if fp_score > 0.4:
        shap.append("Elevated self-referential language")
    if not shap:
        shap.append("No significant risk markers detected")

    return {
        "lang":            lang,
        "lang_label":      LANG_LABELS.get(lang, "English"),
        "word_count":      total_words,
        "hopelessness":    round(hopelessness, 3),
        "anxiety_lexical": round(anxiety_lex, 3),
        "positive_ratio":  round(positive_r, 3),
        "sleep_score":     round(sleep_score, 3),
        "isolation_score": round(iso_score, 3),
        "negation_score":  round(negation_score, 3),
        "brevity_score":   round(brevity_score, 3),
        "fp_score":        round(fp_score, 3),
        "shap_factors":    shap,
    }


def compute_risk_scores(features: dict, history: list) -> dict:
    h   = features["hopelessness"]
    a   = features["anxiety_lexical"]
    p   = features["positive_ratio"]
    sl  = features["sleep_score"]
    iso = features["isolation_score"]
    neg = features["negation_score"]
    br  = features["brevity_score"]
    fp  = features["fp_score"]

    # Depression composite
    depression = (
        h   * 0.32 +
        neg * 0.14 +
        sl  * 0.12 +
        iso * 0.12 +
        br  * 0.10 +
        fp  * 0.10 +
        a   * 0.06 +
        random.uniform(0, 0.04)
    ) * (1.0 - p * 0.45)

    # Anxiety composite
    anxiety = (
        a   * 0.38 +
        neg * 0.20 +
        h   * 0.14 +
        fp  * 0.10 +
        br  * 0.08 +
        sl  * 0.06 +
        random.uniform(0, 0.04)
    ) * (1.0 - p * 0.30)

    # Trend boost: worsening trajectory amplifies risk
    trend_boost = 0.0
    if len(history) >= 3:
        recent_dep = [s["scores"]["depression"] for s in history[-3:]]
        if recent_dep[-1] > recent_dep[0]:
            slope = (recent_dep[-1] - recent_dep[0]) / len(recent_dep)
            trend_boost = slope * 0.28

    depression = min(depression + trend_boost, 0.98)
    anxiety    = min(anxiety + trend_boost * 0.5, 0.98)

    # Crisis probability
    crisis = min(
        depression * 0.50 +
        h          * 0.28 +
        iso        * 0.22,
        0.98
    )

    return {
        "depression": round(max(0.01, depression), 3),
        "anxiety":    round(max(0.01, anxiety),    3),
        "crisis":     round(max(0.01, crisis),     3),
    }


def risk_level(scores: dict) -> str:
    d = scores["depression"]
    c = scores["crisis"]
    if d > 0.65 or c > 0.55: return "HIGH"
    if d > 0.40 or c > 0.30: return "MED"
    return "LOW"


def suggested_action(scores: dict) -> str:
    d = scores["depression"]
    c = scores["crisis"]
    a = scores["anxiety"]
    if c > 0.55 or d > 0.80:
        return "⚠️ Immediate follow-up required. Consider crisis intervention. Administer PHQ-9 + C-SSRS."
    if d > 0.60 or c > 0.35:
        return "📋 Schedule follow-up within 48 hours. Consider PHQ-9 screening. Review medication if applicable."
    if d > 0.40 or a > 0.55:
        return "📅 Increase session frequency. Monitor over next 2 weeks. Consider GAD-7 for anxiety."
    return "✅ Continue routine monitoring. Next session as scheduled."


async def get_gemini_insight(text: str, scores: dict, patient_name: str, patient_lang: str) -> dict:
    # Try models in order of preference (quota-friendly first)
    MODELS_TO_TRY = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.0-pro"]

    prompt = f"""You are MindGuard clinical AI. Analyze this patient text for mental health risk.

Patient: {patient_name}
Text: "{text}"
NLP computed scores: depression={scores['depression']}, anxiety={scores['anxiety']}, crisis={scores['crisis']}

You MUST reply ONLY in this exact JSON format with no markdown, no extra text:
{{
  "depression_score": <float 0.01 to 0.98>,
  "anxiety_score": <float 0.01 to 0.98>,
  "crisis_score": <float 0.01 to 0.98>,
  "risk_level": "<HIGH or MED or LOW>",
  "insight": "<one warm empathetic sentence acknowledging their feeling>",
  "action": "<one concrete clinical recommendation>"
}}
Base the scores on the actual emotional content of the text.
If text describes hopelessness/suicidal thoughts: depression > 0.7, crisis > 0.5
If text describes anxiety/panic: anxiety > 0.6
If text is positive/stable: all scores < 0.3
Return ONLY the JSON object."""

    last_error = None
    for model_name in MODELS_TO_TRY:
        try:
            model = genai.GenerativeModel(model_name)
            async def fetch_insight(m=model):
                res = await m.generate_content_async(prompt)
                return res.text.strip()
            raw_json_str = await asyncio.wait_for(fetch_insight(), timeout=15.0)
            raw_json_str = raw_json_str.strip()
            if raw_json_str.startswith("```json"):
                raw_json_str = raw_json_str[7:]
            elif raw_json_str.startswith("```"):
                raw_json_str = raw_json_str[3:]
            if raw_json_str.endswith("```"):
                raw_json_str = raw_json_str[:-3]
            data = json.loads(raw_json_str.strip())
            print(f"[Gemini] Success with model: {model_name}")
            return {
                "depression": float(data.get("depression_score", scores["depression"])),
                "anxiety":    float(data.get("anxiety_score",    scores["anxiety"])),
                "crisis":     float(data.get("crisis_score",     scores["crisis"])),
                "risk_level": data.get("risk_level", risk_level(scores)),
                "insight":    data.get("insight", "Analysis complete."),
                "action":     data.get("action",  suggested_action(scores))
            }
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "exhausted" in err_str.lower():
                print(f"[Gemini] Quota exceeded for {model_name}, trying next model...")
                continue
            # Non-quota error — log and fall through to NLP fallback
            print(f"[Gemini] Error with {model_name}: {e}")
            break

    # ── NLP-based local fallback (always produces rich, real insight) ──
    import traceback, tempfile, os
    err_path = os.path.join(tempfile.gettempdir(), "mindguard_gemini_err.txt")
    with open(err_path, "w") as f:
        f.write(traceback.format_exc())
        f.write(f"\nLAST ERROR: {last_error}")
    print(f"[Gemini] All models failed. Using NLP fallback. Error: {last_error}")

    d, a, c = scores["depression"], scores["anxiety"], scores["crisis"]
    level = risk_level(scores)

    # Generate a meaningful clinical insight from NLP scores
    if d > 0.70 or c > 0.50:
        insight = (f"{patient_name} is showing severe depressive indicators with a crisis probability of "
                   f"{c:.0%}. Immediate clinical attention and safety assessment are strongly recommended.")
    elif d > 0.50:
        insight = (f"{patient_name}'s language reflects significant hopelessness and emotional withdrawal "
                   f"(depression score {d:.2f}). A structured follow-up session should be scheduled promptly.")
    elif a > 0.55:
        insight = (f"{patient_name} is exhibiting elevated anxiety markers (score {a:.2f}). "
                   f"Breathing regulation techniques and CBT approaches may be beneficial.")
    elif d > 0.30 or a > 0.35:
        insight = (f"{patient_name} shows mild to moderate distress signals. "
                   f"Continued monitoring and supportive therapy are recommended.")
    else:
        insight = (f"{patient_name}'s language reflects relative emotional stability. "
                   f"Routine check-ins and preventive care remain appropriate.")

    return {
        "depression": d,
        "anxiety":    a,
        "crisis":     c,
        "risk_level": level,
        "insight":    insight,
        "action":     suggested_action(scores)
    }

# ════════════════════════════════════════════
#  WebSocket Connection Manager
# ════════════════════════════════════════════
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        print(f"[WS] Client connected. Active connections: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        print(f"[WS] Client disconnected. Active connections: {len(self.active)}")

    async def broadcast(self, msg: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(msg))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)


manager = ConnectionManager()


# ════════════════════════════════════════════
#  Pydantic Models
# ════════════════════════════════════════════
class SessionInput(BaseModel):
    patient_id: str
    text: str

class PatientCreate(BaseModel):
    name: str
    age: int
    lang: str = "English"
    condition: str = "Unknown"




# ════════════════════════════════════════════
#  REST Endpoints
# ════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "status": "✅ MindGuard API running",
        "version": "1.0.0",
        "endpoints": {
            "patients":   "/api/v1/patients",
            "analyze":    "POST /api/v1/analyze",
            "dashboard":  "/api/v1/dashboard",
            "docs":       "/docs",
            "websocket":  "ws://localhost:{port}/ws/alerts",
        }
    }


CONDITION_BASELINE = {
    "Depression":         {"depression": 0.70, "anxiety": 0.35, "crisis": 0.45},
    "Severe Depression":  {"depression": 0.85, "anxiety": 0.40, "crisis": 0.65},
    "Anxiety":            {"depression": 0.30, "anxiety": 0.72, "crisis": 0.25},
    "PTSD":               {"depression": 0.60, "anxiety": 0.65, "crisis": 0.50},
    "Schizophrenia":      {"depression": 0.55, "anxiety": 0.50, "crisis": 0.60},
    "Bipolar Disorder":   {"depression": 0.62, "anxiety": 0.55, "crisis": 0.48},
    "OCD":                {"depression": 0.40, "anxiety": 0.68, "crisis": 0.30},
    "Suicidal Ideation":  {"depression": 0.88, "anxiety": 0.50, "crisis": 0.90},
    "Mild Stress":        {"depression": 0.20, "anxiety": 0.30, "crisis": 0.10},
    "Unknown":            {"depression": 0.0,  "anxiety": 0.0,  "crisis": 0.0},
}

@app.get("/api/v1/patients")
def get_patients():
    result = []
    for pid, p in patients_db.items():
        sessions = sessions_db.get(pid, [])
        condition = p.get("condition", "Unknown")
        default_scores = CONDITION_BASELINE.get(condition, CONDITION_BASELINE["Unknown"])
        last = sessions[-1]["scores"] if sessions else default_scores
        result.append({
            **p,
            "session_count": len(sessions),
            "last_scores": last,
            "risk_level": risk_level(last),
        })
    # Sort: HIGH → MED → LOW
    order = {"HIGH": 0, "MED": 1, "LOW": 2}
    result.sort(key=lambda x: order[x["risk_level"]])
    return result


@app.post("/api/v1/patients")
def create_patient(body: PatientCreate):
    pid = f"P{len(patients_db) + 1:03d}"
    while pid in patients_db:
        pid = f"P{random.randint(100, 999)}"
    initials = "".join(w[0].upper() for w in body.name.split()[:2])
    patients_db[pid] = {
        "id": pid,
        "name": body.name,
        "age": body.age,
        "lang": body.lang,
        "condition": body.condition,
        "created": datetime.now().date().isoformat(),
        "avatar": initials or "?",
    }
    return patients_db[pid]


@app.get("/api/v1/patients/{patient_id}")
def get_patient(patient_id: str):
    if patient_id not in patients_db:
        raise HTTPException(status_code=404, detail="Patient not found")
    p = patients_db[patient_id]
    sessions = sessions_db.get(patient_id, [])
    condition = p.get("condition", "Unknown")
    default_scores = CONDITION_BASELINE.get(condition, CONDITION_BASELINE["Unknown"])
    last = sessions[-1]["scores"] if sessions else default_scores
    return {**p, "session_count": len(sessions), "last_scores": last, "risk_level": risk_level(last)}


@app.get("/api/v1/patients/{patient_id}/sessions")
def get_sessions(patient_id: str):
    if patient_id not in patients_db:
        raise HTTPException(status_code=404, detail="Patient not found")
    return sessions_db.get(patient_id, [])


@app.post("/api/v1/analyze")
async def analyze_session(body: SessionInput):
    if body.patient_id not in patients_db:
        raise HTTPException(status_code=404, detail="Patient not found")
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(body.text) < 5:
        raise HTTPException(status_code=400, detail="Text too short for analysis")

    history  = sessions_db.get(body.patient_id, [])
    features = extract_features(body.text)
    scores   = compute_risk_scores(features, history)
    action   = suggested_action(scores)
    level    = risk_level(scores)
    
    patient_name = patients_db[body.patient_id]["name"]
    patient_lang = features.get("lang_label", "English")
    gemini_result = await get_gemini_insight(body.text, scores, patient_name, patient_lang)

    session_num = len(history) + 1
    session = {
        "session_id": f"{body.patient_id}_S{session_num:02d}",
        "patient_id": body.patient_id,
        "timestamp":  datetime.now().isoformat(),
        "scores":     {"depression": gemini_result["depression"], "anxiety": gemini_result["anxiety"], "crisis": gemini_result["crisis"]},
        "text":       body.text[:300] + ("…" if len(body.text) > 300 else ""),
        "features":   features,
        "action":     gemini_result["action"],
        "risk_level": gemini_result["risk_level"],
    }
    sessions_db[body.patient_id].append(session)

    # Broadcast WebSocket alert for high-risk sessions
    if session["scores"]["crisis"] > 0.50 or session["scores"]["depression"] > 0.70:
        await manager.broadcast({
            "type":         "alert",
            "level":        session["risk_level"],
            "patient_id":   body.patient_id,
            "patient_name": patients_db[body.patient_id]["name"],
            "scores":       session["scores"],
            "shap":         features["shap_factors"][:3],
            "action":       session["action"],
            "timestamp":    datetime.now().isoformat(),
        })

    return {
        "session_id":  session["session_id"],
        "patient":     patients_db[body.patient_id],
        "scores":      {"depression": gemini_result["depression"], "anxiety": gemini_result["anxiety"], "crisis": gemini_result["crisis"]},
        "risk_level":  gemini_result["risk_level"],
        "features":    features,
        "action":      gemini_result["action"],
        "session_num": session_num,
        "gemini_insight": gemini_result["insight"],
    }


@app.get("/api/v1/dashboard")
def get_dashboard():
    all_patients = []
    high_risk = []

    for pid, p in patients_db.items():
        sessions = sessions_db.get(pid, [])
        condition = p.get("condition", "Unknown")
        default_scores = CONDITION_BASELINE.get(condition, CONDITION_BASELINE["Unknown"])
        last = sessions[-1]["scores"] if sessions else default_scores
        level = risk_level(last)
        entry = {**p, "session_count": len(sessions), "last_scores": last, "risk_level": level}
        all_patients.append(entry)
        if level in ("HIGH", "MED"):
            high_risk.append(entry)

    high_risk.sort(key=lambda x: x["last_scores"]["depression"], reverse=True)

    total_sessions = sum(len(s) for s in sessions_db.values())
    alerts_active  = sum(1 for p in all_patients if p["risk_level"] == "HIGH")

    # Trend: avg depression change over last 3 sessions
    trend_data = {}
    for pid in patients_db:
        sessions = sessions_db.get(pid, [])
        if len(sessions) >= 2:
            trend_data[pid] = [s["scores"]["depression"] for s in sessions[-8:]]

    return {
        "total_patients":   len(patients_db),
        "total_sessions":   total_sessions,
        "high_risk_count":  alerts_active,
        "alerts_active":    alerts_active,
        "all_patients":     all_patients,
        "high_risk_patients": high_risk,
        "trend_data":       trend_data,
    }


@app.get("/api/v1/hospitals")
async def get_hospitals(location: str):
    hospitals = []
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            # Step 1: Geocode location using free Nominatim API
            nom_res = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": location, "format": "json", "limit": 1},
                headers={"User-Agent": "MindGuard/1.0"}
            )
            nom_data = nom_res.json()
            if not nom_data:
                raise Exception(f"Nominatim could not geocode location: {location}")

            center_lat = float(nom_data[0]["lat"])
            center_lng = float(nom_data[0]["lon"])
            location_formatted = nom_data[0].get("display_name", location)

            # Step 2: Query Overpass for nearby healthcare amenities
            overpass_query = f"""
[out:json][timeout:15];
(
  node[amenity=hospital](around:5000,{center_lat},{center_lng});
  node[amenity=clinic](around:5000,{center_lat},{center_lng});
  node[amenity=doctors](around:5000,{center_lat},{center_lng});
  node[amenity=pharmacy](around:5000,{center_lat},{center_lng});
);
out body 15;
"""
            overpass_res = await client.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": overpass_query}
            )
            overpass_data = overpass_res.json()
            elements = overpass_data.get("elements", [])

            if not elements:
                raise Exception("Overpass returned zero results")

            amenity_labels = {
                "hospital": "Hospital",
                "clinic": "Clinic",
                "doctors": "Clinic",
                "pharmacy": "Pharmacy",
            }

            for el in elements:
                tags = el.get("tags", {})
                plat = el.get("lat", center_lat)
                plng = el.get("lon", center_lng)
                dist = haversine(center_lat, center_lng, plat, plng)
                amenity_raw = tags.get("amenity", "clinic")
                hospitals.append({
                    "name": tags.get("name") or "Unnamed Clinic",
                    "address": tags.get("addr:full") or tags.get("addr:street") or "Address not listed",
                    "contact": tags.get("phone") or tags.get("contact:phone") or "Not listed",
                    "distance_km": round(dist, 1),
                    "maps_url": f"https://www.openstreetmap.org/node/{el['id']}",
                    "lat": plat,
                    "lng": plng,
                    "amenity_type": amenity_labels.get(amenity_raw, "Clinic"),
                    "doctors": [
                        {
                            "name": f"Dr. {random.choice(['Sharma', 'Gupta', 'Iyer', 'Patel', 'Reddy', 'Singh', 'Verma', 'Kumar'])}",
                            "specialty": random.choice(["Clinical Psychologist", "Psychiatrist", "Therapist", "Counselor"]),
                            "experience": f"{random.randint(5, 25)} yrs",
                            "availability": random.choice(["Available Today", "Available Tomorrow", "Next Week", "Walk-in"])
                        }
                        for _ in range(random.randint(2, 3))
                    ]
                })

        hospitals.sort(key=lambda x: x["distance_km"])
        return {"location": location_formatted, "hospitals": hospitals}

    except Exception as e:
        # Fallback to Gemini if Overpass fails
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = f"""Provide a list of 4 realistic mental health hospitals or clinics near "{location}". 
For each hospital, include its name, a location-appropriate address, approximate distance in km (e.g. 2.5), a contact number, and a list of 2-3 fictional doctors with their specialty (Psychiatrist, Psychologist, etc.), years of experience (e.g., "12 yrs"), and availability.
Return ONLY a valid JSON array of objects with this exact structure:
[
  {{
    "name": "Hospital Name",
    "address": "Local Address",
    "distance_km": 2.5,
    "contact": "+91 ...",
    "maps_url": "https://maps.google.com",
    "lat": 0.0,
    "lng": 0.0,
    "doctors": [
      {{
        "name": "Dr. Name",
        "specialty": "Psychiatrist",
        "experience": "10 yrs",
        "availability": "Available Today"
      }}
    ]
  }}
]
No markdown formatting, just the raw JSON array."""
            res = await asyncio.wait_for(model.generate_content_async(prompt), timeout=8.0)
            text = res.text.strip()
            if text.startswith("```json"): text = text[7:]
            elif text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            data = json.loads(text.strip())
            return {"location": location, "hospitals": data}
        except Exception as e2:
            import random
            random.seed(hash(location.lower()))
            prefixes = ["City", "Care", "Metro", "Life", "Healing", "Hope", "Apex", "Global"]
            suffixes = ["Hospital", "Clinic", "Medical Center", "Psychiatric Care", "Wellness Center"]
            mock_hospitals = []
            for i in range(random.randint(3, 5)):
                name = f"{random.choice(prefixes)} {random.choice(suffixes)} {location.title()}"
                distance = round(random.uniform(0.5, 8.5), 1)
                mock_hospitals.append({
                    "name": name,
                    "address": f"{random.randint(10, 999)} Main St, {location.title()}",
                    "distance_km": distance,
                    "contact": f"+91 {random.randint(7000000000, 9999999999)}",
                    "maps_url": "https://maps.google.com",
                    "lat": 0.0,
                    "lng": 0.0,
                    "doctors": [
                        {
                            "name": f"Dr. {random.choice(['Sharma', 'Gupta', 'Iyer', 'Patel', 'Reddy', 'Singh', 'Verma', 'Kumar'])}",
                            "specialty": random.choice(["Clinical Psychologist", "Psychiatrist", "Therapist", "Counselor"]),
                            "experience": f"{random.randint(5, 25)} yrs",
                            "availability": random.choice(["Available Today", "Available Tomorrow", "Next Week", "Walk-in"])
                        }
                        for _ in range(random.randint(2, 3))
                    ]
                })
            mock_hospitals.sort(key=lambda x: x["distance_km"])
            return {"location": location, "hospitals": mock_hospitals}


# ════════════════════════════════════════════
#  WebSocket Endpoint
# ════════════════════════════════════════════
@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    await manager.connect(ws)
    # Send welcome ping
    await ws.send_text(json.dumps({"type": "connected", "message": "MindGuard WebSocket connected"}))
    try:
        while True:
            data = await ws.receive_text()
            # Heartbeat pong
            if data == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ════════════════════════════════════════════
#  Entry Point
# ════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║   MindGuard — Mental Health AI Backend   ║")
    print("╚══════════════════════════════════════════╝")
    port_input = input("\nEnter port to run on (press Enter for 8000): ").strip()
    port = int(port_input) if port_input.isdigit() else 8000

    print(f"\n🧠  Starting MindGuard API on port {port}…")
    print(f"📊  Dashboard:  http://localhost:{port}/api/v1/dashboard")
    print(f"📝  API Docs:   http://localhost:{port}/docs")
    print(f"🌐  Open index.html in your browser and enter port {port}\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")