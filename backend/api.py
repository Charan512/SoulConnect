"""
FastAPI Backend for the AI Mental Health Assistant
===================================================
Endpoints:
  POST /register      - Create a new user account
  POST /login         - Authenticate and get JWT token
  POST /analyze       - Analyze text (auth required)
  POST /chat          - Full chat pipeline (auth required)
  GET  /history       - Chat history (auth required)
  DELETE /history     - Clear history (auth required)
  GET  /health        - Health check
  GET  /me            - Get current user info (auth required)

Run:
  uvicorn api:app --reload --port 8000
"""

import os
import pymongo
from bson.objectid import ObjectId
import datetime
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import jwt
import nltk
import torch
from fastapi import FastAPI, HTTPException, Query, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import bcrypt as _bcrypt
from pydantic import BaseModel, Field
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

# Twilio Config
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
JWT_SECRET = os.environ.get("JWT_SECRET", "mental-health-app-secret-key-2024")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

# ─────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────
sentiment_model = None
emotion_model = None
vectorizer = None
risk_model = None
tokenizer = None
llm_model = None
conversation_buffers: dict = {}

# ─────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────
MONGO_URL = os.environ.get("MONGO_URL")
mongo_client = None
db = None

def _get_query_user_id(user_id):
    try:
        return int(user_id)
    except ValueError:
        return str(user_id)


# ─────────────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────────────
def _create_token(user_id: int, username: str) -> str:
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    return _decode_token(token)


# ─────────────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global sentiment_model, emotion_model, vectorizer, risk_model, tokenizer, llm_model

    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

    print("[startup] Loading sentiment model ...")
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )

    print("[startup] Loading emotion model ...")
    emotion_model = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
    )

    vec_path, model_path = "vectorizer.pkl", "risk_model.pkl"
    if not os.path.exists(vec_path) or not os.path.exists(model_path):
        raise RuntimeError("Missing risk models. Run `python train_risk_model.py` first.")
    print("[startup] Loading risk model ...")
    vectorizer = joblib.load(vec_path)
    risk_model = joblib.load(model_path)

    if os.environ.get("SKIP_LLM", "").strip() in ("1", "true", "yes"):
        print("[startup] SKIP_LLM set — skipping LLM.")
        tokenizer = None
        llm_model = None
    else:
        try:
            print("[startup] Loading LLM (Qwen/Qwen2.5-0.5B-Instruct) ...")
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            config = AutoConfig.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
                config.pad_token_id = tokenizer.pad_token_id
            # Enable Mac GPU (MPS) securely avoiding huge allocator warmup failure
            target_device = None
            if torch.cuda.is_available():
                dtype = torch.float16
                device_map = "auto"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dtype = torch.float16
                device_map = None
                target_device = "mps"
            else:
                dtype = torch.float32
                device_map = "auto"

            load_kwargs = {
                "config": config,
                "torch_dtype": dtype,
            }
            if device_map is not None:
                load_kwargs["device_map"] = device_map

            llm_model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs
            )
            
            if target_device is not None:
                llm_model.to(target_device)
            # If any params are on meta device, model is disk-offloaded → too slow on CPU
            offloaded = any(
                p.device.type == "meta"
                for p in llm_model.parameters()
            )
            if offloaded:
                print("[startup] LLM params disk-offloaded (no GPU) — using fast fallback responses.")
                tokenizer = None
                llm_model = None
            else:
                print("[startup] LLM loaded")
        except BaseException as e:
            print(f"[startup] WARNING: LLM failed ({e})")
            tokenizer = None
            llm_model = None

    global mongo_client, db
    if MONGO_URL:
        mongo_client = pymongo.MongoClient(MONGO_URL)
        db = mongo_client["Aipowered"]
        print("[startup] Connected to MongoDB")
    print("[startup] Server ready!")
    yield
    print("[shutdown] Done.")


# ─────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────
app = FastAPI(
    title="Mental Health Assistant API",
    version="2.0.0",
    lifespan=lifespan,
)

frontend_url = os.environ.get("FRONTEND_URL")
if not frontend_url:
    raise RuntimeError(
        "FRONTEND_URL environment variable is not set. "
        "Add it to your .env file (e.g. FRONTEND_URL=https://your-app.vercel.app)"
    )
allowed_origins = [frontend_url]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=4)
    emergency_contact: str = Field(..., min_length=5, description="Emergency phone number")


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    username: str
    user_id: str


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)


class AnalyzeResponse(BaseModel):
    sentiment: str
    emotion: str
    risk: str
    risk_probability: float
    mode: str
    therapy: str


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1)
    session_id: str = Field(default="default")


class ChatResponse(BaseModel):
    response: str
    sentiment: str
    emotion: str
    risk: str
    risk_probability: float
    mode: str
    therapy: str
    emergency_contact: Optional[str] = None


class HistoryRecord(BaseModel):
    id: str
    time: str
    user_msg: str
    bot_msg: str
    sentiment: str
    emotion: str
    risk: str
    session_id: str

class SessionRecord(BaseModel):
    session_id: str
    title: str
    last_updated: str


class AudioUpload(BaseModel):
    audio_base64: str
    source_language: str = "en"


# ─────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ──── FUZZY LOGIC SYSTEM SETUP ────
risk_input = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'risk_prob')
sentiment_input = ctrl.Antecedent(np.arange(-1.0, 1.01, 0.01), 'sentiment_val')

urgency_output = ctrl.Consequent(np.arange(0, 101, 1), 'urgency')

risk_input['low'] = fuzz.trimf(risk_input.universe, [0, 0, 0.5])
risk_input['medium'] = fuzz.trimf(risk_input.universe, [0.3, 0.5, 0.75])
risk_input['high'] = fuzz.trimf(risk_input.universe, [0.6, 1.0, 1.0])

sentiment_input['negative'] = fuzz.trimf(sentiment_input.universe, [-1.0, -1.0, -0.1])
sentiment_input['neutral'] = fuzz.trimf(sentiment_input.universe, [-0.5, 0, 0.5])
sentiment_input['positive'] = fuzz.trimf(sentiment_input.universe, [0.1, 1.0, 1.0])

urgency_output['support'] = fuzz.trimf(urgency_output.universe, [0, 0, 50])
urgency_output['therapy'] = fuzz.trimf(urgency_output.universe, [30, 60, 80])
urgency_output['emergency'] = fuzz.trimf(urgency_output.universe, [70, 100, 100])

rule1 = ctrl.Rule(risk_input['high'] | (risk_input['medium'] & sentiment_input['negative']), urgency_output['emergency'])
rule2 = ctrl.Rule(risk_input['medium'] & (sentiment_input['neutral'] | sentiment_input['positive']), urgency_output['therapy'])
rule3 = ctrl.Rule(risk_input['low'] & sentiment_input['negative'], urgency_output['therapy'])
rule4 = ctrl.Rule(risk_input['low'] & (sentiment_input['neutral'] | sentiment_input['positive']), urgency_output['support'])

urgency_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
urgency_sim = ctrl.ControlSystemSimulation(urgency_ctrl)

def _analyze_text(text: str):
    sent_res = sentiment_model(text)[0]
    sent_label = sent_res["label"]
    sent_score = sent_res["score"]
    
    if sent_label in ["negative", "label_0"]:
        sent_val = -sent_score
    elif sent_label in ["positive", "label_2"]:
        sent_val = sent_score
    else:
        sent_val = 0.0

    emo = emotion_model(text)[0]["label"]
    
    vec = vectorizer.transform([text])
    prob = float(risk_model.predict_proba(vec)[0][1])
    
    try:
        urgency_sim.input['risk_prob'] = prob
        urgency_sim.input['sentiment_val'] = sent_val
        urgency_sim.compute()
        urgency = urgency_sim.output['urgency']
    except Exception as e:
        print(f"[Fuzzy] Error computing urgency: {e}")
        urgency = prob * 100  # fallback

    print(f"[Fuzzy] prob: {prob:.2f}, sent_val: {sent_val:.2f} -> Urgency: {urgency:.2f}")

    # Fallback keyword matching for absolute safety
    text_lower = text.lower()
    danger_words = ["die", "kill myself", "suicide", "end it", "don't want to live", "no reason to live", "kill me"]
    if any(w in text_lower for w in danger_words):
        urgency = max(urgency, 85)

    if urgency >= 70:
        risk = "HIGH"
    elif urgency >= 40:
        risk = "MEDIUM"
    else:
        risk = "LOW"
        
    return sent_label, emo, risk, prob


def _decide_mode(sentiment, risk):
    if risk == "HIGH":
        return "EMERGENCY"
    if risk == "MEDIUM":
        return "THERAPY"
    return "SUPPORT"


def _select_therapy(emotion):
    if emotion in ["sadness", "guilt", "fear", "anxiety", "remorse"]:
        return "CBT"
    if emotion in ["nervousness", "confusion", "overwhelmed"]:
        return "STRESS"
    if emotion in ["tired", "fatigue", "exhaustion", "sleepiness"]:
        return "ENERGY"
    return "GENERAL"


def _get_activity_instruction(therapy):
    return {
        "ENERGY": "Suggest a small recovery activity.",
        "STRESS": "Suggest a stress-relief action.",
        "CBT": "Help reframe thoughts gently.",
    }.get(therapy, "Suggest a calming activity.")
def _generate_llm_response(user_text, sentiment, emotion, risk, therapy, mode,
                           session_id="default", emergency_contact=None):
    """Generate response using Qwen with proper chat template."""
    history = conversation_buffers.get(session_id, [])

    is_short = len(user_text.split()) < 8
    is_casual = is_short and risk == "LOW" and sentiment.lower() in ["positive", "neutral", "label_1", "label_2"]
    
    if mode == "EMERGENCY" or risk == "HIGH":
        rules = (
            "CRITICAL PROTOCOL: The user is expressing severe distress, self-harm, or suicidal ideation. "
            "Your immediate goal is to de-escalate, provide extreme empathy, and anchor them to the present moment. "
            "Do NOT tell them you are calling emergency services. Do NOT give them generic hotline numbers. "
            "Just listen to them and try to find one small reason for them to keep talking to you."
        )
        activity_text = "Activity: Ask them to take a single deep breath with you, or tell you one thing they can see right now."
    elif is_casual:
        rules = "Rules: The user is making casual conversation or a brief statement. Respond naturally, warmly, and concisely in 1-2 friendly sentences with emojis. Be conversational like a normal person."
        activity_text = ""
    else:
        activity = _get_activity_instruction(therapy)
        rules = (
            "Rules: You must provide a warm, empathetic, and calming response. "
            "Never offer unsolicited financial, medical, or life advice. "
            "Validate their emotions deeply and gently calm them down. "
            "Respond in 2-3 short supportive sentences."
        )
        activity_text = f"Activity: {activity}\n\n"

    system_msg = (
        "You are 'Soul Connect', a helpful empathetic mental health companion.\n"
        "Talk directly to the user in short conversational English responses and actively use emojis to express empathy.\n"
        f"{rules} {activity_text}"
    )

    messages = [
        {"role": "system", "content": system_msg}
    ]

    for role, text in history[-3:]:
        if role == "User":
            messages.append({"role": "user", "content": text.strip()})
        else:
            messages.append({"role": "assistant", "content": text.strip()})

    messages.append({"role": "user", "content": user_text.strip()})

    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text_prompt], return_tensors="pt").to(llm_model.device)
    
    outputs = llm_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Strip the input prompt from the output
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    if session_id not in conversation_buffers:
        conversation_buffers[session_id] = []
    conversation_buffers[session_id].append(("User", user_text))
    conversation_buffers[session_id].append(("Assistant", response))
    
    return response



import random

def _fallback_response(sentiment, emotion, therapy, mode):
    """Context-aware fallback when LLM is not loaded."""

    # Therapy-specific responses
    therapy_responses = {
        "CBT": [
            "I notice you're going through a tough time. Try this: write down the thought bothering you, then ask yourself — is there another way to look at this? Sometimes our minds jump to the worst conclusion.",
            "When negative thoughts spiral, try the 5-4-3-2-1 technique: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. It helps ground you in the present.",
            "It sounds like things feel heavy right now. Remember, thoughts aren't facts — they're just thoughts. What would you tell a friend in your situation? Often we're kinder to others than ourselves.",
            "I hear you. One helpful practice: catch the automatic thought, examine the evidence for and against it, then try to form a more balanced perspective. You're stronger than you think.",
        ],
        "STRESS": [
            "Stress can feel overwhelming, but your body knows how to calm down. Try box breathing: inhale 4 seconds, hold 4, exhale 4, hold 4. Repeat 4 times. You've got this.",
            "When everything feels like too much, focus on just the next small step. You don't need to solve everything right now — just one thing at a time.",
            "Take a moment to release tension: roll your shoulders, unclench your jaw, relax your hands. Sometimes stress hides in our body without us noticing.",
            "It's okay to feel stressed. Try stepping away for 5 minutes — a short walk, some fresh air, or even just looking out a window can help reset your mind.",
        ],
        "ENERGY": [
            "It sounds like you're running on empty. Be gentle with yourself — rest isn't laziness, it's essential. Even 10 minutes of quiet time can help recharge.",
            "Fatigue often means we need to slow down. Try a micro-break: close your eyes, take 5 deep breaths, and let your body relax for just two minutes.",
            "When energy is low, small wins matter most. Pick one tiny thing you can do right now, and let that be enough for today. You're doing better than you think.",
        ],
    }

    # Sentiment-based responses
    sentiment_responses = {
        "positive": [
            f"It's wonderful to hear some positivity! Your {emotion} energy is something to cherish. What's been bringing you joy lately?",
            f"I love that you're feeling {emotion}! Keep nurturing those moments — they build resilience for tougher days.",
            "That's really great to hear! Savoring positive moments actually rewires your brain for more happiness. Take a moment to really soak it in.",
        ],
        "neutral": [
            "Thanks for sharing. Even quiet moments are worth exploring. How have you been feeling overall lately? I'm here to listen.",
            "I appreciate you reaching out. Sometimes just putting thoughts into words helps us understand them better. What's on your mind?",
            "I'm here for you. Whether things are calm or complicated, talking through it can help. What would feel most helpful right now?",
        ],
        "negative": [
            f"I can sense you're feeling {emotion}, and that's completely valid. You don't have to carry this alone — I'm here with you.",
            f"It takes courage to share when you're feeling {emotion}. Remember, difficult feelings are temporary — they always pass, even when it doesn't feel like it.",
            "I'm sorry you're going through this. Please be as kind to yourself as you would be to someone you love. What's one small thing that might bring you comfort right now?",
        ],
    }

    # Try therapy-specific first, then sentiment-based
    if therapy in therapy_responses:
        return random.choice(therapy_responses[therapy])

    sent_key = sentiment.lower()
    if sent_key in sentiment_responses:
        return random.choice(sentiment_responses[sent_key])

    return "I hear you, and your feelings matter. I'm here to support you through whatever you're experiencing."


# ─────────────────────────────────────────────────────
# TWILIO EMERGENCY CALL
# ─────────────────────────────────────────────────────
def trigger_emergency_call(contact_number: str, username: str, trigger_message: str):
    """
    Triggers an automated voice call via Twilio to the emergency contact.
    This function should be run as a background task.
    """
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
        print("[Twilio] WARNING: Twilio credentials not fully configured. Cannot place call.")
        return

    # Keep it brief but highly descriptive of the triggering event
    twiml_msg = (
        f"<Response>"
        f"<Say voice='alice' language='en-US'>"
        f"Emergency Alert from Mind Care A I. "
        f"This call is to warn you about your friend {username}, who is currently feeling severe mental distress. "
        f"They just sent the following high-risk message to their chatbot: "
        f"<Pause length='1'/>"
        f"'{trigger_message}'"
        f"<Pause length='1'/>"
        f"Please contact or help them as soon as possible."
        f"</Say>"
        f"</Response>"
    )

    try:
        # Auto-format 10-digit Indian numbers since Twilio requires E.164
        formatted_number = contact_number
        if len(contact_number) == 10 and contact_number.isdigit():
            formatted_number = f"+91{contact_number}"
            
        print(f"[Twilio] Initiating emergency voice call to {formatted_number} for user {username}...")
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            twiml=twiml_msg,
            to=formatted_number,

            from_=TWILIO_PHONE_NUMBER
        )
        print(f"[Twilio] Call initiated successfully. Call SID: {call.sid}")
    except Exception as e:
        print(f"[Twilio] ERROR: Failed to initiate emergency call - {str(e)}")


# ═════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═════════════════════════════════════════════════════
@app.post("/register", response_model=AuthResponse, tags=["Auth"])
async def register(req: RegisterRequest):
    existing = db.users.find_one({"username": req.username})
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    pw_hash = _bcrypt.hashpw(req.password.encode(), _bcrypt.gensalt()).decode()
    res = db.users.insert_one({
        "username": req.username,
        "password_hash": pw_hash,
        "emergency_contact": req.emergency_contact,
        "created_at": str(datetime.datetime.now())
    })
    user_id = str(res.inserted_id)

    token = _create_token(user_id, req.username)
    return AuthResponse(token=token, username=req.username, user_id=user_id)


@app.post("/login", response_model=AuthResponse, tags=["Auth"])
async def login(req: LoginRequest):
    user = db.users.find_one({"username": req.username})

    if not user or not _bcrypt.checkpw(req.password.encode(), user["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user_id = str(user.get("id", str(user["_id"])))
    token = _create_token(user_id, user["username"])
    return AuthResponse(token=token, username=user["username"], user_id=user_id)


@app.get("/me", tags=["Auth"])
async def get_me(current_user: dict = Depends(get_current_user)):
    q_id = _get_query_user_id(current_user["user_id"])
    search = {"id": q_id} if isinstance(q_id, int) else {"_id": ObjectId(q_id)}
    user = db.users.find_one(search)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": str(user.get("id", str(user["_id"]))), "username": user["username"], "emergency_contact": user.get("emergency_contact"), "created_at": user.get("created_at")}


# ═════════════════════════════════════════════════════
# HEALTH
# ═════════════════════════════════════════════════════
@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "llm_loaded": llm_model is not None,
        "timestamp": str(datetime.datetime.now()),
    }


# ═════════════════════════════════════════════════════
# ANALYZE
# ═════════════════════════════════════════════════════
@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze(req: AnalyzeRequest, current_user: dict = Depends(get_current_user)):
    sentiment, emotion, risk, prob = _analyze_text(req.text)
    mode = _decide_mode(sentiment, risk)
    therapy = _select_therapy(emotion)
    return AnalyzeResponse(
        sentiment=sentiment, emotion=emotion, risk=risk,
        risk_probability=round(prob, 4), mode=mode, therapy=therapy,
    )


# ═════════════════════════════════════════════════════
# CHAT
# ═════════════════════════════════════════════════════
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    sentiment, emotion, risk, prob = _analyze_text(req.text)
    mode = _decide_mode(sentiment, risk)
    therapy = _select_therapy(emotion)

    # Get user's emergency contact
    q_id = _get_query_user_id(current_user["user_id"])
    search = {"id": q_id} if isinstance(q_id, int) else {"_id": ObjectId(q_id)}
    user = db.users.find_one(search)
    emergency_contact = user["emergency_contact"] if user and "emergency_contact" in user else None

    # Trigger Emergency Call if HIGH risk
    if mode == "EMERGENCY" and emergency_contact:
        background_tasks.add_task(trigger_emergency_call, emergency_contact, current_user["username"], req.text)

    # Generate Response
    if llm_model is None or tokenizer is None:
        if mode == "EMERGENCY":
            response = _generate_llm_response(
                req.text, sentiment, emotion, risk, therapy, mode,
                emergency_contact=None, # Don't show contact explicitly in msg anymore to feel more casual
            )
        else:
            response = _fallback_response(sentiment, emotion, therapy, mode)
    else:
        response = _generate_llm_response(
            req.text, sentiment, emotion, risk, therapy, mode,
            session_id=req.session_id, emergency_contact=None,
        )

    # Save to DB
    db.chats.insert_one({
        "user_id": _get_query_user_id(current_user["user_id"]),
        "session_id": req.session_id,
        "time": str(datetime.datetime.now()),
        "user_msg": req.text,
        "bot_msg": response,
        "sentiment": sentiment,
        "emotion": emotion,
        "risk": risk
    })

    return ChatResponse(
        response=response, sentiment=sentiment, emotion=emotion, risk=risk,
        risk_probability=round(prob, 4), mode=mode, therapy=therapy,
        emergency_contact=emergency_contact if mode == "EMERGENCY" else None,
    )


# ═════════════════════════════════════════════════════
# HISTORY
# ═════════════════════════════════════════════════════
@app.get("/sessions", response_model=list[SessionRecord], tags=["History"])
async def get_sessions(current_user: dict = Depends(get_current_user)):
    q_id = _get_query_user_id(current_user["user_id"])
    pipeline = [
        {"$match": {"user_id": q_id}},
        {"$sort": {"_id": 1}},
        {"$group": {
            "_id": {"$ifNull": ["$session_id", "default"]},
            "last_updated": {"$last": "$time"},
            "title": {"$first": "$user_msg"}
        }},
        {"$sort": {"last_updated": -1}}
    ]
    rows = list(db.chats.aggregate(pipeline))
    sessions = []
    for r in rows:
        session_id = r["_id"]
        title = r["title"] or "New Chat"
        if len(title) > 40:
            title = title[:37] + "..."
        sessions.append(SessionRecord(
            session_id=session_id,
            title=title,
            last_updated=r["last_updated"]
        ))
    return sessions


@app.get("/history/{session_id}", response_model=list[HistoryRecord], tags=["History"])
async def get_history(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    current_user: dict = Depends(get_current_user),
):
    q_id = _get_query_user_id(current_user["user_id"])
    rows = list(db.chats.find({"user_id": q_id, "session_id": session_id}).sort("_id", -1).limit(limit))
    return [HistoryRecord(
        id=str(r.get("id", r["_id"])), time=r.get("time", ""), user_msg=r.get("user_msg", ""),
        bot_msg=r.get("bot_msg", ""), sentiment=r.get("sentiment", ""),
        emotion=r.get("emotion", ""), risk=r.get("risk", ""), session_id=r.get("session_id", "default")
    ) for r in rows]


@app.delete("/history", tags=["History"])
async def clear_history(current_user: dict = Depends(get_current_user)):
    q_id = _get_query_user_id(current_user["user_id"])
    db.chats.delete_many({"user_id": q_id})
    return {"message": "Chat history cleared"}

@app.delete("/history/{session_id}", tags=["History"])
async def delete_session(session_id: str, current_user: dict = Depends(get_current_user)):
    q_id = _get_query_user_id(current_user["user_id"])
    db.chats.delete_many({"user_id": q_id, "session_id": session_id})
    return {"message": "Session deleted"}


# ═════════════════════════════════════════════════════
# VOICE INPUT (BHASHINI)
# ═════════════════════════════════════════════════════

BHASHINI_USER_ID = os.getenv("BHASHINI_USER_ID")
BHASHINI_UDYAT_KEY = os.getenv("BHASHINI_UDYAT_KEY")
BHASHINI_INFERENCE_API_KEY = os.getenv("BHASHINI_INFERENCE_API_KEY")

import requests

@app.post("/speech-to-text", tags=["Voice"])
async def speech_to_text(payload: AudioUpload):
    if not all([BHASHINI_USER_ID, BHASHINI_UDYAT_KEY, BHASHINI_INFERENCE_API_KEY]):
        raise HTTPException(status_code=500, detail="Bhashini credentials not fully configured in backend.")
        
    try:
        # 1. Get Pipeline Service ID
        url_pipeline = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
        headers_pipeline = {
            "userID": BHASHINI_USER_ID,
            "ulcaApiKey": BHASHINI_UDYAT_KEY,
            "Content-Type": "application/json"
        }
        # 64392f96daac500b55c543cd is the standard standard pipelineId
        payload_pipeline = {
            "pipelineTasks": [{"taskType": "asr", "config": {"language": {"sourceLanguage": payload.source_language}}}],
            "pipelineRequestConfig": {"pipelineId": "64392f96daac500b55c543cd"} 
        }
        res_pipeline = requests.post(url_pipeline, headers=headers_pipeline, json=payload_pipeline)
        if res_pipeline.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Bhashini Pipeline Error: {res_pipeline.text}")
            
        data = res_pipeline.json()
        callback_url = data["pipelineInferenceAPIEndPoint"]["callbackUrl"]
        inference_key = data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]
        service_id = data["pipelineResponseConfig"][0]["config"][0]["serviceId"]

        # 2. Perform Inference
        headers_infer = {
            "Authorization": inference_key,
            "Content-Type": "application/json"
        }
        payload_infer = {
            "pipelineTasks": [
                {
                    "taskType": "asr",
                    "config": {
                        "language": {"sourceLanguage": payload.source_language},
                        "serviceId": service_id,
                        "audioFormat": "webm",
                        "samplingRate": 48000
                    }
                }
            ],
            "inputData": {
                "audio": [
                    {"audioContent": payload.audio_base64}
                ]
            }
        }
        
        res_infer = requests.post(callback_url, headers=headers_infer, json=payload_infer)
        if res_infer.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Bhashini Inference Error: {res_infer.text}")
            
        infer_data = res_infer.json()
        transcribed_text = infer_data["pipelineResponse"][0]["output"][0]["source"]
        
        return {"text": transcribed_text}

    except Exception as e:
        print("Bhashini Error:", e)
        raise HTTPException(status_code=500, detail=str(e))
