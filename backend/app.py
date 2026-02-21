import os
import sqlite3
import datetime
import pandas as pd
import nltk
import streamlit as st
import torch

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =====================================================
# NLTK
# =====================================================
nltk.download('punkt')
nltk.download('stopwords')

# =====================================================
# DATABASE
# =====================================================
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT,
    user TEXT,
    bot TEXT,
    sentiment TEXT,
    emotion TEXT,
    risk TEXT
)
""")
conn.commit()

# =====================================================
# SESSION STATE
# =====================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "llm_history" not in st.session_state:
    st.session_state.llm_history = []

if "contact" not in st.session_state:
    st.session_state.contact = ""

# =====================================================
# LOAD SENTIMENT + EMOTION
# =====================================================
@st.cache_resource
def load_analysis_models():
    sentiment = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    emotion = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions"
    )

    return sentiment, emotion

sentiment_model, emotion_model = load_analysis_models()

# =====================================================
# RISK MODEL
# =====================================================
import joblib

@st.cache_resource
def load_risk_model():
    vec_path = "vectorizer.pkl"
    model_path = "risk_model.pkl"

    if not os.path.exists(vec_path) or not os.path.exists(model_path):
        st.error(f"Missing pre-trained models. Please run `python train_risk_model.py` first.")
        st.stop()

    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)

    return vectorizer, model

vectorizer, risk_model = load_risk_model()

# =====================================================
# LOAD LLM (Phi-2)
# =====================================================
@st.cache_resource
def load_llm():
    model_name = "microsoft/phi-2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name) # Load config explicitly

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure the model's configuration also has a pad_token_id
    # The PhiModel initialization expects config.pad_token_id
    if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id # Use the tokenizer's pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config, # Pass the updated config
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    return tokenizer, model

tokenizer, model = load_llm()

# =====================================================
# ANALYSIS
# =====================================================
def analyze_user(text):
    sentiment = sentiment_model(text)[0]["label"]
    emotion = emotion_model(text)[0]["label"]

    vec = vectorizer.transform([text])
    prob = risk_model.predict_proba(vec)[0][1]

    if prob > 0.75:
        risk = "HIGH"
    elif prob > 0.45:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return sentiment, emotion, risk

# =====================================================
# DIALOGUE MANAGER
# =====================================================
def decide_mode(sentiment, risk):
    if risk == "HIGH":
        return "EMERGENCY"
    if sentiment.lower() in ["negative", "label_0"]:
        return "THERAPY"
    return "SUPPORT"
# =====================================================
# ACTIVITY GUIDANCE (Not fixed text)
# =====================================================
def get_activity_instruction(therapy):

    if therapy == "ENERGY":
        return "Suggest a small physical or rest-based recovery activity (rest, stretch, hydration, short nap, screen break)."

    if therapy == "STRESS":
        return "Suggest a simple stress-relief action (deep breathing, task breakdown, journaling, short pause, grounding exercise)."

    if therapy == "CBT":
        return "Help the user reframe their thoughts and suggest one gentle mental exercise (self-compassion, balanced thinking, perspective shift)."

    return "Suggest one small calming or mood-lifting activity appropriate to the situation."
# =====================================================
# THERAPEUTIC LAYER
# =====================================================
def select_therapy(emotion):

    cbt_emotions = ["sadness", "guilt", "fear", "anxiety", "remorse"]
    stress_emotions = ["nervousness", "confusion", "overwhelmed"]
    low_energy = ["tired", "fatigue","exhaustion","sleepiness"]
    if emotion in cbt_emotions:
        return "CBT"
    elif emotion in stress_emotions:
        return "STRESS"
    elif emotion in low_energy:
        return "ENERGY"
    else:
        return "GENERAL"

# =====================================================
# LLM RESPONSE
# =====================================================
def generate_response(user_text, sentiment, emotion, risk, therapy, mode):

    if mode == "EMERGENCY":
        if st.session_state.contact:
            st.warning(f"Emergency alert sent to: {st.session_state.contact}")

        return """
🚨 I’m concerned about your safety.

Please reach out immediately:
India Helpline: 9152987821
AASRA: +91-9820466726

You matter and you are not alone.
"""

    # Conversation context
    history_text = ""
    for role, msg in st.session_state.llm_history[-3:]:
        history_text += f"{role}: {msg}\n"
    activity_instruction=get_activity_instruction(therapy)

    prompt = f"""
You are a warm, empathetic mental health support assistant trained in Cognitive Behavioral Therapy (CBT).

User Emotional Context:
- Emotion: {emotion}
- Sentiment: {sentiment}
- Risk Level: {risk}
- Therapy Type: {therapy}
- Mode: {mode}

Recent Conversation:
{history_text}

Current User Message:
{user_text}

Response Rules (very important):
1. Start by acknowledging the user's situation or feeling in your own words.
2. Do NOT repeat the user's sentence.
3. Keep the response natural, supportive, and human.
4. Length: 4–6 short lines.
5. Give ONE specific and practical suggestion related to their situation.
6. Avoid generic phrases like:
   - "stay positive"
   - "take care"
   - "everything will be fine"
7. Use at most 1–2 gentle emojis (🌿 💛 🌸 🧘 ☕ 🌙).
8. Do NOT give medical or clinical advice.

Therapy Guidance:
If Therapy = CBT:
- Help the user see a more balanced or kinder perspective.
- Encourage self-compassion instead of self-criticism.

If Therapy = STRESS:
- Suggest a simple calming method (breathing, grounding, breaking tasks into small steps).

If Therapy = ENERGY:
- Suggest a short recovery action (rest, stretching, hydration, short break).

If Mode = SUPPORT:
- Focus on emotional validation and gentle encouragement.

Writing Style:
- Calm, conversational, and warm
- Like a caring counselor speaking to a student
- Personal and situation-focused, not generic

Assistant:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = text.split("Assistant:")[-1].strip()

    # Save memory
    st.session_state.llm_history.append(("User", user_text))
    st.session_state.llm_history.append(("Assistant", response))

    return response

# =====================================================
# STREAMLIT UI
# =====================================================
st.title("🧠 AI Mental Health Assistant")

st.sidebar.header("Emergency Contact")
st.session_state.contact = st.sidebar.text_input("Enter contact number")

user_input = st.text_input("How are you feeling today?")

if st.button("Send") and user_input:

    sentiment, emotion, risk = analyze_user(user_input)
    mode = decide_mode(sentiment, risk)
    therapy = select_therapy(emotion)

    response = generate_response(
        user_input,
        sentiment,
        emotion,
        risk,
        therapy,
        mode
    )

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Assistant", response))

    cursor.execute(
        "INSERT INTO chats (time,user,bot,sentiment,emotion,risk) VALUES (?,?,?,?,?,?)",
        (
            str(datetime.datetime.now()),
            user_input,
            response,
            sentiment,
            emotion,
            risk
        )
    )
    conn.commit()

# Display chat
for role, text in st.session_state.chat_history:
    st.markdown(f"**{role}:** {text}")

# View database
if st.checkbox("Show Database"):
    df = pd.read_sql("SELECT * FROM chats", conn)
    st.dataframe(df)
