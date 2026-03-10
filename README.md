# Soul Connect - AI Mental Health Assistant

Soul Connect is an intelligent mental health assistant designed to provide emotional support, risk assessment, and crisis intervention through a conversational interface. It combines advanced NLP models with a robust React & FastAPI architecture to deliver a safe, responsive, and aesthetically pleasing user experience.

---

## 🚀 Setup & Installation (Local Development)

### 1. Backend

Wait for dependencies to install and configure your API keys.

```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file in the `backend` directory:
```env
JWT_SECRET="your-secret-key"
TWILIO_ACCOUNT_SID="your-account-sid"
TWILIO_AUTH_TOKEN="your-auth-token"
TWILIO_PHONE_NUMBER="your-twilio-number"
FRONTEND_URL="http://localhost:5173" # Update later for Vercel
```

### 2. Frontend

Install the Node modules and point Vite to your local backend.

```bash
cd frontend
npm install
```

Create a `.env.local` file in the `frontend` directory:
```env
VITE_API_URL="http://localhost:8000"
```

---

## ▶️ Running the Application

### 1. Start the Backend Server
```bash
cd backend
uvicorn api:app --reload --port 8000
```

### 2. Start the Frontend App
Open a new terminal window:
```bash
cd frontend
npm run dev
```

### 3. Exposing Local Backend for Vercel (Ngrok)
If you deploy the frontend to the internet (Vercel) while keeping the heavy AI backend running locally, you must run an Ngrok tunnel:
```bash
# In an active terminal window
npx ngrok http 8000
```
This gives you a public Forwarding URL (e.g., `https://untinned-houndy...ngrok-free.dev`). 
Set this as the `VITE_API_URL` environment variable inside your Vercel deployment settings. Then update the `.env` file in your `backend` so `FRONTEND_URL` allows your Vercel URL.

---

## 🛠️ Tech Stack & Architecture

### Backend
- **Framework**: FastAPI (Python)
- **Database**: SQLite (`chat_history.db`)
- **Libraries**: `torch`, `transformers`, `scikit-learn`, `joblib`, `bcrypt`, `twilio`, `pandas`

### AI / NLP Models
- **Sentiment**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Emotion**: `SamLowe/roberta-base-go_emotions`
- **Risk Classifier**: Custom trained TF-IDF Logistic Regression model (`risk_model.pkl` & `vectorizer.pkl`)
- **LLM**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` hosted locally for fast generation without API timeouts.

### Frontend
- **Framework**: React + Vite
- **Styling**: Vanilla CSS (`App.css`) with modern UI elements.
- **State Management**: React Context (`AuthContext.jsx`)
- **Icons**: `lucide-react`

---

## ✨ Key Features

- **Engaging UI/UX**: Includes a welcoming entry page and a fully featured Landing page explaining the mission, features, and FAQs.
- **Conversational AI**: Human-like interactions that adapt to user sentiment and emotional states.
- **Text-to-Speech (TTS)**: Integrated Web Speech API to read bot messages aloud in a calming voice.
- **Crisis De-escalation Protocol**: If high risk is detected, the AI uses a strict dynamic system prompt to actively de-escalate the user while *silently* triggering Twilio to place an emergency call to a designated contact. The call explicitly recites the chat message that triggered the alarm.
- **User Authentication**: Secure JWT-based login and registration system.
- **Chat History**: Saves and retrieves conversation history with distinct chat session tracking.

---

## 🔐 Model Retraining

If you want to append new data to the Suicide Risk Classifier:
1. Add new text samples to `backend/Suicide_Detection.csv`.
2. Run `python clean_dataset.py` (if you added any custom noise removal scripts) or let testing strip out URLs and references.
3. Run `python train_risk_model.py`.
4. Overwrite confirmation will appear, saving the new `risk_model.pkl`. 

## License

MIT
