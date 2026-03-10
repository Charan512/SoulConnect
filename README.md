# Soul Connect - AI Mental Health Assistant

Soul Connect is an intelligent mental health assistant designed to provide emotional support, risk assessment, and crisis intervention through a conversational interface. It combines advanced NLP models with a robust React & FastAPI architecture to deliver a safe, responsive, and aesthetically pleasing user experience.

![System Architecture](file:///Users/sriramcharannalla/.gemini/antigravity/brain/f2108c0e-a75c-45b9-907d-926e2629f14c/soul_connect_architecture_1773153078972.png)

## ✨ Features

- **Engaging UI/UX**: Includes a welcoming entry page and a fully featured Landing page explaining the mission, features, and FAQs.
- **Conversational AI**: Powered by `TinyLlama/TinyLlama-1.1B-Chat-v1.0` locally for fast, empathetic, and human-like interactions without the risk of API timeouts.
- **Text-to-Speech (TTS)**: Integrated Web Speech API to read bot messages aloud in a calming voice.
- **Emotional Analysis**: Detects user emotions and sentiment using RoBERTa-based models.
- **Risk Assessment**: Identifies potential self-harm using a custom-trained TF-IDF Logistic Regression model (trained on a noise-cleaned dataset of 230k+ records).
- **Crisis De-escalation Protocol**: If high risk is detected, the AI uses a strict dynamic system prompt to actively de-escalate the user while *silently* triggering Twilio to place an emergency call to a designated contact. 
- **User Authentication**: Secure JWT-based login and registration system.
- **Chat History**: Saves and retrieves conversation history with distinct chat session tracking.

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **Database**: SQLite (`chat_history.db`)
- **NLP Models**:
  - Sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - Emotion: `SamLowe/roberta-base-go_emotions`
  - Risk Classifier: Custom trained model (`risk_model.pkl` & `vectorizer.pkl`)
  - LLM: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Libraries**: `torch`, `transformers`, `scikit-learn`, `joblib`, `bcrypt`, `twilio`, `pandas`

### Frontend
- **Framework**: React + Vite
- **Styling**: Vanilla CSS (`App.css`) with modern UI elements.
- **State Management**: React Context (`AuthContext.jsx`)
- **Icons**: `lucide-react`

---

## 🚀 Setup & Installation (Local Development)

### 1. Backend

Install dependencies and configure your API keys.

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

Run the server:
```bash
cd backend
uvicorn api:app --reload --port 8000
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

Run the Vite Dev Server:
```bash
cd frontend
npm run dev
```

---

## 🌍 Deployment (Vercel + Ngrok)

To deploy the frontend to the internet (Vercel) while keeping the heavy AI backend running locally on your machine, follow these steps:

### 1. Expose Local Backend via Ngrok
Run Ngrok to create a secure public tunnel to your local Port 8000.
```bash
# In an active terminal window
npx ngrok http 8000
```
This will give you a public Forwarding URL (e.g., `https://untinned-houndy-arabella.ngrok-free.dev`).

### 2. Configure Backend CORS
Update the `.env` file in your `backend` to explicitly allow your Vercel deployment domain.
```env
FRONTEND_URL="https://soul-connect-your-app-id.vercel.app"
```
*Restart the Uvicorn server to apply.*

### 3. Deploy Frontend to Vercel
Set Vercel's Environment Variables via the dashboard or CLI:
```bash
# In the frontend directory
npx vercel --prod
```
In the Vercel dashboard, under **Settings > Environment Variables**, add:
- Key: `VITE_API_URL`
- Value: `<Your-Ngrok-Forwarding-URL>`

> **Note on Ngrok Free Tier:** The frontend React code is already configured with an `ngrok-skip-browser-warning: true` header on all `/api/` fetch requests to bypass the Ngrok HTML intercept page.

---

## 🔐 Model Retraining

If you want to append new data to the Suicide Risk Classifier:
1. Add new text samples to `backend/Suicide_Detection.csv`.
2. Run `python clean_dataset.py` (if you added any custom noise removal scripts) or run the training script directly to strip regex URLs and train.
3. Run `python train_risk_model.py`.
4. Overwrite confirmation will appear, saving the new `risk_model.pkl`. 

## License

MIT
