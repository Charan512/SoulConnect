# Soul Connect — Frontend

> 🧠 An AI-powered Mental Health Companion built with React + Vite.

This directory contains the frontend for the **Soul Connect** application — a compassionate, real-time AI chat interface for mental health support. It communicates with the FastAPI backend to deliver sentiment analysis, emotion detection, risk assessment, and empathetic LLM responses.

---

## ✨ Features

- 🔐 **JWT Authentication** — Register, Login, auto-logout on token expiry
- 💬 **AI Chat Interface** — Real-time conversation with the Soul Connect AI companion
- 🎙️ **Voice Input** — Microphone support via Bhashini Speech-to-Text API
- 🔊 **Text-to-Speech** — Plays bot responses aloud using browser's Web Speech API
- 🗂️ **Session History** — Multiple chat sessions with a collapsible sidebar
- 🗑️ **Session Management** — Create new chats, delete old ones
- 🚨 **Emergency Mode UI** — Visual cues when high-risk distress is detected
- 📱 **Responsive Design** — Works on desktop and mobile

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Framework | React 18 |
| Bundler | Vite 5 |
| Routing | React Router DOM v6 |
| Icons | Lucide React |
| Styling | Vanilla CSS (custom design system) |
| Auth | JWT via `localStorage` |

---

## 🚀 Getting Started

### Prerequisites
- Node.js `v18+`
- npm `v9+`
- Backend server running (see `/backend/README.md`)

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Environment

Create a `.env.local` file in this directory:

```env
# URL of your running backend (local or deployed)
VITE_API_URL=http://localhost:8000
```

> For a deployed backend (e.g. via ngrok or Render), replace the URL accordingly.

### 3. Start Development Server

```bash
npm run dev
```

The app will be available at **`http://localhost:5173`**.

---

## 📦 Build for Production

```bash
npm run build
```

Output will be in the `dist/` folder. Preview the production build with:

```bash
npm run preview
```

---

## 📁 Project Structure

```
frontend/
├── public/              # Static assets
├── src/
│   ├── context/
│   │   └── AuthContext.jsx   # Global auth state, login/logout, authFetch helper
│   ├── pages/
│   │   ├── Landing.jsx       # Marketing / landing page
│   │   ├── Login.jsx         # Login form
│   │   ├── Register.jsx      # Registration form
│   │   ├── Welcome.jsx       # Post-login welcome screen
│   │   └── Chat.jsx          # Main chat interface
│   ├── App.jsx               # Route definitions
│   ├── App.css               # Global design system & component styles
│   └── main.jsx              # React app entry point
├── index.html
├── package.json
└── vite.config.js
```

---

## 🔑 Authentication Flow

1. User registers or logs in → backend returns a **JWT token** (valid 24 hours)
2. Token is stored in `localStorage`
3. Every API request uses `Authorization: Bearer <token>`
4. On app load, `/me` is called to validate the token
5. If expired (HTTP 401), the user is **automatically logged out** and redirected to login

---

## 🌐 API Endpoints Used

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/register` | Create a new account |
| `POST` | `/login` | Authenticate and get token |
| `GET` | `/me` | Validate token & get user info |
| `POST` | `/chat` | Send a message and get AI response |
| `GET` | `/sessions` | List all past chat sessions |
| `GET` | `/history/:id` | Load messages for a session |
| `DELETE` | `/history/:id` | Delete a session |
| `POST` | `/speech-to-text` | Transcribe voice audio to text |

---

## 🧩 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `VITE_API_URL` | ✅ Yes | Base URL of the FastAPI backend |

> If `VITE_API_URL` is not set, the app falls back to `/api` (useful when frontend and backend are served from the same domain).

---

## 📝 Notes

- The app uses the browser's built-in **Web Speech API** for TTS — no external service needed.
- Voice input uses **Bhashini ASR** via the backend `/speech-to-text` endpoint.
- Session IDs are generated as Unix timestamps (`Date.now().toString()`) on the client side.
