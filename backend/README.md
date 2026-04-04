# AI Mental Health Assistant - Backend API

This directory contains the FastAPI backend for the AI Mental Health Assistant application.

## Prerequisites
- Python 3.9+ 
- Local or globally installed `uvicorn`

## Setup Instructions

### 1. Install Dependencies
Install all required Python packages using pip:

```bash
pip install -r requirements.txt
```

*(Note: Depending on your environment, you may want to use a virtual environment before installing.)*

### 2. Environment Variables
Create a `.env` file in the root of the `backend` directory. The following environment variables are required for full functionality:

```env
# Twilio Required Credentials (For emergency calls)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone

# Allowed CORS Origins
FRONTEND_URL=https://your_frontend_url.com

# Bhashini Transcription Keys
BHASHINI_USER_ID=your_bhashini_id
BHASHINI_UDYAT_KEY=your_udyat_key
BHASHINI_INFERENCE_API_KEY=your_inference_api_key
```

### 3. Running the Server
You can start the FastAPI backend server using the following command:

```bash
uvicorn api:app --reload --port 8000
```

The API will then be accessible at `http://localhost:8000`. You can test functionality via the interactive Swagger docs at `http://localhost:8000/docs`.

### Notes
- **Models**: The application will automatically download lightweight text classification models on first startup. Please allow a minute or two on initial run.
- **Risk Model**: Ensure `vectorizer.pkl` and `risk_model.pkl` are present in this directory to load successfully.
