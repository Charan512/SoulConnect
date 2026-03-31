import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

BHASHINI_USER_ID = os.getenv("BHASHINI_USER_ID")
BHASHINI_UDYAT_KEY = os.getenv("BHASHINI_UDYAT_KEY")
BHASHINI_INFERENCE_API_KEY = os.getenv("BHASHINI_INFERENCE_API_KEY")

print("Tokens:", BHASHINI_USER_ID, bool(BHASHINI_INFERENCE_API_KEY))

url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
headers = {
    "userID": BHASHINI_USER_ID,
    "ulcaApiKey": BHASHINI_UDYAT_KEY,
    "Content-Type": "application/json"
}

payload = {
    "pipelineTasks": [
        {
            "taskType": "asr",
            "config": {
                "language": {"sourceLanguage": "te"}
            }
        }
    ],
    "pipelineRequestConfig": {
        "pipelineId": "64392f96daac500b55c543cd"
    }
}

try:
    response = requests.post(url, headers=headers, json=payload)
    print("Pipeline Response:", response.status_code)
    # print(json.dumps(response.json(), indent=2))
    
    if response.status_code == 200:
        data = response.json()
        callback_url = data["pipelineInferenceAPIEndPoint"]["callbackUrl"]
        inference_key = data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]
        service_id = data["pipelineResponseConfig"][0]["config"][0]["serviceId"]
        print("Callback:", callback_url)
        print("Service ID:", service_id)
except Exception as e:
    print("Error:", e)
