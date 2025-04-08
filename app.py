from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import datetime
import logging

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow CORS for all origins

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env file")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

# Function to generate the medical prompt
def generate_medical_prompt(symptoms, age=None, gender=None):
    return f"""ACT AS A MEDICAL PROFESSIONAL. ANALYZE THESE SYMPTOMS:

PATIENT DETAILS:
- Symptoms: {symptoms}
- Age: {age if age else 'Not specified'}
- Gender: {gender if gender else 'Not specified'}

RESPONSE REQUIREMENTS:
1. Provide 3 possible medical conditions
2. Include probability (low/medium/high) and urgency (low/medium/high)
3. List 3 recommended actions
4. Add a medical disclaimer
5. Format as perfect JSON:

{{
  "analysis": {{
    "possible_conditions": [
      {{
        "name": "condition_name",
        "probability": "low/medium/high",
        "urgency": "low/medium/high",
        "description": "1-2 sentence explanation"
      }},
      {{
        "name": "condition_name",
        "probability": "low/medium/high",
        "urgency": "low/medium/high",
        "description": "1-2 sentence explanation"
      }},
      {{
        "name": "condition_name",
        "probability": "low/medium/high",
        "urgency": "low/medium/high",
        "description": "1-2 sentence explanation"
      }}
    ],
    "recommended_actions": [
      "Action 1",
      "Action 2",
      "Action 3"
    ],
    "disclaimer": "This is not a diagnosis... Consult a doctor for professional medical advice."
  }},
  "status": "success"
}}

IMPORTANT:
- Respond ONLY with valid JSON
- Do NOT include any additional text
- Maintain professional medical accuracy
- If unsure about symptoms, say so in the description
"""

# Function to extract JSON response safely
def extract_json(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

@app.route('/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    try:
        data = request.get_json()
        if not data or not data.get("message"):
            return jsonify({
                "status": "error",
                "error": "Please describe your symptoms",
                "solution": "Include details like: 'headache for 3 days with fever'"
            }), 400

        symptoms = data["message"]
        age = data.get("age")
        gender = data.get("gender")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    generate_medical_prompt(symptoms, age, gender),
                    generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": 2000
                    }
                )

                response_data = extract_json(response.text)

                # Validate structure
                if "analysis" not in response_data or "status" not in response_data:
                    raise ValueError("Missing required fields in AI response")

                return jsonify(response_data)

            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    raise

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}\nInput: {data}")
        return jsonify({
            "status": "error",
            "error": "We couldn't analyze your symptoms",
            "details": "Our medical engine encountered an issue",
            "solution": "Please try again with more detailed symptoms",
            "troubleshooting": [
                "Rephrase your symptoms (e.g., 'fever for 2 days with headache')",
                "Add duration and severity (e.g., 'severe pain since yesterday')",
                "Try again in 5 minutes"
            ],
            "timestamp": datetime.datetime.now().isoformat()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "operational",
        "model": "gemini-pro",
        "timestamp": datetime.datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
