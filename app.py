import os
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

# Prompt template
prompt_template = """
You are a medical coding assistant. Given a clinical note, extract relevant ICD-10 and CPT codes.
Respond in JSON with the following format:
[
  {
    "type": "Diagnosis" or "Procedure",
    "code_type": "ICD-10" or "CPT",
    "code": "<code>",
    "description": "<description>",
    "reasoning": "<short justification>"
  }
]

Clinical Note:
"""

# Core function
def extract_medical_codes(note):
    full_prompt = prompt_template + note.strip()
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Or "gpt-4o" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful medical coding assistant."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.2
    )
    reply = response.choices[0].message.content
    return reply

# API route
@app.route("/api/extract", methods=["POST"])
def api_extract():
    data = request.get_json()
    note = data.get("note", "")
    if not note:
        return jsonify({"error": "No clinical note provided"}), 400
    try:
        codes = extract_medical_codes(note)
        return jsonify({"result": codes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Use Waitress for production
if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 10000))
    serve(app, host="0.0.0.0", port=port)
