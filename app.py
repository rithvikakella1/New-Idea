import os
from openai import OpenAI
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load OpenAI key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class NoteInput(BaseModel):
    note: str

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

# Core logic
def extract_medical_codes(note: str) -> str:
    full_prompt = prompt_template + note.strip()

    client = OpenAI()  # uses OPENAI_API_KEY from env
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful medical coding assistant."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content



# Root health check
@app.get("/")
def root():
    return {"message": "Hello from New Idea!"}

# Main endpoint
@app.post("/api/extract")
def api_extract(input: NoteInput):
    if not input.note.strip():
        raise HTTPException(status_code=400, detail="No clinical note provided.")
    try:
        result = extract_medical_codes(input.note)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
