import os
from openai import OpenAI
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def landing():
    with open("landing.html", "r") as f:
        return f.read()

# Load OpenAI key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
You are a professional medical coder. Given a clinical note, extract **all possible** relevant ICD-10 diagnosis codes and CPT procedure codes. Include codes for symptoms, comorbidities, tests, and treatments, not just the primary diagnosis.

Respond in this JSON format:
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
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a thorough and accurate medical coding assistant. Include all diagnoses (including comorbidities and risk factors), procedures (including lab tests and imaging), and explain each with short reasoning."
},
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
