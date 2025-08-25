import os
from openai import OpenAI
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()  

# Serve static files from a folder if present
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the landing page
@app.get("/", response_class=HTMLResponse)
async def serve_landing():
    with open("landing.html", "r", encoding="utf-8") as f:
        return f.read()

# Load OpenAI key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import json
import re

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

def _normalize_to_sorted_json(text: str) -> str:
    # Remove code fences if present
    cleaned = re.sub(r"```json|```", "", text).strip()

    # Try to extract JSON array portion
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
    else:
        candidate = cleaned

    try:
        data = json.loads(candidate)
        if isinstance(data, list):
            # Sort list deterministically by key tuple if dicts
            def sort_key(item):
                if isinstance(item, dict):
                    return (
                        str(item.get("type", "")),
                        str(item.get("code_type", "")),
                        str(item.get("code", "")),
                        str(item.get("description", "")),
                        str(item.get("reasoning", "")),
                    )
                return str(item)

            sorted_list = sorted(data, key=sort_key)
            return json.dumps(sorted_list, sort_keys=True, ensure_ascii=False)
        # If not a list, fall back to cleaned string
        return cleaned
    except Exception:
        # If parsing fails, return cleaned text
        return cleaned


# Core logic
def extract_medical_codes(note: str) -> str:
    full_prompt = prompt_template + note.strip()

    # Build request with deterministic settings
    kwargs = dict(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a thorough and accurate medical coding assistant. "
                    "Include all diagnoses (including comorbidities and risk factors), "
                    "procedures (including lab tests and imaging), and explain each with short reasoning."
                ),
            },
            {"role": "user", "content": full_prompt},
        ],
        temperature=0,
        top_p=1,
        n=1,
        presence_penalty=0,
        frequency_penalty=0,
    )

    seed_env = os.getenv("OPENAI_SEED")
    used_seed = None
    if seed_env:
        try:
            used_seed = int(seed_env)
            kwargs["seed"] = used_seed
        except Exception:
            used_seed = None

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        # Fallback if the API rejects the seed parameter
        if "seed" in kwargs:
            try:
                kwargs.pop("seed", None)
                response = client.chat.completions.create(**kwargs)
            except Exception:
                raise e
        else:
            raise e

    content = response.choices[0].message.content
    return _normalize_to_sorted_json(content)



# Root health check (moved to /health to avoid / landing conflict)
@app.get("/health")
def health():
    return {"status": "ok"}

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
