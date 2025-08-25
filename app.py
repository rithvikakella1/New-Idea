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
You are a professional medical coder. Given a clinical note, output the codes relevant to the medical billing claim (CMS-1500/UB-04) in TWO sections:

1) final: Only claim-ready, billable codes that would be placed on the claim today.
2) additional: Other plausible codes that could appear depending on payer rules, documentation details, or coder discretion (do NOT include differentials or non-billable items).

Rules for both sections:
- ICD-10-CM: Include only billable diagnosis codes pertinent to the encounter.
- CPT/HCPCS: Include procedures/services actually performed or likely claimable; add modifiers (e.g., 25, 59, RT/LT) and quantity where applicable.
- For each CPT line, include diagnosis_pointers referencing ICD-10 codes from the same section that justify the service.
- Exclude suspected conditions without confirmation, historical problems not impacting care today, and generic screening unless clearly applicable.
- Deduplicate codes within each section.

Respond with a pure JSON object only, with this shape:
{
  "final": [
    {
      "type": "Diagnosis",
      "code_type": "ICD-10",
      "code": "<ICD-10-CM code>",
      "description": "<billable diagnosis description>",
      "reasoning": "<brief justification>"
    },
    {
      "type": "Procedure",
      "code_type": "CPT",
      "code": "<CPT/HCPCS code>",
      "description": "<procedure description>",
      "modifiers": ["<modifier>"],
      "quantity": 1,
      "diagnosis_pointers": ["<ICD-10 code>", "<ICD-10 code>"],
      "reasoning": "<brief justification>"
    }
  ],
  "additional": [
    // Same object shapes as in "final". Keep to likely, plausible claim codes only.
  ]
}

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

    def sort_key(item):
        if isinstance(item, dict):
            modifiers = item.get("modifiers") or []
            if isinstance(modifiers, list):
                modifiers_key = ",".join(map(str, modifiers))
            else:
                modifiers_key = str(modifiers)

            diagnosis_pointers = item.get("diagnosis_pointers") or []
            if isinstance(diagnosis_pointers, list):
                pointers_key = ",".join(map(str, diagnosis_pointers))
            else:
                pointers_key = str(diagnosis_pointers)

            return (
                str(item.get("type", "")),
                str(item.get("code_type", "")),
                str(item.get("code", "")),
                str(item.get("description", "")),
                modifiers_key,
                str(item.get("quantity", "")),
                pointers_key,
                str(item.get("reasoning", "")),
            )
        return str(item)

    try:
        data = json.loads(candidate)
        if isinstance(data, list):
            sorted_list = sorted(data, key=sort_key)
            return json.dumps({"final": sorted_list, "additional": []}, sort_keys=True, ensure_ascii=False)
        if isinstance(data, dict):
            final_list = data.get("final") or []
            additional_list = data.get("additional") or []
            if isinstance(final_list, list):
                final_list = sorted(final_list, key=sort_key)
            if isinstance(additional_list, list):
                additional_list = sorted(additional_list, key=sort_key)
            return json.dumps({"final": final_list, "additional": additional_list}, sort_keys=True, ensure_ascii=False)
        return json.dumps({"final": [], "additional": []}, sort_keys=True, ensure_ascii=False)
    except Exception:
        return json.dumps({"final": [], "additional": []}, sort_keys=True, ensure_ascii=False)


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
