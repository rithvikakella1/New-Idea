import os
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_landing():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import json
import re

class NoteInput(BaseModel):
    note: str

prompt_template = """
You are a professional medical coder. Given a clinical note, extract all relevant ICD-10 diagnosis codes and CPT procedure codes.

For each code, include a confidence score between 0 and 1 indicating how certain you are.

Respond strictly in this JSON format:
[
  {
    "type": "Diagnosis or Procedure",
    "code_type": "ICD-10 or CPT",
    "code": "<code>",
    "description": "<description>",
    "reasoning": "<short justification>",
    "confidence": <float between 0 and 1>
  }
]

Clinical Note:
"""

def _normalize_to_sorted_json(text: str) -> str:
    cleaned = re.sub(r"```json|```", "", text).strip()

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
    else:
        candidate = cleaned

    try:
        data = json.loads(candidate)

        def coerce(item):
            if isinstance(item, dict) and "confidence" in item:
                try:
                    item["confidence"] = float(item["confidence"])
                except:
                    item["confidence"] = 0.0
            return item

        def sort_key(item):
            if isinstance(item, dict):
                return (
                    str(item.get("type", "")),
                    str(item.get("code_type", "")),
                    str(item.get("code", "")),
                    str(item.get("confidence", "")),
                )
            return str(item)

        if isinstance(data, list):
            data = [coerce(x) for x in data]
            return json.dumps(sorted(data, key=sort_key), sort_keys=True, ensure_ascii=False)

        return cleaned
    except:
        return cleaned


def extract_medical_codes(note: str) -> str:
    full_prompt = prompt_template + note.strip()

    kwargs = dict(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a precise medical coding assistant. Always include confidence scores between 0 and 1."
            },
            {"role": "user", "content": full_prompt},
        ],
        temperature=0,
        top_p=1,
    )

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    return _normalize_to_sorted_json(content)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/extract")
def api_extract(input: NoteInput):
    if not input.note.strip():
        raise HTTPException(status_code=400, detail="No clinical note provided.")
    try:
        result = extract_medical_codes(input.note)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
