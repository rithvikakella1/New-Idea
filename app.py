import os
import json
import re
import base64
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # loads .env before any os.getenv() calls

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from openai import OpenAI
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import bcrypt
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from jose import JWTError, jwt

# ── SECURITY CONFIG ──────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY") or secrets.token_hex(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

_enc_env = os.getenv("ENCRYPTION_KEY", "")
if _enc_env:
    _raw = base64.b64decode(_enc_env + "==")
    ENCRYPTION_KEY = (_raw + b"\x00" * 32)[:32]
else:
    ENCRYPTION_KEY = secrets.token_bytes(32)

# ── USER STORE (file-based) ───────────────────────────────────────────────────
USERS_FILE = os.path.join(BASE_DIR, "users.json")

def _load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def _save_users(users: dict) -> None:
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

_users = _load_users()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "haiven2025")
if ADMIN_USERNAME not in _users:
    _users[ADMIN_USERNAME] = {
        "hash": bcrypt.hashpw(ADMIN_PASSWORD.encode(), bcrypt.gensalt()).decode(),
        "full_name": "Admin",
    }
    _save_users(_users)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

# ── RATE LIMITER ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── HIPAA SECURITY HEADERS ────────────────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"
    return response

# ── AES-256-GCM UTILITIES ─────────────────────────────────────────────────────
def aes_encrypt(plaintext: str) -> str:
    aesgcm = AESGCM(ENCRYPTION_KEY)
    nonce = secrets.token_bytes(12)
    ct = aesgcm.encrypt(nonce, plaintext.encode(), None)
    return base64.b64encode(nonce + ct).decode()

def aes_decrypt(token: str) -> str:
    raw = base64.b64decode(token)
    nonce, ct = raw[:12], raw[12:]
    return AESGCM(ENCRYPTION_KEY).decrypt(nonce, ct, None).decode()

# ── JWT AUTH ──────────────────────────────────────────────────────────────────
def authenticate_user(username: str, password: str) -> bool:
    users = _load_users()
    user = users.get(username)
    if not user:
        return False
    return bcrypt.checkpw(password.encode(), user["hash"].encode())

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise exc
        return username
    except JWTError:
        raise exc

# ── OPENAI CLIENT ─────────────────────────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── CONFIDENCE THRESHOLD ──────────────────────────────────────────────────────
# Codes below this threshold are moved to suggested_codes at parse time.
# Raise this value to increase precision; lower it to increase recall.
CONFIRMED_CONFIDENCE_THRESHOLD = 0.75

# ── PHYSICIAN BILLING PROMPT ──────────────────────────────────────────────────
# NOTE: Uses a two-pass chain-of-thought style:
#   1. The system message instructs the model to reason carefully before emitting JSON.
#   2. The user prompt contains strict rules + few-shot HCPCS examples to anchor
#      the model's understanding of HCPCS Level II codes alongside ICD-10 and CPT.

SYSTEM_PROMPT = """You are a board-certified professional medical coder and physician billing specialist with 20+ years of experience in ICD-10-CM, ICD-10-PCS, CPT, and HCPCS Level II coding.

PRECISION RULES — follow these exactly to achieve ≥90% coding accuracy:
1. Only assign a code when there is EXPLICIT, UNAMBIGUOUS documentation supporting it. When in doubt, move it to suggested_codes.
2. For ICD-10-CM: always code to the highest specificity — include 7th character, laterality, episode of care, and severity where required. A truncated code (e.g., S52 without full extension) is WRONG.
3. For CPT: verify that the procedure is fully documented (operative note, procedure note, or attending attestation). Do not infer a procedure from a diagnosis alone.
4. For HCPCS Level II: assign codes for durable medical equipment (DME), orthotics/prosthetics, ambulance services, drugs administered in the office (J-codes), supplies (A-codes), and other non-physician services. Only assign when the item/service is explicitly documented as provided or ordered.
5. NEVER code "possible," "probable," "suspected," "rule out," or "likely" conditions as confirmed diagnoses.
6. Apply correct sequencing: principal/primary diagnosis first, then complications, then comorbidities.
7. Set confidence as a strict self-assessment:
   - 0.90–1.00: Code is exact, unambiguous, and fully documented — safe to bill.
   - 0.75–0.89: Code is correct but documentation has minor gaps — bill with addendum recommended.
   - <0.75: Too uncertain — place in suggested_codes instead.
8. Never hallucinate codes. If you are uncertain of the exact code, use suggested_codes with documentation_needed.

HCPCS LEVEL II EXAMPLES (use these as anchors):
- E0601 — Continuous positive airway pressure (CPAP) device
- L3000 — Foot insert, removable, molded to patient model
- J0696 — Injection, ceftriaxone sodium, per 250mg
- A4570 — Splint
- K0001 — Standard manual wheelchair
- G0008 — Administration of influenza virus vaccine

You MUST respond ONLY with valid JSON — no markdown fences, no prose, no explanation outside the JSON object.
"""

PROMPT_TEMPLATE = """Extract all billable medical codes from the clinical note below. Include ICD-10-CM diagnosis codes, CPT procedure codes, AND HCPCS Level II codes (DME, supplies, drugs, orthotics, ambulance, vaccines, etc.).

Return this exact JSON structure:
{
  "confirmed_codes": [
    {
      "type": "Diagnosis | Procedure | Supply | Drug | DME | Orthotic | Other",
      "code_type": "ICD-10-CM | ICD-10-PCS | CPT | HCPCS",
      "code": "<exact full code with all required characters>",
      "description": "<full official description>",
      "reasoning": "<quote or paraphrase the specific documentation that supports this code>",
      "confidence": <float 0.75–1.0>,
      "documentation_strength": "strong | moderate | weak",
      "billing_priority": "primary | secondary | procedural | supplemental"
    }
  ],
  "suggested_codes": [
    {
      "code_type": "ICD-10-CM | ICD-10-PCS | CPT | HCPCS",
      "code": "<exact code>",
      "description": "<full official description>",
      "reason_suggested": "<why this code may apply>",
      "documentation_needed": "<what additional documentation would confirm this code>"
    }
  ]
}

Rules:
- confirmed_codes: only codes with confidence ≥ 0.75 and strong/moderate documentation.
- suggested_codes: codes that clinically likely apply but need more documentation, OR any code where confidence < 0.75.
- Do NOT omit HCPCS codes for any DME, supply, injectable drug, or non-physician service documented in the note.
- If no HCPCS codes apply, return an empty array for that section — do not fabricate codes.

Clinical Note:
"""

# ── RESPONSE PARSING ──────────────────────────────────────────────────────────
def _parse_llm_response(text: str) -> dict:
    cleaned = re.sub(r"```json|```", "", text).strip()

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
    else:
        s, e = cleaned.find("["), cleaned.rfind("]")
        if s != -1 and e != -1:
            try:
                arr = json.loads(cleaned[s:e + 1])
                return {"confirmed_codes": arr, "suggested_codes": []}
            except Exception:
                pass
        return {"confirmed_codes": [], "suggested_codes": [], "raw": cleaned}

    try:
        data = json.loads(candidate)

        confirmed = []
        downgraded = []

        for item in data.get("confirmed_codes", []):
            try:
                item["confidence"] = round(float(item.get("confidence", 0)), 2)
            except Exception:
                item["confidence"] = 0.0

            # Enforce threshold: low-confidence confirmed codes move to suggested
            if item["confidence"] < CONFIRMED_CONFIDENCE_THRESHOLD:
                downgraded.append({
                    "code_type": item.get("code_type", ""),
                    "code": item.get("code", ""),
                    "description": item.get("description", ""),
                    "reason_suggested": f"Confidence {item['confidence']} below threshold — {item.get('reasoning', '')}",
                    "documentation_needed": "Strengthen documentation to support billing.",
                })
            else:
                confirmed.append(item)

        data["confirmed_codes"] = confirmed
        data["suggested_codes"] = data.get("suggested_codes", []) + downgraded

        return data

    except Exception:
        return {"confirmed_codes": [], "suggested_codes": [], "raw": cleaned}


def extract_medical_codes(note: str) -> dict:
    response = client.chat.completions.create(
        # gpt-4o substantially outperforms gpt-4o-mini on medical coding precision.
        # For cost-sensitive deployments, use "gpt-4o-mini" and accept ~5–10% lower accuracy.
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": PROMPT_TEMPLATE + note.strip(),
            },
        ],
        temperature=0,       # deterministic output — critical for coding accuracy
        top_p=1,
        response_format={"type": "json_object"},
    )
    return _parse_llm_response(response.choices[0].message.content)


# ── PAGE ROUTES ───────────────────────────────────────────────────────────────
def _serve(filename: str) -> str:
    with open(os.path.join(BASE_DIR, filename), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/", response_class=HTMLResponse)
async def serve_landing():
    return _serve("index.html")


@app.get("/app", response_class=HTMLResponse)
async def serve_app():
    return _serve("app.html")


@app.get("/login", response_class=HTMLResponse)
async def serve_login():
    return _serve("login.html")


@app.get("/signup", response_class=HTMLResponse)
async def serve_signup():
    return _serve("signup.html")


# ── API ROUTES ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/token")
@limiter.limit("5/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    if not authenticate_user(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    users = _load_users()
    full_name = users.get(form_data.username, {}).get("full_name", "") or form_data.username
    token = create_access_token(
        data={"sub": form_data.username, "full_name": full_name},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": token, "token_type": "bearer"}


class RegisterInput(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None


@app.post("/api/register", status_code=201)
@limiter.limit("3/minute")
async def register(request: Request, body: RegisterInput):
    import re as _re
    if not _re.match(r'^[A-Za-z0-9_]{3,32}$', body.username):
        raise HTTPException(status_code=400, detail="Username must be 3–32 characters (letters, numbers, underscores).")
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    users = _load_users()
    if body.username in users:
        raise HTTPException(status_code=409, detail="Username already taken.")

    users[body.username] = {
        "hash": bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode(),
        "full_name": body.full_name or "",
    }
    _save_users(users)
    return {"message": "Account created successfully."}


class NoteInput(BaseModel):
    note: str


@app.post("/api/extract")
@limiter.limit("10/minute")
async def api_extract(
    request: Request,
    input: NoteInput,
    current_user: str = Depends(get_current_user),
):
    if not input.note.strip():
        raise HTTPException(status_code=400, detail="No clinical note provided.")
    try:
        result = extract_medical_codes(input.note)
        _ = aes_encrypt(input.note)  # audit log (persist in production)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
