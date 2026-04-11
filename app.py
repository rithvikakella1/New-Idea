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
# JWT: set JWT_SECRET_KEY in env for persistence across restarts
SECRET_KEY = os.getenv("JWT_SECRET_KEY") or secrets.token_hex(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# AES-256 key (32 bytes). Set ENCRYPTION_KEY as base64-encoded 32 bytes in env.
# If unset, key is ephemeral (regenerated on restart) — fine for demo/dev.
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

# Seed the admin account from env on first run
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


# ── HIPAA SECURITY HEADERS (TLS enforcement signals) ─────────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    # HSTS: tell browsers to always use HTTPS (TLS 1.2+ enforced at transport)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"  # PHI must never be cached
    return response


# ── AES-256-GCM UTILITIES (data at rest) ─────────────────────────────────────
def aes_encrypt(plaintext: str) -> str:
    """Encrypt with AES-256-GCM. Returns base64(nonce + ciphertext)."""
    aesgcm = AESGCM(ENCRYPTION_KEY)
    nonce = secrets.token_bytes(12)  # 96-bit nonce recommended for GCM
    ct = aesgcm.encrypt(nonce, plaintext.encode(), None)
    return base64.b64encode(nonce + ct).decode()


def aes_decrypt(token: str) -> str:
    """Decrypt an AES-256-GCM token."""
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

# ── PHYSICIAN BILLING PROMPT ──────────────────────────────────────────────────
prompt_template = """You are a board-certified professional medical coder and physician billing specialist with 20+ years of experience in ICD-10-CM, ICD-10-PCS, and CPT/HCPCS Level II coding. You are preparing claims for insurance submission on behalf of the treating physician.

CODING STANDARDS YOU MUST FOLLOW:
- ICD-10-CM Official Guidelines for Coding and Reporting (current year)
- AMA CPT® codebook rules and parenthetical notes
- CMS National Correct Coding Initiative (NCCI) edits
- Code to the HIGHEST level of specificity: include 7th character, laterality, severity, and episode of care where applicable
- NEVER code "possible," "probable," "suspected," or "rule out" conditions as confirmed diagnoses — only code what is documented as confirmed
- Include ALL comorbidities and secondary conditions that affect treatment complexity or medical decision-making
- Apply correct sequencing: principal/primary diagnosis first, then complications, then comorbidities
- Consider E&M level codes if the note supports an office or hospital visit

BILLING PRIORITIES:
1. PRIMARY diagnosis (ICD-10-CM) — the main reason for the encounter
2. SECONDARY diagnoses (ICD-10-CM) — comorbidities affecting treatment
3. PROCEDURE codes (CPT) — all billable services rendered
4. Modifier applicability — flag bilateral, multiple procedures, or assistant surgeon scenarios

For each CONFIRMED code, provide:
- confidence: 0.0–1.0 (certainty this code is correct and billable from the documentation alone)
- documentation_strength: "strong" (explicitly documented) | "moderate" (implied but inferrable) | "weak" (risky to bill without addendum)
- billing_priority: "primary" | "secondary" | "procedural" | "supplemental"

Also provide a "suggested_codes" array for codes that clinically likely apply but require physician clarification or additional documentation before submitting to insurance.

Respond ONLY with valid JSON — no markdown fences, no extra text:
{
  "confirmed_codes": [
    {
      "type": "Diagnosis or Procedure",
      "code_type": "ICD-10 or CPT",
      "code": "<exact code>",
      "description": "<full official description>",
      "reasoning": "<specific justification drawn from the clinical note>",
      "confidence": <float 0.0–1.0>,
      "documentation_strength": "strong or moderate or weak",
      "billing_priority": "primary or secondary or procedural or supplemental"
    }
  ],
  "suggested_codes": [
    {
      "code_type": "ICD-10 or CPT",
      "code": "<exact code>",
      "description": "<full official description>",
      "reason_suggested": "<why this code may apply based on clinical context>",
      "documentation_needed": "<what additional documentation would confirm and allow billing>"
    }
  ]
}

Clinical Note:
"""


# ── RESPONSE PARSING ──────────────────────────────────────────────────────────
def _parse_llm_response(text: str) -> dict:
    cleaned = re.sub(r"```json|```", "", text).strip()

    # Prefer JSON object
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
    else:
        # Fallback: legacy array format
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
        for item in data.get("confirmed_codes", []):
            try:
                item["confidence"] = round(float(item.get("confidence", 0)), 2)
            except Exception:
                item["confidence"] = 0.0
        return data
    except Exception:
        return {"confirmed_codes": [], "suggested_codes": [], "raw": cleaned}


def extract_medical_codes(note: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise physician billing specialist and medical coding expert. "
                    "Always respond with valid JSON only. All confidence values must be floats between 0 and 1."
                ),
            },
            {"role": "user", "content": prompt_template + note.strip()},
        ],
        temperature=0,
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
        # Encrypt a copy of the note for audit (HIPAA data-at-rest requirement)
        _ = aes_encrypt(input.note)  # in production, persist this encrypted record
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
