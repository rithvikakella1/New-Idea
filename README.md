# MoBILLity – AI Medical Code Extractor

AI-powered ICD-10 and CPT code extraction from clinical notes, built with FastAPI and GPT-4o-mini.

## Prerequisites

- Python 3.9+
- An OpenAI API key

---

## Setup (fresh machine)

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd New-Idea
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate it

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Create your `.env` file

Create a file called `.env` in the project root with the following:

```
OPENAI_API_KEY=sk-your-key-here
ADMIN_USERNAME=admin
ADMIN_PASSWORD=choose-a-password
JWT_SECRET_KEY=any-long-random-string
```

- `OPENAI_API_KEY` — required, get it from platform.openai.com
- `ADMIN_USERNAME` / `ADMIN_PASSWORD` — the first admin account (seeded automatically on first run)
- `JWT_SECRET_KEY` — any long random string, used to sign login tokens

### 6. Run the server
```bash
uvicorn app:app --reload
```

### 7. Open in browser
```
http://localhost:8000
```

---

## Pages

| URL | Description |
|---|---|
| `http://localhost:8000` | Landing page |
| `http://localhost:8000/signup` | Create an account |
| `http://localhost:8000/login` | Sign in |
| `http://localhost:8000/app` | Code extractor (requires login) |

---

## Features

- **AI code extraction** — ICD-10-CM and CPT codes from free-text clinical notes
- **Physician billing prompt** — codes extracted with billing priority, confidence scores, and documentation strength ratings
- **Suggested codes** — additional codes flagged for physician review
- **Speech-to-text** — dictate clinical notes directly (Chrome/Edge)
- **Authentication** — JWT-based login with bcrypt password hashing
- **Rate limiting** — 10 req/min on extraction, 5 req/min on login, 3 req/min on registration
- **HIPAA-aligned security** — AES-256-GCM encryption utilities, HSTS headers, `Cache-Control: no-store`

---

## Notes

- `.env` is gitignored — never commit your API key
- User accounts are stored in `users.json` (also gitignored — add it if you haven't)
- The admin account from `.env` is seeded automatically on first run
