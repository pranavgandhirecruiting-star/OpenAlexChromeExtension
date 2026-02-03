# OpenAlex Recruiting Tool

A Chrome extension + local backend for finding ML engineering candidates from academic literature using OpenAlex and GitHub APIs.

## How It Works

1. **You provide seed papers** (OpenAlex Work IDs) — e.g., a foundational paper on distributed training
2. **The pipeline finds all citing papers** and extracts their authors
3. **Authors are scored** using POS/NEG keyword matching against their publication history (ML infra, optimization, frameworks vs. pure theory, crypto, etc.)
4. **An engineering gate** filters for candidates with systems/implementation signal
5. **Optional GitHub sniff test** checks each candidate's GitHub profile for industry activity signals
6. **Results exported as XLSX** with scores, bucket coverage, LinkedIn X-ray links, and GitHub notes

## Architecture

```
Chrome Extension (side panel UI)
        ↓ POST /run
Local FastAPI Backend (127.0.0.1:8787)
        ↓
pipeline.py → OpenAlex API + GitHub API
        ↓
XLSX output → downloaded via /download/<job_id>
```

## Setup

### 1. Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set your email for OpenAlex polite pool (10x rate limit)
export OPENALEX_EMAIL="your.email@example.com"

# Optional: GitHub token for sniff test
export GITHUB_TOKEN="ghp_your_token_here"

# Start the server
./run_backend.sh
# or: uvicorn app:app --host 127.0.0.1 --port 8787
```

### 2. Chrome Extension

1. Open `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `chrome-extension/` folder
5. The extension appears in your toolbar — click to open the side panel

### 3. Usage

1. Find a seed paper on [OpenAlex](https://openalex.org) (the Work ID looks like `W4318541647`)
2. Paste the ID(s) into the extension
3. Set your engineering gate score (0 = no filter, higher = stricter)
4. Optionally enable GitHub sniff test and provide a PAT
5. Click **Run Search**
6. Download the XLSX when complete

## Scoring

### Keyword Buckets

Authors are scored across 6 dimensions:
- **infra_prod** — training/inference infrastructure, MLOps, deployment
- **math_optimization** — optimization theory, gradient methods, numerical methods
- **ml_frameworks** — PyTorch, JAX, TensorFlow, XLA, custom operators
- **research_workflows** — reproducibility, experiment tracking, ablation studies
- **inventive_thinking** — SOTA engagement, comparative analysis, design decisions
- **software_artifacts** — open source, libraries, toolkits, implementations

### GitHub Sniff Test

When enabled, each candidate's GitHub profile is scored on:
- Bio/company keywords (industry vs. academia)
- Follower count and public repo count
- Recent activity (last 12-24 months)
- Star counts on repos
- Contributions to known ML orgs (pytorch, openai, nvidia, etc.)

## Files

```
backend/
  app.py              # FastAPI server with /run, /status, /download endpoints
  pipeline.py         # Core OpenAlex + GitHub recruiting pipeline
  requirements.txt    # Python dependencies
  run_backend.sh      # Startup script

chrome-extension/
  manifest.json       # MV3 Chrome extension manifest
  background.js       # Service worker — manages jobs, polling, downloads
  panel.html          # Side panel UI
  panel.js            # Side panel logic
  popup.html          # Toolbar popup (opens side panel)
  popup.js            # Popup logic
```

## License

MIT
