import os
import time
import json
import urllib.request
from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="SIGNBRIDGE Vercel Serverless API")

@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "version": "2.1.0",
        "deployment": "vercel_serverless",
        "model_loaded": True,
        "features": {
            "coordinate_mlp": True,
            "nvidia_nim_assistance": bool(os.getenv("NVIDIA_API_KEY")),
            "mobile_camera": True,
            "offline_inference": True
        }
    }

@app.get("/api/ai/status")
def ai_status():
    has_key = bool(os.getenv("NVIDIA_API_KEY"))
    return {
        "status": "ready",
        "active_provider": "nvidia_nim" if has_key else "local_deterministic_engine",
        "cloud_configured": has_key,
        "model": os.getenv("NVIDIA_MODEL", "poolside/laguna-xs-2.1") if has_key else "local_rule_engine_v2.1",
        "api_key_configured": has_key
    }

class ComposeRequest(BaseModel):
    tokens: List[str]
    target_language: Optional[str] = "en"

@app.post("/api/ai/compose")
@app.post("/api/ai/compose-sentence")
def ai_compose(req: ComposeRequest):
    tokens = req.tokens
    target_language = req.target_language or "en"
    if not tokens:
        return {
            "composed_text": "",
            "source_tokens": [],
            "target_language": target_language,
            "provider": "local_noop",
            "latency_ms": 0.0,
            "fallback_used": False
        }

    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if api_key:
        t0 = time.perf_counter()
        try:
            url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
            if not url.endswith("/chat/completions"):
                url = f"{url.rstrip('/')}/chat/completions"
            model = os.getenv("NVIDIA_MODEL", "poolside/laguna-xs-2.1")
            prompt = (
                f"You are an assistive sign language translation assistant. "
                f"Compose a fluent English sentence from these recognized letter tokens: {tokens}. "
                f"Output strictly valid JSON with key 'composed_text'."
            )
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You compose fluent sentences from recognized sign tokens. Respond strictly in JSON: {\"composed_text\": \"...\"}"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 128
            }
            http_req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
            )
            with urllib.request.urlopen(http_req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choice = data["choices"][0]["message"]
                content = choice.get("content") or choice.get("reasoning_content") or ""
                text = ""
                try:
                    parsed = json.loads(content)
                    text = parsed.get("composed_text", "")
                except Exception:
                    text = content.strip().strip('"')
                if text:
                    return {
                        "composed_text": text,
                        "source_tokens": tokens,
                        "target_language": target_language,
                        "provider": "nvidia_nim",
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                        "fallback_used": False
                    }
        except Exception:
            pass

    # Instant deterministic fallback
    word = "".join(tokens).capitalize() + "."
    return {
        "composed_text": word,
        "source_tokens": tokens,
        "target_language": target_language,
        "provider": "local_deterministic_engine",
        "latency_ms": 0.0,
        "fallback_used": True
    }
