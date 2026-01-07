from typing import Optional, List, Mapping, Any
import os
import requests

from langchain.llms.base import LLM


class GroqLLM(LLM):
    """Minimal LangChain LLM wrapper that calls a Groq-compatible HTTP API.

    Configure via environment variables: `GROQ_API_KEY`, `GROQ_API_URL`, `GROQ_MODEL`.
    """

    model_name: str = "groq-1"
    api_key: Optional[str] = None
    api_url: Optional[str] = None

    def __init__(self, model_name: str = None, api_key: str = None, api_url: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name or os.getenv("GROQ_MODEL", self.model_name)
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.api_url = api_url or os.getenv("GROQ_API_URL")
        if not self.api_key or not self.api_url:
            raise ValueError("GROQ_API_KEY and GROQ_API_URL must be set (see .env.example)")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model_name, "input": prompt, "max_tokens": 512}
        resp = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Try to handle several common response shapes
        if isinstance(data, dict):
            # openai-like choices
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                first = data["choices"][0]
                for key in ("text", "message", "content"):
                    if key in first:
                        val = first[key]
                        if isinstance(val, str):
                            return val
                        if isinstance(val, dict) and "text" in val:
                            return val["text"]

            for key in ("text", "result", "output"):
                if key in data and isinstance(data[key], str):
                    return data[key]

            if "output" in data and isinstance(data["output"], list) and data["output"]:
                out0 = data["output"][0]
                if isinstance(out0, dict):
                    for k in ("text", "content", "generated_text"):
                        if k in out0 and isinstance(out0[k], str):
                            return out0[k]

        return resp.text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}
