import os

from fastapi import FastAPI
from pydantic import BaseModel

from .groq_llm import GroqLLM

app = FastAPI()


class ChatRequest(BaseModel):
    prompt: str


@app.post("/chat")
def chat(req: ChatRequest):
    llm = GroqLLM()
    resp = llm(req.prompt)
    return {"response": resp}


if __name__ == "__main__":
    # Simple CLI runner
    llm = GroqLLM()
    print("Groq LangChain Chat (type 'exit' to quit)")
    try:
        while True:
            prompt = input("You: ")
            if not prompt:
                continue
            if prompt.strip().lower() in ("exit", "quit"):
                break
            resp = llm(prompt)
            print("Assistant:", resp)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting")
