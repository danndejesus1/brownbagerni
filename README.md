# LangChain + Groq minimal chatbot

Quick scaffold to run a LangChain-based chatbot backed by a Groq-compatible API.

Setup
- Copy `.env.example` to `.env` and set `GROQ_API_KEY`, `GROQ_API_URL` and optionally `GROQ_MODEL`.
- Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run (CLI):

```bash
python -m src.chatbot.main
```

Endpoints
- Use the CLI runner: `python -m src.chatbot.main` to chat interactively.
