"""FastAPI entrypoint for exchanging workflow ids for ChatKit client secrets."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Mapping

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

DEFAULT_CHATKIT_BASE = "https://api.openai.com"
SESSION_COOKIE_NAME = "chatkit_session_id"
SESSION_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 30  # 30 days

app = FastAPI(title="Managed ChatKit Session API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Mapping[str, str]:
    return {"status": "ok"}


@app.post("/api/create-session")
async def create_session(request: Request) -> JSONResponse:
    """Exchange a workflow id for a ChatKit client secret."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return respond({"error": "Missing OPENAI_API_KEY environment variable"}, 500)

    body = await read_json_body(request)
    workflow_id = resolve_workflow_id(body)
    if not workflow_id:
        return respond({"error": "Missing workflow id"}, 400)

    user_id, cookie_value = resolve_user(request.cookies)
    api_base = chatkit_api_base()

    try:
        async with httpx.AsyncClient(base_url=api_base, timeout=10.0) as client:
            upstream = await client.post(
                "/v1/chatkit/sessions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "OpenAI-Beta": "chatkit_beta=v1",
                    "Content-Type": "application/json",
                },
                json={"workflow": {"id": workflow_id}, "user": user_id},
            )
    except httpx.RequestError as error:
        return respond(
            {"error": f"Failed to reach ChatKit API: {error}"},
            502,
            cookie_value,
        )

    payload = parse_json(upstream)
    if not upstream.is_success:
        message = None
        if isinstance(payload, Mapping):
            message = payload.get("error")
        message = message or upstream.reason_phrase or "Failed to create session"
        return respond({"error": message}, upstream.status_code, cookie_value)

    client_secret = None
    expires_after = None
    if isinstance(payload, Mapping):
        client_secret = payload.get("client_secret")
        expires_after = payload.get("expires_after")

    if not client_secret:
        return respond(
            {"error": "Missing client secret in response"},
            502,
            cookie_value,
        )

    return respond(
        {"client_secret": client_secret, "expires_after": expires_after},
        200,
        cookie_value,
    )


def respond(
    payload: Mapping[str, Any], status_code: int, cookie_value: str | None = None
) -> JSONResponse:
    response = JSONResponse(payload, status_code=status_code)
    if cookie_value:
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=cookie_value,
            max_age=SESSION_COOKIE_MAX_AGE_SECONDS,
            httponly=True,
            samesite="lax",
            secure=is_prod(),
            path="/",
        )
    return response


def is_prod() -> bool:
    env = (os.getenv("ENVIRONMENT") or os.getenv("NODE_ENV") or "").lower()
    return env == "production"


async def read_json_body(request: Request) -> Mapping[str, Any]:
    raw = await request.body()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def resolve_workflow_id(body: Mapping[str, Any]) -> str | None:
    workflow = body.get("workflow", {})
    workflow_id = None
    if isinstance(workflow, Mapping):
        workflow_id = workflow.get("id")
    workflow_id = workflow_id or body.get("workflowId")
    env_workflow = os.getenv("CHATKIT_WORKFLOW_ID") or os.getenv(
        "VITE_CHATKIT_WORKFLOW_ID"
    )
    if not workflow_id and env_workflow:
        workflow_id = env_workflow
    if workflow_id and isinstance(workflow_id, str) and workflow_id.strip():
        return workflow_id.strip()
    return None


def resolve_user(cookies: Mapping[str, str]) -> tuple[str, str | None]:
    existing = cookies.get(SESSION_COOKIE_NAME)
    if existing:
        return existing, None
    user_id = str(uuid.uuid4())
    return user_id, user_id


def chatkit_api_base() -> str:
    return (
        os.getenv("CHATKIT_API_BASE")
        or os.getenv("VITE_CHATKIT_API_BASE")
        or DEFAULT_CHATKIT_BASE
    )


def parse_json(response: httpx.Response) -> Mapping[str, Any]:
    try:
        parsed = response.json()
        return parsed if isinstance(parsed, Mapping) else {}
    except (json.JSONDecodeError, httpx.DecodingError):
        return {}
# =========================
# Telegram bot integration
# =========================

import html
from fastapi import BackgroundTasks, HTTPException

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "ege_history_secret_2026")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

TELEGRAM_HISTORY = {}

TELEGRAM_SYSTEM_PROMPT = """
Ты — AI-тренажёр и тьютор по подготовке к ЕГЭ по истории России.

Твоя задача — тренировать ученика в формате ЕГЭ: задавать задания, проверять ответы,
объяснять ошибки, давать повторные задания и помогать запоминать материал.

Правила:
1. Отвечай на русском языке.
2. Работай как спокойный и сильный репетитор по истории ЕГЭ.
3. Не выдумывай факты, даты, имена и события.
4. Если ученик ошибся, сначала покажи правильный ответ, затем объясни ошибку.
5. Для заданий с точным ответом всегда сверяй ответ ученика и правильный ключ.
6. Никогда не пиши «верно», если ответ ученика не совпадает с правильным ключом.
7. После ошибки дай короткое мини-задание на закрепление.
8. Не превращай ответ в длинную лекцию, если ученик не просит.
9. Если ученик пишет «диагностика», начни диагностику из 10 заданий.
10. Если ученик пишет «тема: ...», тренируй по этой теме заданиями по одному.
"""

async def tg_api(method: str, payload: dict):
    if not TELEGRAM_BOT_TOKEN:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        await client.post(url, json=payload)


async def tg_send_message(chat_id: int, text: str):
    safe_text = html.escape(text)

    # Telegram не любит очень длинные сообщения, режем на части.
    chunks = [safe_text[i:i + 3900] for i in range(0, len(safe_text), 3900)]

    for chunk in chunks:
        await tg_api("sendMessage", {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "HTML"
        })


async def tg_send_typing(chat_id: int):
    await tg_api("sendChatAction", {
        "chat_id": chat_id,
        "action": "typing"
    })


def extract_response_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]

    parts = []

    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                parts.append(text)

    return "\n".join(parts).strip() or "Не удалось получить ответ."


async def ask_openai_for_telegram(chat_id: int, user_text: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return "Ошибка: на сервере не задан OPENAI_API_KEY."

    normalized = user_text.strip().lower()

    if normalized in ["/reset", "сброс", "начать заново"]:
        TELEGRAM_HISTORY[chat_id] = []
        return "Диалог сброшен. Напишите «диагностика» или выберите тему, например: «тема: Смута»."

    history = TELEGRAM_HISTORY.get(chat_id, [])

    input_messages = history + [
        {"role": "user", "content": user_text}
    ]

    body = {
        "model": OPENAI_MODEL,
        "instructions": TELEGRAM_SYSTEM_PROMPT,
        "input": input_messages
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=body
        )

    if response.status_code >= 400:
        return f"Ошибка OpenAI API: {response.status_code}\n{response.text[:1000]}"

    payload = response.json()
    answer = extract_response_text(payload)

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})
    TELEGRAM_HISTORY[chat_id] = history[-20:]

    return answer


async def process_telegram_update(update: dict):
    message = update.get("message") or update.get("edited_message")

    if not message:
        return

    chat = message.get("chat", {})
    chat_id = chat.get("id")
    text = message.get("text", "")

    if not chat_id:
        return

    if not text:
        await tg_send_message(chat_id, "Пока я работаю только с текстовыми сообщениями.")
        return

    if text.startswith("/start"):
        await tg_send_message(
            chat_id,
            "Здравствуйте! Я тренажёр по истории ЕГЭ.\n\n"
            "Можно написать:\n"
            "• диагностика\n"
            "• тема: Смута\n"
            "• потренируй даты Петра I\n"
            "• проверь мой ответ: ...\n\n"
            "Для сброса диалога: /reset"
        )
        return

    await tg_send_typing(chat_id)

    answer = await ask_openai_for_telegram(chat_id, text)
    await tg_send_message(chat_id, answer)


@app.post("/telegram/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request, background_tasks: BackgroundTasks):
    if secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    update = await request.json()
    background_tasks.add_task(process_telegram_update, update)

    return {"ok": True}
