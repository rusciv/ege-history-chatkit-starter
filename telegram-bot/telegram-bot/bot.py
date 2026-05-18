import os
import html
import httpx
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from openai import AsyncOpenAI

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "ege_history_secret")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
VECTOR_STORE_ID = os.environ.get("OPENAI_VECTOR_STORE_ID", "").strip()

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Простая память диалога. На бесплатном Render может сбрасываться после перезапуска.
CHAT_HISTORY = {}

SYSTEM_PROMPT = """
Ты — AI-тренажёр и тьютор по подготовке к ЕГЭ по истории России.

Твоя задача — тренировать ученика в формате ЕГЭ: задавать задания, проверять ответы,
объяснять ошибки, давать повторные задания и помогать запоминать материал.

Правила:
1. Отвечай на русском языке.
2. Работай как спокойный и сильный репетитор по истории ЕГЭ.
3. Не выдумывай факты, даты, имена и события.
4. Если ученик ошибся, сначала покажи правильный ответ, затем объясни ошибку.
5. Для заданий с точным ответом всегда сверяй:
   - ответ ученика;
   - правильный ответ;
   - вердикт.
6. Никогда не пиши «верно», если ответ ученика не совпадает с правильным ключом.
7. После ошибки дай короткое мини-задание на закрепление.
8. Не превращай ответ в длинную лекцию, если ученик не просит.
9. Если ученик пишет «диагностика», начни диагностику из 10 заданий:
   даты, хронология, персоналии, культура, причинно-следственные связи, мини-развёрнутый ответ.
10. Если ученик пишет «тема: ...», тренируй по этой теме заданиями по одному.
11. Если ученик пишет «/reset» или «сброс», начни диалог заново.

Формат проверки точных ответов:
Ваш ответ: ...
Правильный ответ: ...
Вердикт: верно / неверно.
Краткий разбор.
"""

async def telegram_api(method: str, payload: dict):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    async with httpx.AsyncClient(timeout=30) as http:
        await http.post(url, json=payload)

async def send_message(chat_id: int, text: str):
    # Telegram ограничивает длину сообщения, поэтому режем длинные ответы.
    chunks = [text[i:i+3900] for i in range(0, len(text), 3900)]
    for chunk in chunks:
        await telegram_api("sendMessage", {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "HTML"
        })

async def send_typing(chat_id: int):
    await telegram_api("sendChatAction", {
        "chat_id": chat_id,
        "action": "typing"
    })

async def ask_openai(chat_id: int, user_text: str) -> str:
    history = CHAT_HISTORY.get(chat_id, [])

    if user_text.strip().lower() in ["/reset", "сброс", "начать заново"]:
        CHAT_HISTORY[chat_id] = []
        return "Диалог сброшен. Напишите «диагностика» или выберите тему, например: «тема: Смута»."

    input_messages = history + [
        {"role": "user", "content": user_text}
    ]

    kwargs = {
        "model": OPENAI_MODEL,
        "instructions": SYSTEM_PROMPT,
        "input": input_messages,
    }

    if VECTOR_STORE_ID:
        kwargs["tools"] = [{
            "type": "file_search",
            "vector_store_ids": [VECTOR_STORE_ID]
        }]

    response = await client.responses.create(**kwargs)
    answer = response.output_text or "Не удалось получить ответ."

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})
    CHAT_HISTORY[chat_id] = history[-20:]

    return answer

async def process_update(update: dict):
    message = update.get("message") or update.get("edited_message")
    if not message:
        return

    chat = message.get("chat", {})
    chat_id = chat.get("id")
    text = message.get("text", "")

    if not chat_id:
        return

    if not text:
        await send_message(chat_id, "Пока я работаю только с текстовыми сообщениями.")
        return

    if text.startswith("/start"):
        await send_message(
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

    await send_typing(chat_id)

    try:
        answer = await ask_openai(chat_id, text)
        await send_message(chat_id, html.escape(answer))
    except Exception as e:
        await send_message(chat_id, f"Ошибка бота: {html.escape(str(e))}")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/telegram/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request, background_tasks: BackgroundTasks):
    if secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    update = await request.json()
    background_tasks.add_task(process_update, update)
    return {"ok": True}
