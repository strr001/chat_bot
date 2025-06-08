from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Literal

from utils import find_best_resume_example
from translation_utils import translate_messages, translate_from_english
from llama_client import generate_llama_response
import os

app = FastAPI()

# 🔓 Дозволяємо доступ з будь-якого джерела (для frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📂 Папка з прикладами (якщо зберігаються локально)
app.mount("/examples_cv", StaticFiles(directory="examples_cv"), name="examples_cv")

# 📥 Моделі запиту
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    include_example: bool = False


@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Empty message history")

    # 1. Переклад історії на англійську
    try:
        messages_en = translate_messages([msg.dict() for msg in request.messages], target_lang="EN")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation to English failed: {str(e)}")

    # 2. Якщо потрібно знайти приклад резюме — не генеруємо LLaMA
    if request.include_example:
        user_message_en = messages_en[-1]["content"]
        example = find_best_resume_example(user_message_en)

        if example:
            return {
                "message": "🔍 По вашому запиту я знайшов приклад резюме. Ознайомтесь із ним нижче.",
                "example_filename": example["filename"],
                "example_title": example["title"],
                "example_url": f"https://devfusionbucket.s3.eu-west-3.amazonaws.com/chat-bot-resume-folder/{example['filename']}"
            }
        else:
            return {
                "message": (
                    "🔍 По вашому запиту я не знайшов відповідних резюме. "
                    "Будь ласка, уточніть спеціалізацію, або сформулюйте запит трохи інакше. "
                    "А ще я можу згенерувати резюме, якщо ви залишите свої дані 👇"
                )
            }

    # 3. Генерація відповіді LLaMA
    try:
        assistant_reply_en = generate_llama_response(messages_en)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLaMA response failed: {str(e)}")

    # 4. Переклад відповіді на українську
    try:
        assistant_reply_ua = translate_from_english(assistant_reply_en, target_lang="UK")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation to Ukrainian failed: {str(e)}")

    return {
        "message": assistant_reply_ua
    }
