from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal

from utils import find_best_resume_example
from translation_utils import translate_messages, translate_from_english
from llama_client import generate_llama_response, extract_text_from_string
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

app.mount("/examples_cv", StaticFiles(directory="examples_cv"), name="examples_cv")

# Моделі
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

    # Переклад повідомлень
    try:
        messages_en = translate_messages([msg.dict() for msg in request.messages], target_lang="EN")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation to English failed: {str(e)}")

    # Якщо запитано приклад резюме — тільки приклад + повідомлення
    if request.include_example:
        user_message_en = messages_en[-1]["content"]
        example = find_best_resume_example(user_message_en)

        response = {
            "response": "🔍 По вашому запиту я знайшов приклад резюме. Ознайомтесь із ним нижче."
        }

        if example:
            response["example_filename"] = example["filename"]
            response["example_title"] = example["title"]
            response["example_url"] = f"/examples_cv/{example['filename']}"
        else:
            response["response"] =  ("🔍 По вашому запиту я не знайшов відповідних резюме. Будь ласка, "
                                     "вкажіть точніше ту спеціальність або технологію, згідно яких потрібно "
                                     "надати реальний приклад резюме. Також спробуйте задати запит мені напряму "
                                     "(бажано поетапно), я постараюся згенерувати вам гарне резюме 🤗.")

        return response

    # Генерація відповіді
    try:
        assistant_reply_en = generate_llama_response(messages_en)
        assistant_reply_en = extract_text_from_string(assistant_reply_en)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLaMA response failed: {str(e)}")

    try:
        assistant_reply_ua = translate_from_english(assistant_reply_en, target_lang="UK")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation to Ukrainian failed: {str(e)}")

    return {"response": assistant_reply_ua}
