import requests
import os
from dotenv import load_dotenv

load_dotenv()

DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"

if not DEEPL_API_KEY:
    raise ValueError("❌ DEEPL_API_KEY is not set")


def translate(text: str, target_lang: str = "EN") -> str:
    """
    Переклад одного текстового рядка на вказану мову. Визначення мови автоматичне.
    """
    params = {
        "auth_key": DEEPL_API_KEY,
        "text": text,
        "target_lang": target_lang
    }

    response = requests.post(DEEPL_API_URL, data=params)

    if response.status_code != 200:
        raise RuntimeError(f"DeepL API error: {response.status_code} - {response.text}")

    result = response.json()
    return result["translations"][0]["text"]


def translate_to_english(text: str) -> str:
    return translate(text, target_lang="EN")


def translate_from_english(text: str, target_lang: str = "UK") -> str:
    return translate(text, target_lang=target_lang)


def translate_messages(messages: list[dict], target_lang: str = "EN") -> list[dict]:
    """
    Переводит список сообщений формата {"role": ..., "content": ...}
    Возвращает новый список с переведённым контентом.
    """
    translated = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        translated_content = translate(content, target_lang=target_lang)
        translated.append({"role": role, "content": translated_content})
    return translated
