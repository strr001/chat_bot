import os
import requests
import time
from dotenv import load_dotenv
import re

# 🔄 Завантаження конфігурації
load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "your-endpoint-id")
RUNPOD_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

if not RUNPOD_API_KEY:
    raise ValueError("❌ RUNPOD_API_KEY is not set")

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

# 📌 Інструкції для моделі
SYSTEM_PROMPT = (
    "You are an expert chatbot embedded on a website. "
    "DO NOT ANSWER INQUIRIES ABOUT OTHER PROFESSIONS EXCEPT IT!!! "
    "DON'T REPLY WITH DIALOGUE AND DONT USE ASSISTANT AND USER WORDS IN YOUR RESPONSE."
    "Your specialty is helping users create resumes specifically for IT professions. "
    "You provide guidance, examples, and suggestions tailored to the user's needs.\n"
    "\nResponsibilities:\n"
    "- Giving resume writing advice for IT roles (e.g., Frontend Developer, QA Engineer, Data Scientist).\n"
    "- Providing examples of resume sections: About Me, Work Experience, Education, Skills, Projects, Certifications.\n"
    "- Generating full resumes based on user data.\n"
    "- Describing requirements for Junior, Middle, and Senior levels.\n"
    "- Adapting suggestions to user's level and market standards.\n"
    "\nResponse rules:\n"
    "- ALWAYS respond in English.\n"
    "- Respond clearly, professionally, and in structured format.\n"
    "- Ask for clarification if the request is too general.\n"
    "- Use user input to tailor suggestions.\n"
    "- Provide examples if asked.\n"
)

def build_prompt_from_history(messages: list[dict]) -> str:
    prompt = SYSTEM_PROMPT + "\n\n"
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        if role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
    prompt += "Assistant:"
    return prompt

def extract_text_from_string(text: str) -> str:
    """
    Витягує вміст з 'tokens': ['...'] з JSON-подібного рядка.
    Якщо не знайдено — повертає оригінальний текст.
    """
    # Знаходимо перший токенізований фрагмент
    match = re.search(r"'tokens'\s*:\s*\[\s*'([^']+)'\s*\]", text)
    if match:
        extracted = match.group(1)
        return extracted.strip()
    return text.strip()


def clean_response(response: str) -> str:
    """
    Обрізає все після 'Assistant:' і додає 'Приклад вашої відповіді:' перед останнім реченням користувача.
    """
    assistant_index = response.find("Assistant:")
    if assistant_index != -1:
        before_assistant = response[:assistant_index].strip()

        # Шукаємо останню репліку користувача перед Assistant:
        last_user_index = before_assistant.rfind("User:")
        if last_user_index != -1:
            user_text = before_assistant[last_user_index + len("User:"):].strip()
            cleaned = before_assistant[:last_user_index].strip()
            return f"{cleaned} Приклад вашої відповіді: {user_text}"

        # Якщо немає 'User:', просто обрізаємо
        return before_assistant.strip()

    return response.strip()

def generate_llama_response(messages: list[dict], max_tokens: int = 500, temperature: float = 0.3) -> str:
    full_prompt = build_prompt_from_history(messages)
    print(f"🔎 Sending prompt to Runpod /run:\n{full_prompt[:300]}...")

    payload = {
        "input": {
            "prompt": full_prompt,
            "sampling_params": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        }
    }

    response = requests.post(f"{RUNPOD_BASE_URL}/run", headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Runpod API error: {response.status_code} - {response.text}")

    job_id = response.json().get("id")
    if not job_id:
        raise RuntimeError(f"Runpod did not return a job ID: {response.json()}")

    print(f"✅ Job submitted. ID: {job_id}")

    for _ in range(60):
        status_resp = requests.get(f"{RUNPOD_BASE_URL}/status/{job_id}", headers=HEADERS)
        if status_resp.status_code != 200:
            raise RuntimeError(f"Status check failed: {status_resp.status_code} - {status_resp.text}")

        result = status_resp.json()
        status = result.get("status")

        if status == "COMPLETED":
            output = result.get("output", "")
            output_text = ""

            if isinstance(output, dict):
                output_text = output.get("text") or output.get("output", "")
            elif isinstance(output, list) and output:
                output_text = str(output[0])
            elif isinstance(output, str):
                output_text = output

            if not output_text:
                raise RuntimeError(f"Job completed but no usable text in output: {output}")

            cleaned = clean_response(output_text)
            return cleaned

        elif status in ["FAILED", "CANCELLED"]:
            raise RuntimeError(f"Job failed or cancelled: {result}")

        print(f"⏳ Waiting... Status: {status}")
        time.sleep(2)

    raise RuntimeError("Runpod job timeout after 2 minutes.")
