"""Sends an image+prompt to the locally-served vision-language model (llama-server).

Payload shape confirmed against /Developer/edgeAI/jetson/jetson-llm/vision_test.py.
"""
import httpx

LLAMA_URL = "http://localhost:8080/v1/chat/completions"

DEFAULT_PROMPT = "Describe this scene in one short sentence for a visually impaired listener."


def build_qa_prompt(question: str) -> str:
    return (
        "You are a visual assistant for a blind or low-vision user. Answer this "
        f"question about what the camera currently sees, in one or two short spoken "
        f"sentences: {question}"
    )


async def describe_image(image_data_url: str, prompt: str = DEFAULT_PROMPT, max_tokens: int = 128) -> str:
    payload = {
        "model": "local",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(LLAMA_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"].strip()
