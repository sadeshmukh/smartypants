"""VLM client: sends questions to llama-server using a tool-call loop.

The model is given a look_at_camera tool and forced to call it before answering.
The server resolves the tool call by returning the latest cached frame, so the
browser never needs to send image data.
"""
import httpx

LLAMA_URL = "http://localhost:8080/v1/chat/completions"

DEFAULT_PROMPT = "Describe this scene in one short sentence for a visually impaired listener."

_LOOK_TOOL = {
    "type": "function",
    "function": {
        "name": "look_at_camera",
        "description": (
            "Capture the current camera view. Always call this before answering "
            "any question about what the camera sees."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def build_qa_prompt(question: str) -> str:
    return (
        "You are a visual assistant for a blind or low-vision user. "
        "Use the look_at_camera tool to see the scene, then answer this question "
        f"in one or two short spoken sentences: {question}"
    )


async def ask_with_camera_tool(question: str, get_frame_b64, max_tokens: int = 128) -> str:
    prompt = build_qa_prompt(question)
    messages = [{"role": "user", "content": prompt}]

    async with httpx.AsyncClient(timeout=60) as client:
        # Turn 1: force the model to call look_at_camera
        r1 = await client.post(LLAMA_URL, json={
            "model": "local",
            "messages": messages,
            "tools": [_LOOK_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "look_at_camera"}},
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        r1.raise_for_status()

        assistant_msg = r1.json()["choices"][0]["message"]
        tool_calls = assistant_msg.get("tool_calls") or []

        frame_b64 = get_frame_b64()

        if not tool_calls or not frame_b64:
            # Fallback: model didn't call the tool or no frame cached — inject image directly
            if frame_b64:
                r_fb = await client.post(LLAMA_URL, json={
                    "model": "local",
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                    ]}],
                    "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                })
                r_fb.raise_for_status()
                return r_fb.json()["choices"][0]["message"]["content"].strip()
            return assistant_msg.get("content") or "No camera frame available yet."

        # Turn 2: resolve the tool call with the captured frame
        tool_call = tool_calls[0]
        messages.append({"role": "assistant", "tool_calls": tool_calls})
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": [{"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{frame_b64}"
            }}],
        })

        r2 = await client.post(LLAMA_URL, json={
            "model": "local",
            "messages": messages,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        r2.raise_for_status()
        return r2.json()["choices"][0]["message"]["content"].strip()


async def ask_with_nvidia_nim_vlm(question: str, get_frame_b64, max_tokens: int = 128) -> str:
    import os
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        return "Error: NVIDIA_API_KEY environment variable is not set."

    frame_b64 = get_frame_b64()
    if not frame_b64:
        return "No camera frame available yet."

    prompt = build_qa_prompt(question)

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta/llama-3.2-11b-vision-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}}
                ]
            }
        ],
        "max_tokens": max_tokens
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        res_json = response.json()
        return res_json["choices"][0]["message"]["content"].strip()

