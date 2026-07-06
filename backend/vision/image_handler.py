"""
backend/vision/image_handler.py
---------------------------------
Uses Mistral's vision-capable model 'pixtral-large-latest' to understand
images (describe / analyze / answer questions about them).

Usage:
    from backend.vision.image_handler import analyze_image

    answer = analyze_image(
        image_path="backend/vision/sample.png",
        question="What is in this image?"
    )
    print(answer)
"""

import os
import base64
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

load_dotenv()  # Loads MISTRAL_API_KEY from .env

# -----------------------------
# VISION MODEL SETUP
# -----------------------------
vision_llm = ChatMistralAI(
    model="pixtral-large-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.2,
)


def _encode_image_to_base64(image_path: str) -> str:
    """Converts an image file into a base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_mime_type(image_path: str) -> str:
    """Returns the mime type based on the file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    return mime_map.get(ext, "image/png")


def analyze_image(image_path: str, question: str = "Describe this image in detail.") -> str:
    """
    Takes an image file and returns the vision model's answer to the given question.

    Args:
        image_path: Path to the local image file
        question: What to ask about the image

    Returns:
        The model's text response (string)
    """
    if not os.path.exists(image_path):
        return f"⚠️ Image file not found: {image_path}"

    base64_image = _encode_image_to_base64(image_path)
    mime_type = _get_mime_type(image_path)

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {
                "type": "image_url",
                "image_url": f"data:{mime_type};base64,{base64_image}",
            },
        ]
    )

    response = vision_llm.invoke([message])
    return response.content


def analyze_image_bytes(image_bytes: bytes, mime_type: str, question: str = "Describe this image in detail.") -> str:
    """
    Used when a UI (like Streamlit) provides raw image bytes directly,
    without saving the file to disk first.

    Args:
        image_bytes: Raw image bytes (e.g. from st.file_uploader)
        mime_type: e.g. "image/png", "image/jpeg"
        question: What to ask about the image

    Returns:
        The model's text response (string)
    """
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {
                "type": "image_url",
                "image_url": f"data:{mime_type};base64,{base64_image}",
            },
        ]
    )

    response = vision_llm.invoke([message])
    return response.content


# -----------------------------
# QUICK TEST (for running this file directly)
# -----------------------------
if __name__ == "__main__":
    test_image_path = "backend/vision/sample.png"  # put the path to your test image
    result = analyze_image(test_image_path, "What is shown in this image?")
    print(result)