from memu.prompts.preprocess import audio, document, image, video

PROMPTS: dict[str, str] = {
    "video": video.PROMPT.strip(),
    "image": image.PROMPT.strip(),
    "document": document.PROMPT.strip(),
    "audio": audio.PROMPT.strip(),
}

__all__ = ["PROMPTS"]
