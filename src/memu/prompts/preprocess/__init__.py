from memu.prompts.preprocess import audio, conversation, document, image, video

PROMPTS: dict[str, str] = {
    "conversation": conversation.PROMPT.strip(),
    "cross_conversation": conversation.CROSS_CONVERSATION_PROMPT.strip(),
    "video": video.PROMPT.strip(),
    "image": image.PROMPT.strip(),
    "document": document.PROMPT.strip(),
    "audio": audio.PROMPT.strip(),
}

__all__ = ["PROMPTS"]
