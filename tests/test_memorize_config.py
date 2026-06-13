import pytest

from memu.app.settings import MemorizeConfig


def test_memorize_config_rejects_removed_conversation_preprocess_prompt() -> None:
    with pytest.raises(ValueError, match="Conversation preprocess prompt override"):
        MemorizeConfig(multimodal_preprocess_prompts={"conversation": "split it"})
