import pytest

from memu.app.settings import MemorizeConfig
from memu.prompts.category_summary import PROMPT, PROMPT_WITH_REFS


def test_memorize_config_rejects_removed_conversation_preprocess_prompt() -> None:
    with pytest.raises(ValueError, match="Conversation preprocess prompt override"):
        MemorizeConfig(multimodal_preprocess_prompts={"conversation": "split it"})


def test_category_summary_defaults_to_300_words() -> None:
    assert MemorizeConfig().category_summary_target_length == 300
    assert "within {target_length} words" in PROMPT
    assert "within {target_length} words" in PROMPT_WITH_REFS
