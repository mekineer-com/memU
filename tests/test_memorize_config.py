import pytest

from memu.app.settings import MemorizeConfig


def test_memorize_config_rejects_removed_conversation_preprocess_prompt() -> None:
    with pytest.raises(ValueError, match="Conversation preprocess prompt override"):
        MemorizeConfig(multimodal_preprocess_prompts={"conversation": "split it"})


def test_category_summary_defaults_to_300_words() -> None:
    assert MemorizeConfig().category_summary_target_words == 300


def test_episodes_per_segment_must_be_positive() -> None:
    with pytest.raises(ValueError):
        MemorizeConfig(episodes_per_segment=0)


def test_episodes_per_segment_accepts_configured_limit() -> None:
    assert MemorizeConfig(episodes_per_segment=4).episodes_per_segment == 4
