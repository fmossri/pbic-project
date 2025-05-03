import pytest
import re
from src.utils.text_normalizer import TextNormalizer
from src.config.models import TextNormalizerConfig # Import config model

# Config fixtures
@pytest.fixture
def config_all_on():
    """Config with all steps enabled."""
    return TextNormalizerConfig(
        use_unicode_normalization=True,
        use_lowercase=True,
        # use_remove_special_chars=True, # Not implemented in current normalizer
        use_remove_extra_whitespace=True
    )

@pytest.fixture
def config_all_off():
    """Config with all steps disabled."""
    return TextNormalizerConfig(
        use_unicode_normalization=False,
        use_lowercase=False,
        # use_remove_special_chars=False,
        use_remove_extra_whitespace=False
    )

@pytest.fixture
def config_lowercase_only():
    """Config with only lowercase enabled."""
    return TextNormalizerConfig(
        use_unicode_normalization=False,
        use_lowercase=True,
        # use_remove_special_chars=False,
        use_remove_extra_whitespace=False
    )

# Default normalizer fixture (uses config_all_on)
@pytest.fixture
def normalizer(config_all_on):
    """Fixture for TextNormalizer instance with default (all on) config."""
    return TextNormalizer(config=config_all_on, log_domain="test_domain")

# --- Tests for individual steps (still useful) ---
# These tests now implicitly use the default 'all_on' config,
# but they target the private methods directly, bypassing the config checks.
# If private methods change, these might need updates.

def test_normalize_unicode(normalizer):
    """Testa normalização básica de Unicode via _normalize_unicode."""
    # normalizer = TextNormalizer(log_domain="test_domain") # No longer needed
    test_cases = [
        # Pontuação de largura total
        ("Hello，World！", "Hello,World!"),
        # Aspas inteligentes
        ("\"Hello\"", "\"Hello\""),
        # Apóstrofos Unicode
        ("don't", "don't"),
        # Caracteres Unicode misturados
        ("Hello，World！", "Hello,World!"),
        # String vazia
        ("", ""),
        # Apenas pontuação
        ("，。！？", ",。!?"),
    ]
    for input_text, expected in test_cases:
        result = normalizer._normalize_unicode(input_text)
        assert result == expected, f"Falhou para entrada: '{input_text}'\nEsperado: '{expected}'\nObtido: '{result}'"

def test_normalize_whitespace(normalizer):
    """Testa normalização de espaços em branco via _normalize_whitespace."""
    # normalizer = TextNormalizer(log_domain="test_domain") # No longer needed
    test_cases = [
        # Normalização básica de espaços
        ("Hello   World", "Hello World"),
        ("Hello\tWorld", "Hello World"),
        ("Hello\nWorld", "Hello World"),
        ("Hello\r\nWorld", "Hello World"),
        # Espaços no início e fim
        ("  Hello World  ", "Hello World"),
        # Tipos mistos de espaços
        ("Hello \t \n \r\n World", "Hello World"),
        # Preserva pontuação existente sem adicionar espaços
        ("Hello,World", "Hello,World"),
        ("Hello, World", "Hello, World"),
        # String vazia
        ("", ""),
        # Apenas espaços em branco
        ("   \t\n\r", ""),
        # Caso complexo com espaços existentes
        ("  Hello,  World!  ", "Hello, World!"),
    ]
    for input_text, expected in test_cases:
        result = normalizer._normalize_whitespace(input_text)
        assert result == expected, f"Falhou para entrada: '{input_text}'\nEsperado: '{expected}'\nObtido: '{result}'"

def test_normalize_case(normalizer):
    """Testa normalização de maiúsculas/minúsculas via _normalize_case."""
    # normalizer = TextNormalizer(log_domain="test_domain") # No longer needed
    test_cases = [
        # Conversão básica para minúsculas
        ("HELLO WORLD", "hello world"),
        ("Hello World", "hello world"),
        # Maiúsculas e minúsculas misturadas
        ("HeLLo WoRLD", "hello world"),
        # Números e pontuação (devem ser preservados)
        ("Hello123!@#", "hello123!@#"),
        # String vazia
        ("", ""),
        # Já está em minúsculas
        ("hello world", "hello world"),
        # Caracteres especiais
        ("CAFÉ", "café"),
        # Umlauts alemães
        ("ÜBER", "über"),
        # Scripts mistos
        ("Hello МИР", "hello мир"),
    ]
    for input_text, expected in test_cases:
        result = normalizer._normalize_case(input_text)
        assert result == expected, f"Falhou para entrada: '{input_text}'\nEsperado: '{expected}'\nObtido: '{result}'"

# --- Test the main normalize method with configuration ---

def test_normalize_pipeline_all_on(normalizer):
    """Testa o pipeline completo de normalização com config all_on."""
    # normalizer uses the fixture with config_all_on
    test_cases = [
        # Combines unicode, whitespace, case
        ("  HELLO   WORLD  ", "hello world"),
        
        # Unicode + espaços + maiúsculas/minúsculas
        ("  Hello，  World！  ", "hello, world!"),
        
        # Maiúsculas/minúsculas misturadas com caracteres especiais
        ("  CAFÉ  123  ", "café 123"),
        
        # Caso complexo com todas as características
        ("  HeLLo，  WoRLD！  How are you?  ", "hello, world! how are you?"),
        
        # String vazia
        ("", ""),
        
        # Apenas espaços em branco
        ("   \t\n\r", ""),
        
        # Scripts mistos
        ("  Hello   МИР！  ", "hello мир!"),
        
        # Texto alemão com umlauts
        ("  ÜBER   ALLES  ", "über alles"),
        
        # Aspas inteligentes e apóstrofos
        ("  \"Hello\"   don't  ", "\"hello\" don't"),
    ]
    for input_text, expected in test_cases:
        # normalize now returns a list
        result = normalizer.normalize(input_text)[0]
        assert result == expected, f"Failed for input: '{input_text}'\nExpected: '{expected}'\nGot: '{result}'"

def test_normalize_list_input(normalizer):
    """Test normalize with a list input."""
    texts = ["  HELLO   WORLD  ", "  Test TWO  "]
    expected = ["hello world", "test two"]
    assert normalizer.normalize(texts) == expected

def test_normalize_mixed_list_input(normalizer):
    """Test normalize raises TypeError for list containing non-strings."""
    texts = ["  String ONE  ", 123, "String TWO", None]
    # Expect TypeError because the function now raises an error on non-string list items
    with pytest.raises(TypeError, match="O texto de entrada deve ser uma string ou uma lista de strings"):
        normalizer.normalize(texts)

def test_normalize_type_error(normalizer):
    """Test normalize with invalid input type."""
    with pytest.raises(TypeError):
        normalizer.normalize(123) # No longer handles single int
    with pytest.raises(TypeError):
        normalizer.normalize(None)
    with pytest.raises(TypeError):
        normalizer.normalize({"a": 1})

# --- Tests for specific configurations ---

def test_normalize_all_off(config_all_off):
    """Test normalization pipeline with all steps off."""
    normalizer_off = TextNormalizer(config=config_all_off, log_domain="test_off")
    text = "  HeLLo，  WoRLD！ \t \n "
    expected = "  HeLLo，  WoRLD！ \t \n " # Should be unchanged
    assert normalizer_off.normalize(text)[0] == expected

def test_normalize_lowercase_only_config(config_lowercase_only):
    """Test normalization pipeline with only lowercase enabled."""
    normalizer_lower = TextNormalizer(config=config_lowercase_only, log_domain="test_lower")
    text = "  HeLLo，  WoRLD！ \t \n "
    expected = "  hello，  world！ \t \n " # Only case changed
    assert normalizer_lower.normalize(text)[0] == expected

def test_normalize_whitespace_only_config(config_all_off):
    """Test normalization pipeline with only whitespace enabled."""
    config_ws_only = config_all_off.model_copy(update={"use_remove_extra_whitespace": True})
    normalizer_ws = TextNormalizer(config=config_ws_only, log_domain="test_ws")
    text = "  HeLLo，  WoRLD！ \t \n "
    expected = "HeLLo， WoRLD！"
    assert normalizer_ws.normalize(text)[0] == expected

def test_normalize_unicode_only_config(config_all_off):
    """Test normalization pipeline with only unicode enabled."""
    config_unicode_only = config_all_off.model_copy(update={"use_unicode_normalization": True})
    normalizer_unicode = TextNormalizer(config=config_unicode_only, log_domain="test_unicode")
    text = "  HeLLo，  WoRLD！ \t \n "
    expected = "  HeLLo,  WoRLD! \t \n " # Only unicode chars changed
    assert normalizer_unicode.normalize(text)[0] == expected

# --- update_config Tests ---

def test_update_config_no_change(normalizer, config_all_on):
    """Test update_config when the new config is identical."""
    initial_config_ref = normalizer.config # Store initial ref
    new_config = config_all_on.model_copy() # Identical copy

    normalizer.update_config(new_config)

    # Assert config reference DID NOT change because values were identical
    assert normalizer.config is initial_config_ref 
    # Optional: Also check values are still equal (though covered by 'is')
    assert normalizer.config == new_config

def test_update_config_flags_change(normalizer, config_all_off):
    """Test update_config changes behavior of normalize."""
    # Initial state (all on)
    assert normalizer.normalize("  HELLO   WORLD!  ")[0] == "hello world!"

    # Update config to all off
    normalizer.update_config(config_all_off) 

    # Check config object was updated
    assert normalizer.config is config_all_off
    assert normalizer.config.use_lowercase is False
    assert normalizer.config.use_remove_extra_whitespace is False
    assert normalizer.config.use_unicode_normalization is False

    # Verify normalize behavior changed - text should be unchanged
    assert normalizer.normalize("  HELLO   WORLD!  ")[0] == "  HELLO   WORLD!  "

    # Update config back to all on (using a copy)
    config_on_again = config_all_off.model_copy(update={
        "use_unicode_normalization": True,
        "use_lowercase": True,
        "use_remove_extra_whitespace": True
    })
    normalizer.update_config(config_on_again)

    # Verify behavior reverts
    assert normalizer.normalize("  HELLO   WORLD!  ")[0] == "hello world!"
    assert normalizer.config is config_on_again

