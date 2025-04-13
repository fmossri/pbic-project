import pytest
import re
from src.utils.text_normalizer import TextNormalizer

@pytest.fixture
def normalizer():
    return TextNormalizer(log_domain="test_domain")

def test_normalize_unicode():
    """Testa normalização básica de Unicode."""
    normalizer = TextNormalizer(log_domain="test_domain")
    
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

def test_normalize_whitespace():
    """Testa normalização de espaços em branco - múltiplos caracteres de espaço devem se tornar um único espaço."""
    normalizer = TextNormalizer(log_domain="test_domain")
    
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

def test_normalize_case():
    """Testa normalização de maiúsculas/minúsculas - verifica se o texto é convertido para minúsculas."""
    normalizer = TextNormalizer(log_domain="test_domain")
    
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

def test_normalize():
    """Testa o pipeline completo de normalização - Unicode, espaços em branco e maiúsculas/minúsculas."""
    normalizer = TextNormalizer(log_domain="test_domain")
    
    test_cases = [
        # Normalização básica
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
        result = normalizer.normalize(input_text)[0]
        assert result == expected, f"Failed for input: '{input_text}'\nExpected: '{expected}'\nGot: '{result}'"

