from src.tokenizer.character import CharacterTokenizer
from src.tokenizer.subword import SubwordTokenizer

def get_tokenizer(tokenizer_type="char"):
    if tokenizer_type == "char":
        return CharacterTokenizer()
    elif tokenizer_type == "subword":
        return SubwordTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Expected 'char' or 'subword'.")
