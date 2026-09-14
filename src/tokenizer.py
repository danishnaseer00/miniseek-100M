from transformers import GPT2Tokenizer
from typing import List


class Tokenizer:
    def __init__(self, tokenizer_name: str = "gpt2", max_length: int = 2048):
        self.name = tokenizer_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
    
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size
    
    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id
    
    def encode(self, text: str, add_special_tokens: bool = True, truncate: bool = True) -> List[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=self.max_length if truncate else None,
            truncation=truncate,
        )
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
