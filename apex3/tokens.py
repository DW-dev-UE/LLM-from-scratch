from __future__ import annotations

NAMED = ["<|endoftext|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|think|>", "<|/think|>", "<|tool_call|>",
         "<|/tool_call|>", "<|tool_result|>", "<|/tool_result|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|repo|>", "<|file|>"]
NUM_SPECIAL = 1024
SPECIAL_TOKENS = NAMED + [f"<|reserved_{i}|>" for i in range(NUM_SPECIAL - len(NAMED))]
EOS = "<|endoftext|>"


def load_tokenizer(path: str, raw_text: bool = True):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(path)
    tok.encode_special_tokens = raw_text
    return tok


def special_ids(tok) -> dict[str, int]:
    ids = {name: tok.token_to_id(name) for name in NAMED}
    missing = [n for n, i in ids.items() if i is None]
    if missing:
        raise ValueError(f"tokenizer lacks special tokens: {missing}")
    return ids
