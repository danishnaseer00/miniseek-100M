"""Process-pool tokenization worker for the corpus builder.

encode_range lives in this module (importable by name) so multiprocessing.Pool
can pickle it across worker processes. It must NOT live in a hyphenated-path
module (data-layer/), which is not importable by name.
"""


def encode_range(args):
    """Process-pool worker: read one byte-range chunk and tokenize each doc line."""
    import json

    import numpy as np
    from src.tokenizer import Tokenizer

    path, start_byte, end_byte, tokenizer_name, max_length = args
    tokenizer = Tokenizer(tokenizer_name, max_length=max_length)
    pieces = []
    lens = []
    with open(path, "rb") as f:
        f.seek(start_byte)
        data = f.read(end_byte - start_byte)
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            text = json.loads(line.decode("utf-8", errors="replace")).get("text")
        except Exception:
            continue
        if not text or not text.strip():
            continue
        toks = tokenizer.encode(text, add_special_tokens=True, truncate=False)
        if len(toks) <= 1:
            continue
        pieces.append(np.asarray(toks, dtype=np.int32))
        lens.append(len(toks))
    return (pieces, lens)