from tokenizers import Tokenizer, AddedToken

def create_added_token(token: str) -> AddedToken:
    return AddedToken(
        token,
        normalized=False,   # Prevents lowercasing/spacing changes
        special=True,       # Keeps it atomic
        single_word=False   # Ensures it's treated as an un-splittable block
    )

def promote_tokens_from_json(json_path: str, n: int):
    """
    Load a tokenizer from JSON, take all tokens with id > n,
    wrap them as AddedToken, and save back under the same name.
    """
    # Load tokenizer
    tokenizer = Tokenizer.from_file(json_path)

    # Get vocab {token: id}
    vocab = tokenizer.get_vocab()

    # Collect tokens with id > n
    extra_tokens = [tok for tok, idx in vocab.items() if idx > n]

    # Wrap them as AddedToken objects
    added_tokens = [create_added_token(tok) for tok in extra_tokens]

    # Add them to tokenizer
    tokenizer.add_tokens(added_tokens)

    # Save back to same JSON file
    tokenizer.save(json_path)

    return added_tokens


# Example usage:
promoted = promote_tokens_from_json("MiRe_bpe_c.json", 5)
print(f"Promoted {len(promoted)} tokens as added tokens")
