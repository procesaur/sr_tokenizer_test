import json
from tokenizers import Tokenizer  # HuggingFace Tokenizers
from tokenizers.decoders import ByteLevel
decoder = ByteLevel()

def decode_tokenizer_vocab(input_path, output_path):
    # Load the tokenizer from JSON
    tokenizer = Tokenizer.from_file(input_path)

    # Load raw JSON to access vocab indices
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Depending on tokenizer type, vocab may be under "model"->"vocab" or "vocab"
    if "model" in raw and "vocab" in raw["model"]:
        vocab = raw["model"]["vocab"]
    elif "vocab" in raw:
        vocab = raw["vocab"]
    else:
        raise ValueError("No vocab found in tokenizer JSON")

    decoded_vocab = []
    for token, idx in vocab.items():
        # Use tokenizer.decode on the ID to get the proper string
        decoded_str = decoder.decode([token]) 
        decoded_vocab.append((idx, decoded_str))

    #decoded_vocab.sort(key=lambda x: x[0])
    sorted_vocab = {tok: idx for idx, tok in decoded_vocab}

    # Save back with non-ASCII preserved
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_vocab, f, ensure_ascii=False, indent=2)

    print(f"Decoded vocab saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    decode_tokenizer_vocab("original_bpe.json", "decoded_original_bpe.json")
