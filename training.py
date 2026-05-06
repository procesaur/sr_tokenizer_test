from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, Regex
from collections import Counter
from tokenization_srna import SrnaTokenizer
from tqdm import tqdm


vocab_size = 30000
suffix_vocab_size = 200
latin = True
base_bpe= True
srna = True
original_bpe = False
tanja_bpe = False


if latin:
    dataset = load_from_disk("./serbian_tokenizer_dataset", keep_in_memory=True)
else:
    dataset = load_dataset(
        "procesaur/sr-tokenizer-test",
        split="train",
        keep_in_memory=True
    )


#dataset = dataset.select(range(3000))
initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

def batch_iterator(dataset, batch_size=5000):
    batch = []
    for example in dataset:
        batch.append(example["text"])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

if base_bpe:
    # normal BPE baseline
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer)
    tokenizer.save("tokenizers/bpe.json")


if srna:
    srnatok = SrnaTokenizer()

    print(srnatok.prepare_for_tokenization("This is a Test"))

    def srna_batch_iterator(dataset, batch_size=10000):
        batch = []
        for example in dataset:
            batch.append(srnatok.prepare_for_tokenization(example["text"]))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "<|up|>", "<|cap|>", "<|cyr_end|>", "<|cyr_start|>"]
    )
    tokenizer.train_from_iterator(srna_batch_iterator(dataset), trainer=trainer)
    tokenizer.save("tokenizers/srna.json")


if original_bpe:
    # original BPE
    tokenizer = Tokenizer(models.BPE(ignore_merges=True))
    tokenizer.normalizer = normalizers.Replace(Regex(r"[^\w]+"), "_ ")
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    ])

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        initial_alphabet=initial_alphabet,
    )
    tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer)
    tokenizer.save("tokenizers/original_bpe.json")


    ini_vocab_size = len(initial_alphabet) + len(special_tokens)

    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])  # sort by ID
    ini_vocab = dict(sorted_vocab[:ini_vocab_size])
    rest = dict(sorted_vocab[ini_vocab_size:])

    filtered_vocab = [tok for tok, idx in rest.items() if tok.endswith("_") and tok[:-1] not in ini_vocab.keys()][:suffix_vocab_size]
    filtered_vocab = {tok:i+len(ini_vocab) for i, tok in enumerate(filtered_vocab)}
    tokenizer = Tokenizer(models.BPE(vocab={**ini_vocab, **filtered_vocab}, merges=[]))
    tokenizer.normalizer = normalizers.Replace(Regex(r"[^\w]+"), "_ ")
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    ])


    if tanja_bpe:
        counter = Counter()
        for example in tqdm(dataset, total=len(dataset)):
            encoded = tokenizer.encode(example["text"])
            tokens = encoded.tokens
            
            # Look for sequences of single characters (unmerged)
            seq = []
            for tok in tokens:
                if len(tok) == 1 and tok != "_":  # raw char
                    seq.append(tok)
                else:
                    if tok == "_":
                        seq.append(tok)
                    if len(seq) > 1 :
                        counter["".join(seq)] += 1
                    seq = []
            # handle trailing sequence
            if len(seq) > 1:
                counter["".join(seq)] += 1

        scores = {seq: freq * len(seq) for seq, freq in counter.items()}
        top_sequences = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:vocab_size-ini_vocab_size]
        new_tokens = [seq for seq, _ in top_sequences]

        tokenizer.add_tokens(new_tokens)
        tokenizer.save("tokenizers/tanja_bpe.json")




dataset = load_dataset(
    "procesaur/sr-tokenizer-test",
    split="train",
    keep_in_memory=True
)


#dataset = dataset.select(range(3000))
initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

def batch_iterator(dataset, batch_size=5000):
    batch = []
    for example in dataset:
        batch.append(example["text"])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

if base_bpe:
    # normal BPE baseline
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer)
    tokenizer.save("tokenizers/bpe_c.json")


if srna:
    srnatok = SrnaTokenizer()

    print(srnatok.prepare_for_tokenization("This is a Test"))

    def srna_batch_iterator(dataset, batch_size=10000):
        batch = []
        for example in dataset:
            batch.append(srnatok.prepare_for_tokenization(example["text"]))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "<|up|>", "<|cap|>", "<|cyr_end|>", "<|cyr_start|>"]
    )
    tokenizer.train_from_iterator(srna_batch_iterator(dataset), trainer=trainer)
    tokenizer.save("tokenizers/srna_c.json")


if original_bpe:
    # original BPE
    tokenizer = Tokenizer(models.BPE(ignore_merges=True))
    tokenizer.normalizer = normalizers.Replace(Regex(r"[^\w]+"), "_ ")
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    ])

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        initial_alphabet=initial_alphabet,
    )
    tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer)
    tokenizer.save("tokenizers/original_bpe_c.json")


    ini_vocab_size = len(initial_alphabet) + len(special_tokens)

    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])  # sort by ID
    ini_vocab = dict(sorted_vocab[:ini_vocab_size])
    rest = dict(sorted_vocab[ini_vocab_size:])

    filtered_vocab = [tok for tok, idx in rest.items() if tok.endswith("_") and tok[:-1] not in ini_vocab.keys()][:suffix_vocab_size]
    filtered_vocab = {tok:i+len(ini_vocab) for i, tok in enumerate(filtered_vocab)}
    tokenizer = Tokenizer(models.BPE(vocab={**ini_vocab, **filtered_vocab}, merges=[]))
    tokenizer.normalizer = normalizers.Replace(Regex(r"[^\w]+"), "_ ")
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    ])


    if tanja_bpe:
        counter = Counter()
        for example in tqdm(dataset, total=len(dataset)):
            encoded = tokenizer.encode(example["text"])
            tokens = encoded.tokens
            
            # Look for sequences of single characters (unmerged)
            seq = []
            for tok in tokens:
                if len(tok) == 1 and tok != "_":  # raw char
                    seq.append(tok)
                else:
                    if tok == "_":
                        seq.append(tok)
                    if len(seq) > 1 :
                        counter["".join(seq)] += 1
                    seq = []
            # handle trailing sequence
            if len(seq) > 1:
                counter["".join(seq)] += 1

        scores = {seq: freq * len(seq) for seq, freq in counter.items()}
        top_sequences = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:vocab_size-ini_vocab_size]
        new_tokens = [seq for seq, _ in top_sequences]

        tokenizer.add_tokens(new_tokens)
        tokenizer.save("tokenizers/tanja_bpe_c.json")


