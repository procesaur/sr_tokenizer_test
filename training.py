from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, Regex
from collections import Counter
from tokenization_srna import SrnaTokenizer
from tqdm import tqdm
from json import load, dump


vocab_size = 30000
suffix_vocab_size = 200
latin = True
base_bpe= False
srna = False
original_bpe = True
MiRe_bpe = True
end_suffix = "Ġ"
initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
MiRe_cutoff = 768


def get_dataset(latin, test=False):
    if latin:
        if test:
            return load_from_disk("./serbian_tokenizer_dataset_test", keep_in_memory=True)
        return load_from_disk("./serbian_tokenizer_dataset", keep_in_memory=True)
    if test:
        return load_dataset(
        "procesaur/sr-tokenizer-test",
        split="train",
        keep_in_memory=True
        )    
    return load_dataset(
        "procesaur/sr-tokenizer-test",
        split="test",
        keep_in_memory=True
        )  


def batch_iterator(dataset, batch_size=10000, fn=None):
    batch = []
    for example in dataset:
        if fn:
            batch.append(fn(example["text"]))
        else:
            batch.append(example["text"])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def train_bpe(dataset, latin=True):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer)
    if latin:
        tokenizer.save("tokenizers/bpe.json")
    else:
        tokenizer.save("tokenizers/bpe_c.json")


def train_srna(dataset, latin=True):
    srnatok = SrnaTokenizer()
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "<|up|>", "<|cap|>", "<|cyr_end|>", "<|cyr_start|>"]
    )
    tokenizer.train_from_iterator(batch_iterator(dataset, fn=srnatok.prepare_for_tokenization), trainer=trainer)
    if latin:
        tokenizer.save("tokenizers/srna.json")
    else:
        tokenizer.save("tokenizers/srna_c.json")


def train_original_bpe(dataset, latin=True):
    tokenizer = Tokenizer(models.BPE(ignore_merges=True))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    ])

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        initial_alphabet=initial_alphabet,
        end_of_word_suffix=end_suffix
    )
    tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer)
    if latin:
        tokenizer.save("tokenizers/original_bpe.json")
    else:
        tokenizer.save("tokenizers/original_bpe_c.json")


def train_MiRe_bpe(dataset, latin=True):
    if latin:
        with open("tokenizers/original_bpe.json", "r", encoding="utf-8") as f:
            data = load(f)
    else:
        with open("tokenizers/original_bpe_c.json", "r", encoding="utf-8") as f:
            data = load(f)      

    original_vocab = data["model"]["vocab"]
    sorted_vocab = sorted(original_vocab.items(), key=lambda x: x[1])
    new_vocab = {token: idx for token, idx in sorted_vocab[:MiRe_cutoff]}
    data["model"]["vocab"] = new_vocab

    if "merges" in data["model"]:
        data["model"]["merges"] = []

    with open("temp.json", "w", encoding="utf-8") as f:
        dump(data, f, ensure_ascii=False)

    tokenizer = Tokenizer.from_file("temp.json")
    current_id = tokenizer.get_vocab_size()
    print(f"New Vocab Size: {current_id}")

    counter = Counter()
    for example in tqdm(dataset, total=len(dataset)):
        encoded = tokenizer.encode(example["text"])
        tokens = encoded.tokens
        
        # Look for sequences of single characters (unmerged)
        seq = []
        for tok in tokens:
            if len(tok) == 1 and tok != end_suffix:  # raw char
                seq.append(tok)
            else:
                if tok == end_suffix:
                    seq.append(tok)
                if len(seq) > 1 :
                    counter["".join(seq)] += 1
                seq = []
        # handle trailing sequence
        if len(seq) > 1:
            counter["".join(seq)] += 1

    scores = {seq: freq * len(seq) for seq, freq in counter.items()}
    top_sequences = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:vocab_size-MiRe_cutoff]
    new_tokens = [seq for seq, _ in top_sequences]

    for token in new_tokens:
        if token not in new_vocab:
            new_vocab[token] = current_id
            current_id += 1

    data["model"]["vocab"] = new_vocab
    with open("temp.json", "w", encoding="utf-8") as f:
        dump(data, f, ensure_ascii=False)

    tokenizer = Tokenizer.from_file("temp.json")

    if latin:
        tokenizer.save("tokenizers/MiRe_bpe.json")
    else:
        tokenizer.save("tokenizers/MiRe_bpe_c.json")

for x in [True, False]:
    dataset = get_dataset(x)
    #dataset = dataset.select(range(1000))

    #train_bpe(dataset, x)
    #train_srna(dataset, x)
    #train_original_bpe(dataset, x)
    train_MiRe_bpe(dataset, x)
