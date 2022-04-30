import json
import random

import torch
from fire import Fire
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer


def test_lengths(
    path: str = "json/dev.json",
    name: str = "bert-base-uncased",
    percentile: float = 0.95,
):
    texts = []

    with open(path) as f:
        for line in f:
            raw = json.loads(line)
            for lst in raw["sentences"]:
                texts.append(" ".join(lst))

    random.seed(0)
    for t in random.sample(texts, k=10):
        print(t)

    tokenizer = AutoTokenizer.from_pretrained(name)
    lengths = [len(tokenizer(t).input_ids) for t in tqdm(texts)]
    value = sorted(lengths)[round(len(texts) * percentile)]
    print(dict(value=value, percentile=percentile, min=min(lengths), max=max(lengths)))


def test_table_size(
    batch_size: int = 32,
    seq_len: int = 48,
    hidden_size: int = 151,
    num_labels: int = 14,
):
    head = torch.rand(batch_size, seq_len, hidden_size).cuda()
    tail = torch.rand(batch_size, seq_len, hidden_size).cuda()
    U = torch.rand(num_labels, hidden_size, hidden_size).cuda()

    triplet_score = torch.einsum("bxi, oij, byj -> bxyo", head, U, tail)
    print(dict(triplet_score=triplet_score.shape))  # (batch, len, len, labels)

    value = torch.rand(batch_size, seq_len, hidden_size).cuda()
    V = torch.zeros(num_labels, hidden_size * 2, hidden_size).cuda()
    pair = torch.cat(
        [
            head.unsqueeze(dim=2).expand(-1, -1, seq_len, -1),
            tail.unsqueeze(dim=1).expand(-1, seq_len, -1, -1),
        ],
        dim=-1,
    )
    final_score = torch.einsum("bxyi, oij, bzj -> bxyzo", pair, V, value)
    print(dict(final_score=final_score.shape))
    breakpoint()


def test_nyt(
    path="nyt/train.json",
    path_words="nyt/words2id.json",
    path_relations="nyt/relations2id.json",
):
    with open(path_words) as f:
        w2i = json.load(f)
        i2w = {i: w for w, i in w2i.items()}

    with open(path_relations) as f:
        r2i = json.load(f)
        i2r = {i: r for r, i in r2i.items()}

    with open(path) as f:
        raw = json.load(f)
        a, b, c = raw
        assert len(a) == len(b)
        assert len(a) == len(c)

        for i in tqdm(range(len(a))):
            words = [i2w[_] for _ in b[i]]
            assert len(c[i]) % 3 == 0
            assert len(c[i]) > 0

            for j in range(0, len(c[i]), 3):
                head, tail, k = c[i][j : j + 3]
                relation = i2r[k]
                info = dict(
                    a=a[i],
                    a_word=i2w[a[i]],
                    text=" ".join(words),
                    head=words[head],
                    tail=words[tail],
                    relation=relation,
                )
                print(json.dumps(info, indent=2))


if __name__ == "__main__":
    Fire()
