import ast
import json
import os
import random
from collections import Counter
from pathlib import Path
from pprint import pprint
from typing import List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from fire import Fire
from pydantic.main import BaseModel
from torch.nn.functional import one_hot
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer

from data.q_process import (
    RawPred,
    Sentence,
    SparseCube,
    load_raw_preds,
    load_sents,
    save_sents,
)
from inputs.datasets.q_dataset import Dataset
from models.joint_decoding.q_decoder import EntRelJointDecoder, decode_nonzero_cuboids
from models.joint_decoding.q_tagger import decode_nonzero_spans
from modules.token_embedders.bert_encoder import BertLinear
from q_main import evaluate, load_model, prepare_inputs, process_outputs, score_preds
from scoring import QuintupletScorer


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


def test_data(path: str = "data/ACE2005/test.json"):
    with open(path) as f:
        sents = [Sentence(**json.loads(line)) for line in f]
    print(dict(sents=len(sents)))

    for s in sents:
        if s.relationMentions:
            for k, v in s.dict().items():
                print(k, str(v)[:120])
            break

    print("\nHow many have span overlap?")
    print(len([s for s in sents if s.check_span_overlap()]))


def test_quintuplet_sents(path: str = "data/quintuplet/dev.json"):
    with open(path) as f:
        sents = [Sentence(**json.loads(line)) for line in tqdm(f)]

    print("\nHow many entities per sentence?")
    lengths = [len(s.entityMentions) for s in sents]
    print(np.mean(lengths))

    print("\nWhat fraction of the cubes (quintuplets) are empty?")
    total = 0
    filled = 0
    sizes = []
    counts = []
    for s in sents:
        assert s.quintupletMatrix is not None
        total += s.quintupletMatrix.numel()
        sizes.append(s.quintupletMatrix.numel())
        filled += len(s.quintupletMatrix.entries)
        counts.append(len(s.quintupletMatrix.entries))
    print(dict(frac=1 - (filled / total), sizes=np.mean(sizes), counts=np.mean(counts)))

    print("\nWhat fraction of the tables (relations) are empty?")
    total = 0
    filled = 0
    for s in sents:
        table = np.array(s.jointLabelMatrix)
        total += table.size
        filled += np.sum(table > 0)
    print(1 - (filled / total))

    print("\nWhat fraction of the diagonals (entities) are empty?")
    total = 0
    filled = 0
    lengths = []
    for s in sents:
        row = list(np.diagonal(np.array(s.jointLabelMatrix)))
        total += len(row)
        filled += sum(row)
        lengths.append(sum(row))
    print(dict(min=min(lengths), avg=np.mean(lengths), max=max(lengths)))
    print(1 - (filled / total))

    print("\nWhat is the average sentence length?")
    lengths = [len(s.sentText.split()) for s in sents]
    print(sum(lengths) / len(lengths))

    print("\nWhat is the average cube length?")
    lengths = [s.quintupletMatrix.shape[0] for s in sents]
    print(sum(lengths) / len(lengths))

    print("\nWhat is the average number of entity tokens in a sentence?")
    lengths = []
    for s in sents:
        tags = [0 for _ in s.sentText.split()]
        for e in s.entityMentions:
            for i in range(e.offset[0], e.offset[1]):
                tags[i] = 1
        lengths.append(sum(tags))
        assert list(np.diagonal(np.array(s.jointLabelMatrix))) == tags
    print(sum(lengths) / len(lengths))

    print("\nWhat is average entity length?")
    lengths = []
    for s in sents:
        for e in s.entityMentions:
            start, end = e.offset
            assert end > start
            lengths.append(end - start)
    print(dict(lengths=np.mean(lengths)))

    print("\nHow many quintuplets per sent on average?")
    lengths = [len(s.qualifierMentions) for s in sents]
    print(dict(lengths=np.mean(lengths)))

    print("\nManually analyze cube")
    sizes = []
    counts = []
    for s in sents:
        seq_len = len(s.sentText.split())
        cube = np.zeros(shape=(seq_len, seq_len, seq_len))
        id_to_span = {e.emId: e.offset for e in s.entityMentions}
        for q in s.qualifierMentions:
            head = id_to_span[q.em1Id]
            tail = id_to_span[q.em2Id]
            value = id_to_span[q.em3Id]
            assert len(set([head, tail, value])) == 3
            for i in range(*head):
                for j in range(*tail):
                    for k in range(*value):
                        cube[i, j, k] = 1
        sizes.append(cube.size)
        counts.append(cube.sum())
    print(
        dict(
            frac=sum(counts) / sum(sizes), sizes=np.mean(sizes), counts=np.mean(counts)
        )
    )

    print("\nWhat fraction of sentences have overlapping entities?")
    selected = []
    seen = set()
    for s in sents:
        tags = [0 for _ in s.sentText.split()]
        for e in s.entityMentions:
            for i in range(e.offset[0], e.offset[1]):
                if tags[i] == 1 and s.sentText not in seen:
                    seen.add(s.sentText)
                    selected.append(s)
                else:
                    tags[i] = 1
    print(dict(frac=len(selected) / len(sents)))

    print("\nIf restrict to top-50 qualifiers, how many quintuplets are affected?")
    top_k = 50
    qualifiers = []
    for s in sents:
        for q in s.qualifierMentions:
            qualifiers.append(q.label)
    counter = Counter(qualifiers)
    threshold = sorted(counter.values())[-top_k]
    remainder = sum(v for v in counter.values() if v >= threshold)
    print(dict(threshold=threshold, remainder=remainder, total=len(qualifiers)))


def test_sparse_cube(path: str = "data/q10/dev.json"):
    with open(path) as f:
        for line in tqdm(f.readlines()):
            sent = Sentence(**json.loads(line))
            matrix = sent.quintupletMatrix
            x = matrix.numpy()
            new = SparseCube.from_numpy(x)
            if not matrix.check_equal(new):
                print("Rarely (0.001), orig cube has multiple entries in same i,j,k")


def test_raw_q_preds(path: str = "ckpt/q10/raw_test.pkl"):
    preds = load_raw_preds(path)
    print("\nHow many preds have at least one q_matrix entry?")
    num = sum(1 for p in preds if len(p.quintuplet_preds.entries) > 0)
    print(dict(total=len(preds), num=num))


def test_decode_nonzero_spans():
    for labels in [[0, 0, 0], [0, 0, 1, 2], [0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 1]]:
        spans = decode_nonzero_spans(labels)
        print(dict(labels=labels, spans=spans, values=[labels[a:b] for a, b in spans]))


def analyze_sents(sents: List[Sentence]) -> dict:
    relations = [r.label for s in sents for r in s.relationMentions]
    qualifiers = [q.label for s in sents for q in s.qualifierMentions]
    entity_labels = [e.label for s in sents for e in s.entityMentions]
    info = dict(
        triplets=len(relations),
        quintuplets=len(qualifiers),
        ents=len(entity_labels),
        relations=len(set(relations)),
        qualifiers=len(set(qualifiers)),
        entity_labels=len(set(entity_labels)),
    )
    return info


def compare_tag_data(
    path_tag: str = "data/q10_tagger/dev.json", path_orig="data/q10/dev.json"
):
    with open(path_orig) as f:
        sents_orig = [Sentence(**json.loads(line)) for line in f]
    with open(path_tag) as f:
        sents_tag = [Sentence(**json.loads(line)) for line in f]

    print("\nOrig stats?")
    print(json.dumps(analyze_sents(sents_orig)))
    print("\nNew stats?")
    print(json.dumps(analyze_sents(sents_tag)))

    print("\nCan the spans in table be decoded correctly?")
    decoded_spans = []
    correct_spans = []
    for s in sents_tag:
        labels = np.array(s.jointLabelMatrix).diagonal()
        gold = set(e.offset for e in s.entityMentions)
        spans = decode_nonzero_spans([int(x) for x in labels])
        decoded_spans.extend(spans)
        correct_spans.extend([sp for sp in spans if sp in gold])
    print(dict(decoded_spans=len(decoded_spans), correct_spans=len(correct_spans)))


def test_decode_nonzero_cuboids(path: str = "data/q10/dev.json"):
    with open(path) as f:
        sents = [Sentence(**json.loads(line)) for line in tqdm(f.readlines())]

    cubes = [torch.tensor(s.quintupletMatrix.numpy()) for s in tqdm(sents)]
    for i, c in enumerate(tqdm(cubes, desc="nonzero")):
        assert c.nonzero().shape[0] > 0
        cuboids = decode_nonzero_cuboids(c)
        if len(sents[i].qualifierMentions) != len(cuboids):
            pprint(sents[i].qualifierMentions)
            pprint(cuboids)
            print()


def test_roberta(
    path: str = "ckpt/q10r_fix_q_loss/dataset.pickle", name: str = "roberta-base"
):
    device = torch.device("cpu")
    # device = torch.device("cuda")
    ds = Dataset.load(path)
    bs = 2
    model = AutoModel.from_pretrained(name)
    tok = AutoTokenizer.from_pretrained(name)
    model = model.to(device)

    for epoch, batch in ds.get_batch("train", bs, None):
        a = torch.tensor(batch["wordpiece_tokens"], device=device)
        b = torch.tensor(batch["wordpiece_segment_ids"], device=device)
        print(dict(a=a.shape, b=b.shape))
        mask = (a != 0).long()
        outputs = model(input_ids=a, attention_mask=mask)
        # outputs = model(input_ids=a, token_type_ids=b, attention_mask=mask)
        print(dict(epoch=epoch, **{k: v.shape for k, v in outputs.items()}))

        for lst in tok.batch_decode(a):
            print(lst)
        break


def compare_sents(
    path_a: str = "data/q10/dev.json", path_b: str = "data/q10_copy/dev.json"
):
    with open(path_a) as f:
        sents_a = [Sentence(**json.loads(line)) for line in f]
    with open(path_b) as f:
        sents_b = [Sentence(**json.loads(line)) for line in f]

    assert len(sents_a) == len(sents_b)
    for a, b in zip(sents_a, sents_b):
        assert a == b


def test_top_k():
    bs = 3
    seq_len = 4
    k = 2
    x = torch.rand(bs, seq_len)
    t = x.topk(k=k, dim=-1)
    print(x)
    print(t.indices)
    # breakpoint()


def test_transformer(
    device_ids: List[int],
    bs: int = 64,
    seq_len: int = 512,
    name: str = "bert-base-uncased",
):
    print(locals())
    devices = [torch.device(f"cuda:{i}") for i in device_ids]
    models = [AutoModel.from_pretrained(name).to(d) for d in devices]

    for _ in tqdm(range(int(1e9))):
        for d, m in zip(devices, models):
            x = torch.zeros(bs, seq_len, dtype=torch.long, device=d)
            y = m(x)
            assert y is not None


def test_prune_eval(
    path: str = "ckpt/quintuplet/best_model",
    path_data="ckpt/quintuplet/dataset.pickle",
    data_split: str = "dev",
    task: str = "quintuplet",
    path_in: str = "",
):
    model = EntRelJointDecoder.load(path)
    model.prune_topk = 20
    # model.prune_topk = 80

    dataset = Dataset.load(path_data)
    cfg = model.cfg
    evaluate(cfg, dataset, model, data_split, path_in=path_in)


def test_loader(path: str = "ckpt/q10/dataset.pickle"):
    ds = Dataset.load(path)
    bs = 32
    limit = 1000
    for i, _ in enumerate(tqdm(ds.get_batch("train", bs, None), total=limit)):
        print(_.keys())
        if i > limit:
            break


def test_tensor():
    bs = 32
    size = 80
    limit = 1000
    shape = (bs, size, size, size)
    zero = torch.zeros(*shape)

    for _ in tqdm(range(limit)):
        # x = torch.zeros(*shape)
        x = torch.zeros_like(zero)
        # x = np.zeros(shape)
        assert x.shape == zero.shape


def find_best(pattern: str = "ckpt/*prune*/train.log"):
    for path in sorted(Path().glob(pattern)):
        print(path)
        with open(path) as f:
            lines = [x for x in f.readlines() if "best_score" in x]
            print(lines[-1])


def test_adjacent_qualifiers(path: str = "data/q10/test.json"):
    with open(path) as f:
        sents = [Sentence(**json.loads(line)) for line in f]

    total = 0
    selected = 0
    for s in sents:
        groups = {}
        for q in s.qualifierMentions:
            groups.setdefault((q.em1Id, q.em2Id), []).append(q)
        for lst in groups.values():
            tags = [0 for _ in s.sentText.split()]
            for q in lst:
                span = ast.literal_eval(q.em3Id)
                total += 1
                for i in range(span[0], span[1]):
                    if tags[i] == 1:
                        selected += 1
                        break
                    tags[i] = 1
    print(dict(frac=selected / total))


class Biaffine(nn.Module):
    def __init__(self, f1: int, f2: int, f_out: int):
        super().__init__()
        self.bilinear = nn.Bilinear(f1, f2, f_out, bias=False)
        self.linear = BertLinear(f1 + f2, f_out, activation=nn.Identity())

    def get_shapes(self, x1, x2):
        assert len(x1.shape) == len(x2.shape)
        shape = list(x1.shape)
        for i, j in enumerate(x2.shape):
            if shape[i] == 1:
                shape[i] = j
        return shape, shape[:-1] + [x2.shape[-1]]

    def forward(self, x1, x2):
        shape1, shape2 = self.get_shapes(x1, x2)
        x1 = x1.expand(*shape1)
        x2 = x2.expand(*shape2)
        a = self.linear(torch.cat([x1, x2], dim=-1))
        b = self.bilinear(x1, x2)
        return a + b


def test_biaffine():
    bs = 32
    length = 20
    dim1 = 40
    dim2 = 23
    num_labels = 15

    head = torch.zeros(bs, length, dim1).unsqueeze(2)
    tail = torch.zeros(bs, length, dim2).unsqueeze(1)
    layer = Biaffine(dim1, dim2, num_labels)
    x = layer(head, tail)
    print(dict(x=x.shape))


def test_ign_score(
    path_pred: str = "ckpt/q10_pair2_fix_q_loss_prune_20/test.json",
    path_gold: str = "data/q10/test.json",
    path_train: str = "data/q10/train.json",
):
    # score_preds(path_pred, path_gold)
    preds = load_sents(path_pred)
    sents = load_sents(path_gold)

    facts = set(
        q.as_texts(s.tokens, s.relationMentions)
        for s in load_sents(path_train)
        for q in s.qualifierMentions
    )

    for s in sents:
        s.qualifierMentions = [
            q
            for q in s.qualifierMentions
            if q.as_texts(s.tokens, s.relationMentions) not in facts
        ]

    for s in preds:
        s.qualifierMentions = [
            q
            for q in s.qualifierMentions
            if q.as_texts(s.tokens, s.relationMentions) not in facts
        ]

    save_sents(sents, "temp_gold.json")
    save_sents(preds, "temp_pred.json")
    score_preds("temp_pred.json", "temp_gold.json")
    os.remove("temp_pred.json")
    os.remove("temp_gold.json")


# "precision": 0.5429590996431513,
# "recall": 0.5486823855755895,
# "f1": 0.5458057395143487


def find_words(text: str, words: List[str]) -> bool:
    return any(w in text for w in words)


def classify_qualifier(label: str, value: str) -> str:
    if find_words(label, ["time", "date"]):
        return "time"
    if value.isdigit() or find_words(
        label, "ordinal number quantity ranking appearances proportion level".split()
    ):
        return "number"
    if find_words(
        label,
        "together part separated represent league instance affiliation member replace follow".split(),
    ):
        return "part-whole"
    if label == "of" or find_words(
        label, "adjacent connect locat district country towards diocese".split()
    ):
        return "location"
    # if find_words(
    #     label,
    #     "replace follow degree statement cause work major member use".split(),
    # ):
    #     return "cause"
    if find_words(
        label,
        "mother nominee winner performer position role academic work operator statement father".split(),
    ):
        return "role"
    return "others"


def filter_qualifiers(s: Sentence, label: str) -> Sentence:
    s = s.copy(deep=True)
    mentions = []
    for q in s.qualifierMentions:
        _, _, _, qualifier, value = q.as_texts(s.tokens, s.relationMentions)
        if classify_qualifier(qualifier, value) == label:
            mentions.append(q)
    s.qualifierMentions = mentions
    return s


def test_separate_eval(path_pred: str, path_gold: str):
    path_temp_pred = "temp_pred.json"
    path_temp_gold = "temp_gold.json"
    sents_pred = load_sents(path_pred)
    sents_gold = load_sents(path_gold)

    records = []
    for label in "location time number part-whole role".split():
        # pred = sents_pred
        # gold = sents_gold
        pred = [filter_qualifiers(s, label) for s in sents_pred]
        gold = [filter_qualifiers(s, label) for s in sents_gold]
        save_sents(pred, path_temp_pred)
        save_sents(gold, path_temp_gold)
        r = score_preds(path_temp_pred, path_temp_gold)
        r = dict(label=label, score=r["quintuplet"]["f1"])
        records.append(r)

        os.remove(path_temp_pred)
        os.remove(path_temp_gold)
        for r in records:
            print(r)


class TacredSentence(BaseModel):
    id: str
    docid: str
    relation: str
    token: List[str]
    subj_start: int
    subj_end: int
    obj_start: int
    obj_end: int
    subj_type: str
    obj_type: str

    @property
    def text(self) -> str:
        return " ".join(self.token)

    @property
    def triplet_texts(self) -> Tuple[str, str, str]:
        head = " ".join(self.token[self.subj_start : self.subj_end + 1])
        tail = " ".join(self.token[self.obj_start : self.obj_end + 1])
        return (head, self.relation, tail)


class TacredData(BaseModel):
    sents: List[TacredSentence]

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            sents = [TacredSentence(**raw) for raw in tqdm(json.load(f), desc=path)]
        return cls(sents=sents)

    def analyze(self):
        info = dict(sents=len(self.sents), texts=len(set(s.text for s in self.sents)))
        print(json.dumps(info, indent=2))


def test_tacred(pattern: str = "data/tacred/data/json/*.json"):
    facts = []

    for path in sorted(Path().glob(pattern)):
        print(path)
        data = TacredData.load(str(path))
        data.analyze()
        for s in data.sents:
            facts.append(s.triplet_texts)

    print(facts[:10])
    print(dict(unique_facts=len(set(facts))))


class NytSentence(BaseModel):
    text: str
    pointer: str

    @property
    def tokens(self) -> List[str]:
        return self.text.split()

    @property
    def triplets(self) -> List[Tuple[int, int, int, int, str]]:
        triplets = []
        parts = self.pointer.split(" | ")
        for p in parts:
            a, b, c, d, e = p.split()
            triplets.append((int(a), int(b), int(c), int(d), e))
        return triplets

    @property
    def triplet_texts(self) -> List[Tuple[str, str, str]]:
        tokens = self.tokens
        texts = []
        for a, b, c, d, e in self.triplets:
            head = " ".join(tokens[a : b + 1])
            tail = " ".join(tokens[c : d + 1])
            texts.append((head, e, tail))
        return texts


class NytData(BaseModel):
    sents: List[NytSentence]

    @classmethod
    def load(cls, path_text: str, path_pointer: str):
        with open(path_text) as f:
            texts = [line.strip() for line in f]
        with open(path_pointer) as f:
            pointers = [line.strip() for line in f]
        assert len(texts) == len(pointers)
        return cls(
            sents=[NytSentence(text=t, pointer=p) for t, p in zip(texts, pointers)]
        )

    def analyze(self):
        info = dict(
            sents=len(self.sents),
            facts=sum(len(s.triplets) for s in self.sents),
            texts=len(set(s.text for s in self.sents)),
        )
        print(json.dumps(info, indent=2))


def test_nyt_data(pattern: str = "data/PtrNetDecoding4JERE/*"):
    for folder in sorted(Path().glob(pattern)):
        if not folder.is_dir():
            continue
        facts = []
        for data_split in "train dev test".split():
            path_text = str(folder / f"{data_split}.sent")
            path_pointer = str(folder / f"{data_split}.pointer")
            data = NytData.load(path_text, path_pointer)
            print(path_text)
            data.analyze()

            for s in data.sents:
                for t in s.triplet_texts:
                    facts.append(t)

        print(facts[:10])
        print(dict(unique_facts=len(set(facts))))
        print("#" * 80)


def test_unique_facts(folder: str = "data/q10"):
    for data_split in "train dev test".split():
        path = Path(folder) / f"{data_split}.json"
        sents = load_sents(str(path))
        facts = []
        entities = []
        lengths = []
        ent_lengths = []

        for s in sents:
            lengths.append(len(s.tokens))
            for q in s.qualifierMentions:
                f = q.as_texts(s.tokens, s.relationMentions)
                facts.append(f)
                entities.append(f[0])
                entities.append(f[2])
                entities.append(f[4])
            for e in s.entityMentions:
                start, end = e.offset
                assert start < end
                ent_lengths.append(end - start)

        info = dict(
            facts=len(set(facts)),
            entities=len(set(entities)),
            lengths=sum(lengths) / len(lengths),
            ent_lengths=np.mean(ent_lengths),
        )
        print(json.dumps(info, indent=2))


def sent_to_tuples(s: Sentence) -> Set[Tuple[str, str, str, str, str]]:
    return set(q.as_texts(s.tokens, s.relationMentions) for q in s.qualifierMentions)


def test_cases(
    path_cube: str = "ckpt/q10_pair2_no_value_prune_20_seed_0/test.json",
    path_pipe: str = "ckpt/q10_tags_distilbert_seed_0/pred.json",
    path_gen: str = "data/q10/gen_pred.json",
    path_gold: str = "data/q10/test.json",
):
    scorer = QuintupletScorer()
    sents_gold = load_sents(path_gold)
    sents_cube = scorer.match_gold_to_pred(load_sents(path_cube), sents_gold)
    sents_pipe = scorer.match_gold_to_pred(load_sents(path_pipe), sents_gold)
    sents_gen = scorer.match_gold_to_pred(load_sents(path_gen), sents_gold)

    records = []
    for i, s in enumerate(sents_gold):
        gold = sent_to_tuples(s)
        cube = sent_to_tuples(sents_cube[i])
        pipe = sent_to_tuples(sents_pipe[i])
        gen = sent_to_tuples(sents_gen[i])
        if gold == cube and pipe == set() and gold != gen and pipe != gen:
            info = dict(
                text=s.sentText,
                gold=str(gold),
                cube=str(cube),
                pipe=str(pipe),
                gen=str(gen),
            )
            records.append(info)

    records = sorted(records, key=lambda x: len(str(x)), reverse=True)
    for info in records:
        print(json.dumps(info, indent=2))
    print(dict(records=len(records)))

    """
    {
      "text": "Nancy Davis Reagan ( born Anne Frances Robbins , July 6 , 1921 ) is a former actress and 
      the widow of the 40th President of the United States , Ronald Reagan .",                     
      "gold": "{('Ronald', 'position held', 'President', 'series ordinal', '40th')}",
      "cube": "{('Ronald', 'position held', 'President', 'series ordinal', '40th')}",
      "pipe": "set()",                                                              
      "gen": "{('Nancy Davis Reagan', 'spouse', 'Ronald', 'series ordinal', '40th')}"
    }
    {
      "text": "Nancy Davis Reagan is a former actress and the widow of the 
               40th President of the United States , Ronald Reagan .",                     
      "gold": "{('Ronald', 'position held', 'President', 'series ordinal', '40th')}",
      "cube": "{('Ronald', 'position held', 'President', 'series ordinal', '40th')}",
      "pipe": "set()",                                                              
      "gen": "{('Nancy Davis Reagan', 'spouse', 'Ronald', 'series ordinal', '40th')}"
    }
    The pipeline model is unable to detect the hyper-relational fact due to cascading errors
    that impair the recall performance.
    The generative model did not explicitly consider the interaction between relation triplet
    and qualifier, hence it predicted an invalid hyper-relational fact.
    """


def test_decoding(
    path: str = "ckpt/q10_pair2_no_value_prune_20_seed_0/best_model",
    path_data: str = "ckpt/q10_pair2_no_value_prune_20_seed_0/dataset.pickle",
    path_gold: str = "data/q10/test.json",
    data_split: str = "test",
    task: str = "quintuplet",
    path_in: str = "",
):
    model = load_model(task, path)
    dataset = Dataset.load(path_data)
    cfg = model.cfg
    all_outputs = []

    num_batches = dataset.get_dataset_size(data_split) // cfg.test_batch_size
    for _, batch in tqdm(
        dataset.get_batch(data_split, cfg.test_batch_size, None), total=num_batches
    ):
        inputs = prepare_inputs(batch, cfg.device)
        num_r = model.vocab.get_vocab_size("ent_rel_id")
        num_q = model.ent_rel_file["q_num_logits"]
        q_scores = one_hot(inputs["quintuplet_matrix"], num_q).float()
        r_scores = one_hot(inputs["joint_label_matrix"], num_r).float()
        batch_seq_tokens_lens = inputs["tokens_lens"]
        assert isinstance(model, EntRelJointDecoder)

        outputs = model.soft_joint_decoding(
            batch_normalized_joint_score=r_scores,
            batch_seq_tokens_lens=batch_seq_tokens_lens,
            batch_normalized_q_score=q_scores,
            prune_indices=None,
        )
        outputs.update(
            quintuplet_preds=inputs["quintuplet_matrix"],
            joint_label_preds=inputs["joint_label_matrix"],
        )
        all_outputs.extend(process_outputs(inputs, outputs))

    sents = [RawPred(**r).as_sentence(model.vocab) for r in all_outputs]
    save_sents(sents, "temp.json")
    score_preds("temp.json", path_gold)
    os.remove("temp.json")

    # "precision": 0.9992310649750096,
    # "recall": 0.9714072136049337,
    # "f1": 0.9851227139202122


def test_preds(path_pred: str, path_gold: str):
    sents_pred = load_sents(path_pred)
    sents_gold = load_sents(path_gold)
    text_to_gold = {s.sentText: s for s in sents_gold}
    limit = 10

    count = 0
    for s in sents_pred:
        s2 = text_to_gold[s.sentText]
        lst = [
            q.as_texts(s.tokens, s.relationMentions)[:3] for q in s.qualifierMentions
        ]
        gold = [
            q.as_texts(s2.tokens, s2.relationMentions)[:3] for q in s2.qualifierMentions
        ]

        if sorted(lst) != sorted(gold):
            print(s.sentText)
            print(dict(gold=gold))
            print(dict(pred=lst))
            print()
            count += 1
            if count > limit:
                break

    info = dict(
        pred_labels=Counter(r.label for s in sents_pred for r in s.relationMentions),
        gold_labels=Counter(r.label for s in sents_gold for r in s.relationMentions),
        pred_tuples=sum(len(s.qualifierMentions) for s in sents_pred),
        gold_tuples=sum(len(s.qualifierMentions) for s in sents_gold),
    )
    print(json.dumps(info, indent=2))
    breakpoint()


def score_preds_many(folder: str, path_gold: str):
    results = []
    for path in tqdm(sorted(Path(folder).glob("*/test.json"))):
        r = score_preds(str(path), path_gold)
        r["path"] = str(path)
        results.append(r)

    results = sorted(results, key=lambda r: r["quintuplet"]["f1"])
    for r in results:
        print(round(r["quintuplet"]["f1"], 3), r["path"])


"""
Findings
- FP16 doesn't significantly change speed or results
- RoBERTa is better than BERT (+1 F1) 
- Removing softmax before crossentropy helps (+6 F1)
- Separate MLP for triplet and quintuplet helps (+2 F1)
- Auxiliary entity seq labeling loss doesn't help (-4 F1)
- Distant training then labeled continue train helps (+3 F1)
- Cube-pruning helps
- No decay helps for tagger, it may help cube model (+0.5 F1)

Tasks
- position embeddings

p analysis.py test_separate_eval ckpt/q10_pair2_no_value_prune_20_seed_0/test.json data/q10/test.json
{'label': 'time', 'score': 0.623048033208144}    
{'label': 'number', 'score': 0.7924528301886793}
{'label': 'role', 'score': 0.523168908819133}
{'label': 'part-whole', 'score': 0.751417004048583}
{'label': 'location', 'score': 0.5505984766050054}

p analysis.py test_separate_eval ckpt/q10_tags_distilbert_seed_0/pred.json data/q10/test.json
{'label': 'time', 'score': 0.5956365176869308}
{'label': 'number', 'score': 0.7781672508763143}
{'label': 'role', 'score': 0.4806338028169014}
{'label': 'part-whole', 'score': 0.7060931899641578}
{'label': 'location', 'score': 0.5080091533180778}

p analysis.py test_separate_eval data/q10/gen_pred.json data/q10/test.json
{'label': 'time', 'score': 0.5808540781218376}
{'label': 'number', 'score': 0.765371372356124}
{'label': 'role', 'score': 0.5331230283911672}
{'label': 'part-whole', 'score': 0.6255430060816682}
{'label': 'location', 'score': 0.5340659340659342}

"""


if __name__ == "__main__":
    Fire()
