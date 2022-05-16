import json
import pickle
import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
from fire import Fire
from pydantic import BaseModel
from pydantic.main import Extra
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer

from data.q_process import Sentence as QuintupletSentence
from inputs.vocabulary import Vocabulary
from models.joint_decoding.q_decoder import Sentence as PredSentence


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


class Entity(BaseModel, extra=Extra.forbid):
    emId: str
    text: str
    offset: Tuple[int, int]  # Token spans, start inclusive, end exclusive
    label: str


class Relation(BaseModel, extra=Extra.forbid):
    em1Id: str
    em1Text: str
    em2Id: str
    em2Text: str
    label: str


class Sentence(BaseModel):
    articleId: str
    sentId: int
    sentText: str
    entityMentions: List[Entity]
    relationMentions: List[Relation]
    wordpieceSentText: str
    wordpieceTokensIndex: List[Tuple[int, int]]
    wordpieceSegmentIds: List[int]
    jointLabelMatrix: List[List[int]]

    def check_span_overlap(self) -> bool:
        entity_pos = [0 for _ in range(9999)]
        for e in self.entityMentions:
            st, ed = e.offset
            for i in range(st, ed):
                if entity_pos[i] != 0:
                    return True
                entity_pos[i] = 1
        return False

    def as_relation_tuples(self) -> List[Tuple[int, int, int, int, str]]:
        id_to_entity = {e.emId: e for e in self.entityMentions}
        tuples = []
        for r in self.relationMentions:
            head = id_to_entity[r.em1Id].offset
            tail = id_to_entity[r.em2Id].offset
            tuples.append((head[0], head[1], tail[0], tail[1], r.label))
        return tuples


class RawPred(BaseModel, extra=Extra.forbid, arbitrary_types_allowed=True):
    tokens: np.ndarray
    span2ent: Dict[Tuple[int, int], str]
    span2rel: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int]
    seq_len: int
    joint_label_matrix: np.ndarray
    joint_label_preds: np.ndarray
    separate_positions: List[int]
    all_separate_position_preds: List[int]
    all_ent_preds: Dict[Tuple[int, int], str]
    all_rel_preds: Dict[Tuple[Tuple[int, int], Tuple[int, int]], str]

    def assert_valid(self):
        assert self.tokens.size > 0
        assert self.joint_label_matrix.size > 0
        assert self.joint_label_preds.size > 0

    @classmethod
    def empty(cls):
        return cls(
            tokens=np.empty(shape=(1,)),
            span2ent={},
            span2rel={},
            seq_len=0,
            joint_label_matrix=np.empty(shape=(1,)),
            joint_label_preds=np.empty(shape=(1,)),
            separate_positions=[],
            all_separate_position_preds=[],
            all_ent_preds={},
            all_rel_preds={},
        )

    def check_if_empty(self):
        return self.seq_len == 0

    def has_relations(self) -> bool:
        return len(self.all_rel_preds.keys()) > 0

    def as_relation_tuples(self) -> List[Tuple[int, int, int, int, str]]:
        tuples = []
        for (head, tail), label in self.all_rel_preds.items():
            tuples.append((head[0], head[1], tail[0], tail[1], label))
        return tuples

    def as_sentence(self, vocab: Vocabulary) -> Sentence:
        tokens = [vocab.get_token_from_index(i, "tokens") for i in self.tokens]
        tokens = [t for t in tokens if t != vocab.DEFAULT_PAD_TOKEN]
        text = " ".join(tokens)

        span_to_ent = {}
        for span, label in self.all_ent_preds.items():
            e = Entity(emId=str((span, label)), offset=span, text="", label=label)
            span_to_ent[span] = e

        relations = []
        for (head, tail), label in self.all_rel_preds.items():
            head_id = span_to_ent[head].emId
            tail_id = span_to_ent[tail].emId
            r = Relation(
                em1Id=head_id, em2Id=tail_id, em1Text="", em2Text="", label=label
            )
            relations.append(r)

        return Sentence(
            articleId=str((text, relations)),
            sentText=text,
            entityMentions=list(span_to_ent.values()),
            relationMentions=relations,
            sentId=0,
            wordpieceSentText="",
            wordpieceTokensIndex=[],
            wordpieceSegmentIds=[],
            jointLabelMatrix=[],
        )


def is_span_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return (
        b[0] <= a[0] < b[1]
        or b[0] <= a[1] < b[1]
        or a[0] <= b[0] < a[1]
        or a[0] <= b[1] < a[1]
    )


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


def match_sent_preds(
    sents: List[Sentence], raw_preds: List[RawPred], vocab: Vocabulary
) -> List[Sentence]:
    preds = [p.as_sentence(vocab) for p in raw_preds]
    text_to_pred = {p.sentText.lower(): p for p in preds}

    empty = RawPred.empty().as_sentence(vocab)
    outputs = [text_to_pred.get(s.sentText.lower(), empty) for s in sents]

    print("\nHow many pairs have empty preds?")
    print(dict(num=len([p for p in outputs if p == empty])))
    return outputs


class Scorer:
    def run(self, pred: List[Sentence], gold: List[Sentence]) -> Dict[str, float]:
        raise NotImplementedError


class StrictScorer:
    def make_sent_tuples(
        self, s: Sentence
    ) -> List[Tuple[Tuple[int, int, str], Tuple[int, int, str], str]]:
        id_to_entity = {e.emId: e for e in s.entityMentions}
        tuples = []
        for r in s.relationMentions:
            head = id_to_entity[r.em1Id]
            tail = id_to_entity[r.em2Id]
            t = (
                (head.offset[0], head.offset[1], head.label),
                (tail.offset[0], tail.offset[1], tail.label),
                r.label,
            )
            tuples.append(t)
        return tuples

    def run(self, pred: List[Sentence], gold: List[Sentence]) -> Dict[str, float]:
        assert len(pred) == len(gold)
        num_correct = 0
        num_pred = 0
        num_gold = 0

        for p, g in zip(pred, gold):
            tuples_pred = self.make_sent_tuples(p)
            tuples_gold = self.make_sent_tuples(g)
            num_pred += len(tuples_pred)
            num_gold += len(tuples_gold)
            for a in tuples_pred:
                for b in tuples_gold:
                    if a == b:
                        num_correct += 1

        precision = num_correct / num_pred
        recall = num_correct / num_gold
        f1 = (2 * precision * recall) / (precision + recall)
        return dict(
            num_correct=num_correct,
            num_pred=num_pred,
            num_gold=num_gold,
            precision=precision,
            recall=recall,
            f1=f1,
        )


def test_preds(
    path_pred: str = "ckpt/ace2005_bert/pred.pkl",
    path_gold: str = "data/ACE2005/test.json",
    path_vocab: str = "ckpt/ace2005_bert/vocabulary.pickle",
):
    raw_preds = []
    with open(path_pred, "rb") as f:
        raw = pickle.load(f)
        for r in raw:
            p = RawPred(**r)
            p.assert_valid()
            raw_preds.append(p)

    vocab = Vocabulary.load(path_vocab)
    for p in raw_preds:
        if p.has_relations():
            for k, v in p.dict().items():
                print(k, str(v)[:80])
            tokens = [vocab.get_token_from_index(i, "tokens") for i in p.tokens]
            print(dict(tokens=tokens))
            break

    print(dict(preds=len(raw_preds)))
    print("\nHow many preds have seq_len==0?")
    print(dict(num=len([p for p in raw_preds if p.seq_len == 0])))
    with open(path_gold) as f:
        sents = [Sentence(**json.loads(line)) for line in f]

    preds = match_sent_preds(sents, raw_preds, vocab)
    scorer = StrictScorer()
    results = scorer.run(preds, sents)
    print(json.dumps(results, indent=2))


def test_quintuplet_sents(path: str = "data/quintuplet/dev.json"):
    with open(path) as f:
        sents = [QuintupletSentence(**json.loads(line)) for line in tqdm(f)]

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
    for s in sents:
        row = list(np.diagonal(np.array(s.jointLabelMatrix)))
        total += len(row)
        filled += sum(row)
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


def test_q_preds(
    path: str = "ckpt/quintuplet/pred_dev.json",
    path_gold: str = "data/quintuplet/dev.json",
):
    with open(path) as f:
        sents = [PredSentence(**json.loads(line)) for line in tqdm(f)]
    print(dict(sents=len(sents)))

    print("\nWhat are ent/relation/qualifier counts and scores?")
    for key in ["ents", "relations", "qualifiers"]:
        counts = [len(getattr(s, key)) for s in sents]
        scores = [getattr(x, "score") for s in sents for x in getattr(s, key)]
        info = dict(
            key=key,
            counts=dict(min=min(counts), max=max(counts), avg=np.mean(counts)),
            scores=dict(min=min(scores), max=max(scores), avg=np.mean(scores)),
        )
        print(json.dumps(info, indent=2))

    print("\nWhat are the predicted qualifier labels?")
    print(Counter(q.label for s in sents for q in s.qualifiers))

    random.seed(0)
    for _ in range(10):
        i = random.randint(0, len(sents))
        s = sents[i]
        q = sorted(s.qualifiers, key=lambda q: q.score)[-1]
        tokens = s.text.split()
        head = " ".join(tokens[q.head[0] : q.head[1]])
        tail = " ".join(tokens[q.tail[0] : q.tail[1]])
        value = " ".join(tokens[q.value[0] : q.value[1]])
        print((head, tail, value, q.label))


if __name__ == "__main__":
    Fire()
