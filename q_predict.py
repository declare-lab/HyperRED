import json
from ast import literal_eval
from typing import Dict, List, Tuple

import torch
from fire import Fire
from tqdm import tqdm

from data.q_process import Sentence as OrigSentence
from inputs.datasets.q_dataset import Dataset
from models.joint_decoding.q_decoder import (Entity, EntRelJointDecoder,
                                             Qualifier, Relation, Sentence)
from q_main import prepare_inputs


def run_predict(
    path: str = "ckpt/quintuplet/best_model",
    path_data="ckpt/quintuplet/dataset.pickle",
    path_out: str = "ckpt/quintuplet/pred_dev.json",
    data_split: str = "dev",
):
    model = EntRelJointDecoder.load(path)
    dataset = Dataset.load(path_data)
    cfg = model.cfg
    model.eval()
    outputs = []

    num_batches = dataset.get_dataset_size(data_split) // cfg.test_batch_size
    sents = []
    for _, batch in tqdm(
        dataset.get_batch(data_split, cfg.test_batch_size, None), total=num_batches
    ):
        with torch.no_grad():
            batch = prepare_inputs(batch, cfg.device)
            for raw in model.raw_predict(batch):
                outputs.append(raw)
                s = model.decode(**raw)
                sents.append(s)

    with open(path_out, "w") as f:
        for s in sents:
            f.write(s.json() + "\n")


def load_process_sents(path: str) -> List[Sentence]:
    sents = []
    with open(path) as f:
        for line in tqdm(f.readlines()):
            orig = OrigSentence(**json.loads(line))
            s = Sentence(
                text=orig.sentText,
                ents=[
                    Entity(span=e.offset, label=e.label) for e in orig.entityMentions
                ],
                relations=[
                    Relation(
                        head=literal_eval(r.em1Id),
                        tail=literal_eval(r.em2Id),
                        label=r.label,
                    )
                    for r in orig.relationMentions
                ],
                qualifiers=[
                    Qualifier(
                        head=literal_eval(q.em1Id),
                        tail=literal_eval(q.em2Id),
                        value=literal_eval(q.em3Id),
                        label=q.label,
                    )
                    for q in orig.qualifierMentions
                ],
            )
            sents.append(s)
    return sents


class Scorer:
    def run(self, pred: List[Sentence], gold: List[Sentence]) -> Dict[str, float]:
        raise NotImplementedError


class QuintupletScorer:
    def make_sent_tuples(
        self, s: Sentence
    ) -> List[Tuple[int, int, int, int, int, int, str, str]]:
        tuples = []
        relations = {(r.head, r.tail): r for r in s.relations}
        for q in s.qualifiers:
            r = relations.get((q.head, q.tail))
            if r is not None:
                tuples.append((*q.head, *q.tail, *q.value, r.label, q.label))
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


def match_sents(sents: List[Sentence], gold: List[Sentence]) -> List[Sentence]:
    text_to_sents = {s.text.lower(): s for s in sents}
    empty = Sentence.empty()
    outputs = [text_to_sents.get(s.text.lower(), empty) for s in gold]

    print("\nHow many pairs have empty preds?")
    print(dict(num=len([s for s in outputs if s.json() == empty.json()])))
    return outputs


def evaluate(
    path: str = "ckpt/quintuplet/pred_dev.json",
    path_gold: str = "data/quintuplet/dev.json",
):
    with open(path) as f:
        sents = [Sentence(**json.loads(line)) for line in tqdm(f)]
    print(dict(sents=len(sents)))
    gold = load_process_sents(path_gold)
    sents = match_sents(sents, gold)
    scorer = QuintupletScorer()
    results = scorer.run(sents, gold)
    print(json.dumps(results, indent=2))


def binary_search(fn, left: float, right: float, threshold: float):
    mid = (left + right) / 2
    if abs(fn(left) - fn(mid)) < threshold:
        return mid
    if fn(left) > fn(right):
        return binary_search(fn, left, mid, threshold)
    else:
        return binary_search(fn, mid, right, threshold)


"""
p q_predict.py run_predict
p q_predict.py evaluate

How many pairs have empty preds?
{
  "num_correct": 227,
  "num_pred": 695066,
  "num_gold": 6860,
  "precision": 0.0003265876909530893,
  "recall": 0.033090379008746354,
  "f1": 0.0006467918270586927
}
"""


if __name__ == "__main__":
    Fire()
