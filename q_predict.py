import json
import pickle
from typing import List

from fire import Fire

from data.q_process import RawPred, Sentence
from inputs.datasets.q_dataset import Dataset
from inputs.vocabulary import Vocabulary
from models.joint_decoding.q_decoder import EntRelJointDecoder
from q_main import evaluate
from scoring import EntityScorer, QuintupletScorer, StrictScorer


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


def binary_search(fn, left: float, right: float, threshold: float):
    mid = (left + right) / 2
    if abs(fn(left) - fn(mid)) < threshold:
        return mid
    if fn(left) > fn(right):
        return binary_search(fn, left, mid, threshold)
    else:
        return binary_search(fn, mid, right, threshold)


def run_eval(
    path: str = "ckpt/quintuplet/best_model",
    path_data="ckpt/quintuplet/dataset.pickle",
    data_split: str = "dev",
):
    model = EntRelJointDecoder.load(path)
    dataset = Dataset.load(path_data)
    cfg = model.cfg
    evaluate(cfg, dataset, model, data_split)


"""
p q_predict.py run_eval --data_split dev
p q_predict.py run_eval --data_split test

p analysis.py test_preds \
--path_pred ckpt/quintuplet/raw_dev.pkl \
--path_gold data/quintuplet/dev.json \
--path_vocab ckpt/quintuplet/vocabulary.pickle

{
  "scorer": "StrictScorer",
  "num_correct": 3566,                                  
  "num_pred": 5167,   
  "num_gold": 6203,
  "precision": 0.6901490226437004,
  "recall": 0.5748831210704498,   
  "f1": 0.6272647317502199                              
}
{
  "scorer": "QuintupletScorer",
  "num_correct": 2021,
  "num_pred": 3341,
  "num_gold": 6860,
  "precision": 0.6049087099670757,
  "recall": 0.2946064139941691,
  "f1": 0.3962356631702774
}

p analysis.py test_preds \
--path_pred ckpt/quintuplet/raw_test.pkl \
--path_gold data/quintuplet/test.json \
--path_vocab ckpt/quintuplet/vocabulary.pickle

{
  "scorer": "StrictScorer",
  "num_correct": 3639,                                  
  "num_pred": 5199,   
  "num_gold": 6093,
  "precision": 0.6999422965954991,
  "recall": 0.5972427375677006,  
  "f1": 0.6445270988310309                              
}
{
  "scorer": "QuintupletScorer",
  "num_correct": 2000,
  "num_pred": 3294,
  "num_gold": 6738,
  "precision": 0.607164541590771,
  "recall": 0.2968239833778569,
  "f1": 0.39872408293460926
}

"""


if __name__ == "__main__":
    Fire()
