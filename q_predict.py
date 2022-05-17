from typing import List

from fire import Fire

from data.q_process import RawPred, Sentence
from inputs.datasets.q_dataset import Dataset
from inputs.vocabulary import Vocabulary
from models.joint_decoding.q_decoder import EntRelJointDecoder
from q_main import evaluate


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


def run_eval(
    path: str = "ckpt/quintuplet/best_model",
    path_data="ckpt/quintuplet/dataset.pickle",
    data_split: str = "dev",
):
    model = EntRelJointDecoder.load(path)
    dataset = Dataset.load(path_data)
    cfg = model.cfg
    evaluate(cfg, dataset, model, data_split)


if __name__ == "__main__":
    Fire()
