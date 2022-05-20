import pickle
from typing import List

from fire import Fire

from data.q_process import RawPred, Sentence
from inputs.datasets.q_dataset import Dataset
from inputs.vocabulary import Vocabulary
from models.joint_decoding.q_decoder import EntRelJointDecoder
from models.joint_decoding.q_tagger import EntRelJointDecoder as Tagger
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
    task: str = "quintuplet",
):
    if task == "tagger":
        model = Tagger.load(path)
    else:
        model = EntRelJointDecoder.load(path)

    dataset = Dataset.load(path_data)
    cfg = model.cfg
    evaluate(cfg, dataset, model, data_split)


def load_raw_preds(path: str) -> List[RawPred]:
    raw_preds = []
    with open(path, "rb") as f:
        raw = pickle.load(f)
        for r in raw:
            p = RawPred(**r)
            p.assert_valid()
            raw_preds.append(p)
    return raw_preds


def merge_pipeline_preds(
    path_triplets: str,
    path_tags: str,
    path_vocab_triplets: str,
    path_vocab_tags: str,
    path_out: str,
    sep: str = " | ",  # Refer to convert_sent_to_tags
    ent_label: str = "Entity",  # Refer to make sentences
):
    v_triplets = Vocabulary.load(path_vocab_triplets)
    v_tags = Vocabulary.load(path_vocab_tags)
    raw_triplets = load_raw_preds(path_triplets)
    raw_tags = load_raw_preds(path_tags)
    s_triplets = [r.as_sentence(v_triplets) for r in raw_triplets]
    s_tags = [r.as_sentence(v_tags) for r in raw_tags]
    text_to_i = {s.sentText: i for i, s in enumerate(s_triplets)}

    assert all(sep not in text for text in text_to_i.keys())
    for j, s in enumerate(s_tags):
        text, *parts = s.sentText.split(sep)
        assert len(parts) == 3
        tokens = text.split()
        i = text_to_i.get(text)

        if i is None:
            print(dict(unmatched=text))
        else:
            id_to_entity = {e.emId: e for e in s_triplets[i].entityMentions}
            for r in s_triplets[i].relationMentions:
                head = id_to_entity[r.em1Id]
                tail = id_to_entity[r.em2Id]
                head.text = " ".join(tokens[slice(*head.offset)])
                tail.text = " ".join(tokens[slice(*tail.offset)])
                if (head.text, r.label, tail.text) == tuple(parts):
                    for e in s.entityMentions:
                        raw_triplets[i].all_q_preds[
                            (head.offset, tail.offset, e.offset)
                        ] = e.label
                        raw_triplets[i].all_ent_preds[e.offset] = ent_label

    print(dict(quintuplets=sum(len(r.all_q_preds) for r in raw_triplets)))
    with open(path_out, "wb") as f:
        outputs = [r.dict() for r in raw_triplets]
        pickle.dump(outputs, f)


"""
p q_predict.py merge_pipeline_preds \
--path_triplets ckpt/q10_triplet/pred.pkl \
--path_tags ckpt/q10_tagger/raw_test.pkl \
--path_vocab_triplets ckpt/q10_triplet/vocabulary.pickle \
--path_vocab_tags ckpt/q10_tagger/vocabulary.pickle \
--path_out ckpt/q10_tagger/pred.pkl

p analysis.py test_preds \
--path_pred ckpt/q10_tagger/pred.pkl \
--path_gold data/q10/test.json \
--path_vocab ckpt/q10_triplet/vocabulary.pickle

{
  "scorer": "QuintupletScorer",
  "num_correct": 1762,
  "num_pred": 1959,
  "num_gold": 2595,
  "precision": 0.8994384890250128,
  "recall": 0.6789980732177264,
  "f1": 0.7738252086078172
}

"""

if __name__ == "__main__":
    Fire()
