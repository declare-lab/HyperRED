import pickle

from fire import Fire

from data.q_process import load_raw_preds, process_tags
from inputs.vocabulary import Vocabulary
from q_main import run_eval, score_preds

assert run_eval is not None
assert score_preds is not None


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


def prepare_tagger_pred_inputs(
    path_triplets: str,
    path_vocab: str,
    path_out: str,
    path_temp: str = "temp.json",
    **kwargs
):
    vocab = Vocabulary.load(path_vocab)
    with open(path_temp, "w") as f:
        for r in load_raw_preds(path_triplets):
            f.write(r.as_sentence(vocab).json() + "\n")
    process_tags(path_in="", path_out=path_out, path_sents_in=path_temp, **kwargs)


"""
p q_predict.py prepare_tagger_pred_inputs \
--path_triplets ckpt/q10_triplet/pred.pkl \
--path_vocab ckpt/q10_triplet/vocabulary.pickle \
--path_out ckpt/q10_triplet/tagger_in.json \
--label_file data/q10/label.json

p q_predict.py run_eval ckpt/q10_tagger/best_model ckpt/q10_tagger/dataset.pickle \
--data_split dummy \
--task tagger \
--path_in ckpt/q10_triplet/tagger_in.json

p q_predict.py merge_pipeline_preds \
--path_triplets ckpt/q10_triplet/pred.pkl \
--path_tags ckpt/q10_tagger/raw_pred.pkl \
--path_vocab_triplets ckpt/q10_triplet/vocabulary.pickle \
--path_vocab_tags ckpt/q10_tagger/vocabulary.pickle \
--path_out ckpt/q10_tagger/pred.pkl

p q_predict.py score_preds \
--path_pred ckpt/q10_tagger/pred.pkl \
--path_gold data/q10/test.json \
--path_vocab ckpt/q10_triplet/vocabulary.pickle

{
  "scorer": "QuintupletScorer",
  "num_correct": 1762,
  "num_pred": 2560,
  "num_gold": 2595,
  "precision": 0.68828125,
  "recall": 0.6789980732177264,
  "f1": 0.68360814742968
}

p q_predict.py prepare_tagger_pred_inputs \
--path_triplets ckpt/q30_triplet/pred.pkl \
--path_vocab ckpt/q30_triplet/vocabulary.pickle \
--path_out ckpt/q30_triplet/tagger_in.json \
--label_file data/q30/label.json

p q_predict.py run_eval ckpt/q30_tagger/best_model ckpt/q30_tagger/dataset.pickle \
--data_split dummy \
--task tagger \
--path_in ckpt/q30_triplet/tagger_in.json

p q_predict.py merge_pipeline_preds \
--path_triplets ckpt/q30_triplet/pred.pkl \
--path_tags ckpt/q30_tagger/raw_pred.pkl \
--path_vocab_triplets ckpt/q30_triplet/vocabulary.pickle \
--path_vocab_tags ckpt/q30_tagger/vocabulary.pickle \
--path_out ckpt/q30_tagger/pred.pkl

p q_predict.py score_preds \
--path_pred ckpt/q30_tagger/pred.pkl \
--path_gold data/q30/test.json \
--path_vocab ckpt/q30_triplet/vocabulary.pickle

{
  "scorer": "QuintupletScorer",
  "num_correct": 1700,
  "num_pred": 2355,
  "num_gold": 2302,
  "precision": 0.721868365180467,
  "recall": 0.738488271068636,
  "f1": 0.7300837449001503
}

"""

if __name__ == "__main__":
    Fire()
