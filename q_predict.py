import json
import os
from pathlib import Path
from typing import List

from fire import Fire

from data.q_process import Qualifier, Sentence, process
from q_main import run_eval, score_preds
from scoring import QuintupletScorer

assert run_eval is not None
assert score_preds is not None


def merge_pipeline_preds(
    path_triplets: str,
    path_tags: str,
    sep: str = " | ",  # Refer to convert_sent_to_tags
) -> List[Sentence]:
    with open(path_triplets) as f:
        sents = [Sentence(**json.loads(line)) for line in f]
    with open(path_tags) as f:
        s_tags = [Sentence(**json.loads(line)) for line in f]
    text_to_i = {s.sentText: i for i, s in enumerate(sents)}
    assert all(sep not in text for text in text_to_i.keys())

    for s in s_tags:
        text, head, relation, tail = s.sentText.split(sep)
        if text in text_to_i.keys():
            i = text_to_i[text]
            mention_to_e = {e.text: e for e in sents[i].entityMentions}
            ids = set(e.emId for e in mention_to_e.values())

            for e in s.entityMentions:
                q = Qualifier(
                    em1Id=mention_to_e[head].emId,
                    em2Id=mention_to_e[tail].emId,
                    em3Id=e.emId,
                    label=e.label,
                )
                e.label = sents[i].entityMentions[0].label
                if e.emId not in ids:
                    sents[i].entityMentions.append(e)
                    ids.add(e.emId)
                sents[i].qualifierMentions.append(q)
        else:
            print(dict(unmatched=text))

    return sents


def eval_pipeline(
    dir_triplets: str,
    dir_tags: str,
    dir_data: str,
    path_label_tags: str,
    data_split: str,
    temp_tags_in: str = "temp_tags_in.json",
):
    temp_triplets = str(Path(dir_triplets) / f"{data_split}.json")
    process(
        source_file=temp_triplets,
        target_file=temp_tags_in,
        label_file=path_label_tags,
        mode="tags",
    )
    run_eval(
        path=str(Path(dir_tags) / "best_model"),
        path_data=str(Path(dir_tags) / "dataset.pickle"),
        data_split="dummy",
        task="tagger",
        path_in=temp_tags_in,
    )

    temp_tags = str(Path(dir_tags) / "pred.json")
    preds = merge_pipeline_preds(path_triplets=temp_triplets, path_tags=temp_tags)
    os.remove(temp_tags_in)
    os.remove(temp_tags)

    with open(Path(dir_data) / f"{data_split}.json") as f:
        sents = [Sentence(**json.loads(line)) for line in f]
    scorer = QuintupletScorer()
    results = scorer.run(preds, sents)
    print(json.dumps(results, indent=2))


"""
p q_predict.py eval_pipeline \
--dir_triplets ckpt/q10_triplet \
--dir_tags ckpt/q10_tags \
--dir_data data/q10 \
--path_label_tags data/q10_tags/label.json \
--data_split test

precision:   0.6987045549519432
recall:      0.644315992292871
f1:          0.6704089815557338

p q_predict.py eval_pipeline \
--dir_triplets ckpt/q10_triplet \
--dir_tags ckpt/q10_tags_freeze \
--dir_data data/q10 \
--path_label_tags data/q10_tags/label.json \
--data_split test

precision:   0.5771947527749748
recall:      0.44084778420038534
f1:          0.49989075813851863

p q_predict.py run_eval ckpt/double/q10_triplet/best_model ckpt/double/q10_triplet/dataset.pickle test --task triplet

"precision": 0.7602870813397129,
"recall": 0.6872837370242214,
"f1": 0.7219445706497047

p q_predict.py eval_pipeline \
--dir_triplets ckpt/double/q10_triplet \
--dir_tags ckpt/double/q10_tags \
--dir_data data/q10 \
--path_label_tags data/q10_tags/label.json \
--data_split test

"precision": 0.7016129032258065,
"recall": 0.6034682080924856,
"f1": 0.6488502175264139

"""

if __name__ == "__main__":
    Fire()
