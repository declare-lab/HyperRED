import json
import os
from pathlib import Path
from typing import List

from fire import Fire

from data_process import Sentence, process, Data
from scoring import EntityScorer, QuintupletScorer, StrictScorer
from training import run_eval, score_preds

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
    text_to_i = {s.text: i for i, s in enumerate(sents)}
    assert all(sep not in text for text in text_to_i.keys())

    for s in s_tags:
        text, head, relation, tail = s.text.split(sep)
        if text in text_to_i.keys():
            i = text_to_i[text]
            relations = sents[i].relations
            spans = set(e.span for e in sents[i].entities)

            for r in relations:
                r_head = " ".join(s.tokens[slice(*r.head)])
                r_tail = " ".join(s.tokens[slice(*r.tail)])
                if (r_head, r_tail, r.label) == (head, tail, relation):
                    for e in s.entities:
                        r.qualifiers.append(e)
                        if e.span not in spans:
                            spans.add(e.span)
                            sents[i].entities.append(e)
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
    temp_tags_in = str(Path(dir_tags) / temp_tags_in)
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
    Data(sents=preds).save(temp_tags)

    with open(Path(dir_data) / f"{data_split}.json") as f:
        sents = [Sentence(**json.loads(line)) for line in f]

    results = {}
    for scorer in [EntityScorer(), StrictScorer(), QuintupletScorer()]:
        results[scorer.name] = scorer.run(preds, sents)
    print(json.dumps(results, indent=2))


"""

p prediction.py eval_pipeline \
--dir_triplets ckpt/triplet_distilbert_seed_0/ \
--dir_tags ckpt/tags_distilbert_seed_0/ \
--dir_data data/processed \
--path_label_tags data/processed_tags/label.json \
--data_split test

"precision": 0.6837302470509682,
"recall": 0.5740982993832928,
"f1": 0.6241365298659082

################################################################################
Triplet Scores

p prediction.py score_preds ckpt/q10_pair2_no_value_prune_20_seed_0/test.json data/q10/test.json
"precision": 0.7252410166520596,
"recall": 0.6974294142435735,  
"f1": 0.7110633727175081

p prediction.py score_preds ckpt/q10_tags_distilbert_seed_0/pred.json data/q10/test.json
"precision": 0.7587951807228915,
"recall": 0.6635061104087653,
"f1": 0.7079586330935252

p prediction.py score_preds data/q10/gen_pred.json data/q10/test.json
"precision": 0.6971830985915493,             
"recall": 0.6466498103666245,
"f1": 0.6709663314385658

################################################################################
Model speed comparison

p prediction.py run_eval \
ckpt/q10_pair2_no_value_prune_20_seed_0/best_model \
ckpt/q10_pair2_no_value_prune_20_seed_0/dataset.pickle \
test

Cube: 25s for 4k samples, 6.6GB

p prediction.py run_eval \
ckpt/q10_triplet_distilbert_seed_0/best_model \
ckpt/q10_triplet_distilbert_seed_0/dataset.pickle \
--task triplet \
test

Triplet: 18s for 4k samples, 3.7GB

p prediction.py run_eval \
ckpt/q10_tags_distilbert_seed_0/best_model \
ckpt/q10_tags_distilbert_seed_0/dataset.pickle \
--task tagger \
test

Tagger: 4s for 4k samples, 1.8GB
Generative: 107s for 4k samples, 3.9GB

################################################################################
Eval pipeline base

p prediction.py eval_pipeline \
--dir_triplets ckpt/q10_triplet_seed_0/ \
--dir_tags ckpt/q10_tags_seed_0/ \
--dir_data data/q10 \
--path_label_tags data/q10_tags/label.json \
--data_split test

p prediction.py score_preds \
ckpt/q10_pair2_no_value_prune_20_seed_0/test.json \
data/q10/test.json

p prediction.py score_preds data/q10/gen_pred.json data/q10/test.json
p prediction.py score_preds data/q10/gen_1.json data/q10/test.json
p prediction.py score_preds data/q10/gen_2.json data/q10/test.json
p prediction.py score_preds data/q10/gen_3.json data/q10/test.json
p prediction.py score_preds data/q10/gen_4.json data/q10/test.json

p prediction.py eval_pipeline \
--dir_triplets ckpt/q10_triplet_large_seed_4/ \
--dir_tags ckpt/q10_tags_large_seed_4/ \
--dir_data data/q10 \
--path_label_tags data/q10_tags/label.json \
--data_split test

"""

if __name__ == "__main__":
    Fire()
