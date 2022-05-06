import json
from collections import Counter
from pathlib import Path
from typing import Counter, List, Tuple

import fire
from pydantic import BaseModel
from transformers import AutoTokenizer


class FlatQuintuplet(BaseModel):
    tokens: List[str]
    head: Tuple[int, int]
    tail: Tuple[int, int]
    value: Tuple[int, int]
    relation: str
    qualifier: str


def load_quintuplets(path: str) -> List[FlatQuintuplet]:
    with open(path) as f:
        return [FlatQuintuplet(**json.loads(line)) for line in f]


def make_label_file(
    data_pattern: str = "../quintuplet/outputs/data/flat/*.json",
    path_out: str = "data/quintuplet/ent_rel_file.json",
):
    paths = sorted(Path().glob(data_pattern))
    quintuplets = [q for p in paths for q in load_quintuplets(str(p))]
    relations = [q.relation for q in quintuplets]
    qualifiers = [q.qualifier for q in quintuplets]
    labels = ["None", "Entity"]
    labels.extend(sorted(set(relations + qualifiers)))
    label_map = {name: i for i, name in enumerate(labels)}

    entity_ids = [label_map["Entity"]]
    relation_ids = sorted(set(label_map[name] for name in relations))
    qualifier_ids = sorted(set(label_map[name] for name in qualifiers))

    num_none = 0
    num_entity = len(quintuplets) * 3  # Three entities per quintuplet
    counter = Counter(relations + qualifiers)
    counts = [num_none, num_entity] + [
        counter[name] for name in sorted(set(relations + qualifiers))
    ]

    assert len(counts) == len(label_map.keys())
    info = dict(
        id=label_map,
        entity=entity_ids,
        relation=relation_ids,
        qualifier=qualifier_ids,
        symmetric=[],
        asymmetric=sorted(set(relation_ids + qualifier_ids)),
        count=counts,
    )
    Path(path_out).parent.mkdir(exist_ok=True, parents=True)
    with open(path_out, "w") as f:
        f.write(json.dumps(info, indent=2))


def add_cross_sentence(sentences, tokenizer, max_length=200):
    """add_cross_sentence add cross sentences with adding equal number of
    left and right context tokens.
    """

    new_sents = []
    sent_lens = []
    last_id = sentences[0]["sentId"] - 1
    article_id = sentences[0]["articleId"]

    for s in sentences:
        assert s["articleId"] == article_id
        assert s["sentId"] > last_id
        last_id = s["sentId"]
        tokens = s["sentText"].split(" ")
        sent_lens.append(len(tokens))

    cur_pos = 0
    for sent, sent_len in zip(sentences, sent_lens):
        cls = tokenizer.cls_token
        sep = tokenizer.sep_token

        wordpiece_tokens = [cls]
        wordpiece_tokens.append(sep)

        context_len = len(wordpiece_tokens)
        wordpiece_segment_ids = [0] * context_len

        wordpiece_tokens_index = []
        cur_index = len(wordpiece_tokens)
        for token in sent["sentText"].split(" "):
            tokenized_token = list(tokenizer.tokenize(token))
            wordpiece_tokens.extend(tokenized_token)
            wordpiece_tokens_index.append([cur_index, cur_index + len(tokenized_token)])
            cur_index += len(tokenized_token)
        wordpiece_tokens.append(sep)

        wordpiece_segment_ids += [1] * (len(wordpiece_tokens) - context_len)

        new_sent = {
            "articleId": sent["articleId"],
            "sentId": sent["sentId"],
            "sentText": sent["sentText"],
            "entityMentions": sent["entityMentions"],
            "relationMentions": sent["relationMentions"],
            "wordpieceSentText": " ".join(wordpiece_tokens),
            "wordpieceTokensIndex": wordpiece_tokens_index,
            "wordpieceSegmentIds": wordpiece_segment_ids,
        }
        new_sents.append(new_sent)

        cur_pos += sent_len

    return new_sents


def add_joint_label(sent, ent_rel_id):
    """add_joint_label add joint labels for sentences"""

    none_id = ent_rel_id["None"]
    sentence_length = len(sent["sentText"].split(" "))
    label_matrix = [
        [none_id for j in range(sentence_length)] for i in range(sentence_length)
    ]
    ent2offset = {}
    for ent in sent["entityMentions"]:
        ent2offset[ent["emId"]] = ent["offset"]
        for i in range(ent["offset"][0], ent["offset"][1]):
            for j in range(ent["offset"][0], ent["offset"][1]):
                label_matrix[i][j] = ent_rel_id[ent["label"]]
    for rel in sent["relationMentions"]:
        for i in range(ent2offset[rel["em1Id"]][0], ent2offset[rel["em1Id"]][1]):
            for j in range(ent2offset[rel["em2Id"]][0], ent2offset[rel["em2Id"]][1]):
                label_matrix[i][j] = ent_rel_id[rel["label"]]
    sent["jointLabelMatrix"] = label_matrix


def process(source_file, ent_rel_file, target_file, pretrained_model, max_length=200):
    auto_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    print("Load {} tokenizer successfully.".format(pretrained_model))

    ent_rel_id = json.load(open(ent_rel_file, "r", encoding="utf-8"))["id"]

    with open(source_file, "r", encoding="utf-8") as fin, open(
        target_file, "w", encoding="utf-8"
    ) as fout:
        sentences = []
        for line in fin:
            sent = json.loads(line.strip())

            if len(sentences) == 0 or sentences[0]["articleId"] == sent["articleId"]:
                sentences.append(sent)
            else:
                for new_sent in add_cross_sentence(
                    sentences, auto_tokenizer, max_length
                ):
                    add_joint_label(new_sent, ent_rel_id)
                    print(json.dumps(new_sent), file=fout)
                sentences = [sent]

        for new_sent in add_cross_sentence(sentences, auto_tokenizer, max_length):
            add_joint_label(new_sent, ent_rel_id)
            print(json.dumps(new_sent), file=fout)


if __name__ == "__main__":
    fire.Fire()