import json
import random
from ast import literal_eval
from collections import Counter
from typing import List, Tuple

from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

Span = Tuple[int, int]


class Entity(BaseModel):
    emId: str
    text: str
    offset: Span  # Token spans, start inclusive, end exclusive
    label: str

    def as_tuple(self) -> Tuple[int, int, str]:
        return self.offset[0], self.offset[1], self.label


class Relation(BaseModel):
    em1Id: str
    em1Text: str
    em2Id: str
    em2Text: str
    label: str


class Qualifier(BaseModel):
    em1Id: str
    em2Id: str
    em3Id: str
    label: str

    def parse_span(self, i: str) -> Tuple[int, int]:
        x = literal_eval(i)
        if not isinstance(x[0], int):
            return x[0]
        return x

    def parse_relation(self, triplets: List[Relation]) -> str:
        label = ""
        for r in triplets:
            if self.parse_span(r.em1Id) == self.parse_span(self.em1Id):
                if self.parse_span(r.em2Id) == self.parse_span(self.em2Id):
                    label = r.label
        return label

    def as_texts(
        self, tokens: List[str], triplets: List[Relation]
    ) -> Tuple[str, str, str, str, str]:
        head = " ".join(tokens[slice(*self.parse_span(self.em1Id))])
        tail = " ".join(tokens[slice(*self.parse_span(self.em2Id))])
        value = " ".join(tokens[slice(*self.parse_span(self.em3Id))])
        relation = self.parse_relation(triplets)
        return (head, relation, tail, self.label, value)


class Sentence(BaseModel):
    articleId: str
    sentId: int
    sentText: str
    entityMentions: List[Entity]
    relationMentions: List[Relation]
    qualifierMentions: List[Qualifier]

    @property
    def tokens(self) -> List[str]:
        return self.sentText.split(" ")


def load_sents(path: str) -> List[Sentence]:
    with open(path) as f:
        return [Sentence(**json.loads(line)) for line in tqdm(f.readlines(), desc=path)]


def save_sents(sents: List[Sentence], path: str):
    with open(path, "w") as f:
        for s in sents:
            f.write(s.json() + "\n")


def test_data(path: str):
    sents = load_sents(path)
    qualifiers = [q.label for s in sents for q in s.qualifierMentions]
    relations = [r.label for s in sents for r in s.relationMentions]

    print("Relations", len(set(relations)), "#" * 80)
    print(Counter(relations))
    print("Qualifiers", len(set(qualifiers)), "#" * 80)
    print(Counter(qualifiers))

    print("Samples", "#" * 80)
    random.seed(0)
    for s in random.sample(sents, k=3):
        print(s.tokens)
        for q in s.qualifierMentions:
            print(q.as_texts(s.tokens, s.relationMentions))
        print()


if __name__ == "__main__":
    Fire()
