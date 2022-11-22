import json
import logging
import pickle
import random
from abc import ABC, abstractclassmethod
from collections import defaultdict
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class Field(ABC):
    """Abstract class `Field` define one indexing method,
    genenrate counter from raw text data and index token in raw text data

    Arguments:
        ABC {ABC} -- abstract base class
    """

    @abstractclassmethod
    def count_vocab_items(self, counter, sentences):
        """This function constructs counter using each sentence content,
        prepare for vocabulary

        Arguments:
            counter {dict} -- element count dict
            sentences {list} -- text data
        """

        raise NotImplementedError

    @abstractclassmethod
    def index(self, instance, voacb, sentences):
        """This function constrcuts instance using sentences and vocabulary,
        each namespace is a mappping method using different type data

        Arguments:
            instance {dict} -- collections of various fields
            voacb {dict} -- vocabulary
            sentences {list} -- text data
        """

        raise NotImplementedError


class TokenField(Field):
    """Token field: regard sentence as token list"""

    def __init__(self, namespace, vocab_namespace, source_key, is_counting=True):
        """This function sets namesapce of field, vocab namespace for indexing, dataset source key

        Arguments:
            namespace {str} -- namesapce of field, counter namespace if constructing a counter
            vocab_namespace {str} -- vocab namespace for indexing
            source_key {str} -- indicate key in text data

        Keyword Arguments:
            is_counting {bool} -- decide constructing a counter or not (default: {True})
        """

        super().__init__()
        self.namespace = str(namespace)
        self.counter_namespace = str(namespace)
        self.vocab_namespace = str(vocab_namespace)
        self.source_key = str(source_key)
        self.is_counting = is_counting

    def count_vocab_items(self, counter, sentences):
        """This function counts tokens in sentences,
        then update counter

        Arguments:
            counter {dict} -- counter
            sentences {list} -- text content after preprocessing
        """

        if self.is_counting:
            for sentence in sentences:
                for token in sentence[self.source_key]:
                    counter[self.counter_namespace][str(token)] += 1

            logger.info(
                "Count sentences {} to update counter namespace {} successfully.".format(
                    self.source_key, self.counter_namespace
                )
            )

    def index(self, instance, vocab, sentences):
        """This function indexed token using vocabulary,
        then update instance

        Arguments:
            instance {dict} -- numerical represenration of text data
            vocab {Vocabulary} -- vocabulary
            sentences {list} -- text content after preprocessing
        """

        for sentence in sentences:
            instance[self.namespace].append(
                [
                    vocab.get_token_index(token, self.vocab_namespace)
                    for token in sentence[self.source_key]
                ]
            )

        logger.info(
            "Index sentences {} to construct instance namespace {} successfully.".format(
                self.source_key, self.namespace
            )
        )


class RawTokenField(Field):
    """This Class preserves raw text of tokens"""

    def __init__(self, namespace, source_key):
        """This function sets namesapce of field, dataset source key

        Arguments:
            namespace {str} -- namesapce of field
            source_key {str} -- indicate key in text data
        """

        super().__init__()
        self.namespace = str(namespace)
        self.source_key = str(source_key)

    def count_vocab_items(self, counter, sentences):
        """`RawTokenField` doesn't update counter

        Arguments:
            counter {dict} -- counter
            sentences {list} -- text content after preprocessing
        """

        pass

    def index(self, instance, vocab, sentences):
        """This function doesn't use vocabulary,
        perserve raw text of sentences(tokens)

        Arguments:
            instance {dict} -- numerical represenration of text data
            vocab {Vocabulary} -- vocabulary
            sentences {list} -- text content after preprocessing
        """

        for sentence in sentences:
            instance[self.namespace].append(
                [token for token in sentence[self.source_key]]
            )

        logger.info(
            "Index sentences {} to construct instance namespace {} successfully.".format(
                self.source_key, self.namespace
            )
        )


class MapTokenField(Field):
    """Map token field: preocess maping tokens"""

    def __init__(self, namespace, vocab_namespace, source_key, is_counting=True):
        """This function sets namesapce of field, vocab namespace for indexing, dataset source key

        Arguments:
            namespace {str} -- namesapce of field, counter namespace if constructing a counter
            vocab_namespace {str} -- vocab namespace for indexing
            source_key {str} -- indicate key in text data

        Keyword Arguments:
            is_counting {bool} -- decide constructing a counter or not (default: {True})
        """

        super().__init__()
        self.namespace = str(namespace)
        self.counter_namespace = str(namespace)
        self.vocab_namespace = str(vocab_namespace)
        self.source_key = str(source_key)
        self.is_counting = is_counting

    def count_vocab_items(self, counter, sentences):
        """This function counts dict's values in sentences,
        then update counter, each sentence is a dict

        Arguments:
            counter {dict} -- counter
            sentences {list} -- text content after preprocessing, list of dict
        """

        if self.is_counting:
            for sentence in sentences:
                for value in sentence[self.source_key].values():
                    counter[self.counter_namespace][str(value)] += 1

            logger.info(
                "Count sentences {} to update counter namespace {} successfully.".format(
                    self.source_key, self.counter_namespace
                )
            )

    def index(self, instance, vocab, sentences):
        """This function indexes token using vocabulary, then update instance

        Arguments:
            instance {dict} -- numerical represenration of text data
            vocab {Vocabulary} -- vocabulary
            sentences {list} -- text content after preprocessing
        """

        for sentence in sentences:
            instance[self.namespace].append(
                {
                    key: vocab.get_token_index(value, self.vocab_namespace)
                    for key, value in sentence[self.source_key].items()
                }
            )

        logger.info(
            "Index sentences {} to construct instance namespace {} successfully.".format(
                self.source_key, self.namespace
            )
        )


class Instance:
    """`Instance` is the collection of multiple `Field`"""

    def __init__(self, fields):
        """This function initializes instance

        Arguments:
            fields {list} -- field list
        """

        self.fields = list(fields)
        self.instance = {}
        for field in self.fields:
            self.instance[field.namespace] = []
        self.vocab_dict = {}
        self.vocab_index()

    def __getitem__(self, namespace):
        if namespace not in self.instance:
            logger.error("can not find the namespace {} in instance.".format(namespace))
            raise RuntimeError(
                "can not find the namespace {} in instance.".format(namespace)
            )
        else:
            self.instance.get(namespace, None)

    def __iter__(self):
        return iter(self.fields)

    def __len__(self):
        return len(self.fields)

    def add_fields(self, fields):
        """This function adds fields to instance

        Arguments:
            field {Field} -- field list
        """

        for field in fields:
            if field.namesapce not in self.instance:
                self.fields.append(field)
                self.instance[field.namesapce] = []
            else:
                logger.warning("Field {} has been added before.".format(field.name))

        self.vocab_index()

    def count_vocab_items(self, counter, sentences):
        """This funtion constructs multiple namespace in counter

        Arguments:
            counter {dict} -- counter
            sentences {list} -- text content after preprocessing
        """

        for field in self.fields:
            field.count_vocab_items(counter, sentences)

    def index(self, vocab, sentences):
        """This funtion indexes token using vocabulary,
        then update instance

        Arguments:
            vocab {Vocabulary} -- vocabulary
            sentences {list} -- text content after preprocessing
        """

        for field in self.fields:
            field.index(self.instance, vocab, sentences)

    def get_instance(self):
        """This function get instance

        Returns:
            dict -- instance
        """

        return self.instance

    def get_size(self):
        """This funtion gets the size of instance

        Returns:
            int -- instance size
        """

        return len(self.instance[self.fields[0].namespace])

    def vocab_index(self):
        """This function constructs vocabulary dict of fields"""

        for field in self.fields:
            if hasattr(field, "vocab_namespace"):
                self.vocab_dict[field.namespace] = field.vocab_namespace

    def get_vocab_dict(self):
        """This function gets the vocab dict of instance

        Returns:
            dict -- vocab dict
        """

        return self.vocab_dict


class DataReader:
    """Define text data reader and preprocess data for entity relation
    joint decoding on ACE dataset.
    """

    def __init__(self, file_path, is_test=False, max_len=dict()):
        """This function defines file path and some settings

        Arguments:
            file_path {str} -- file path

        Keyword Arguments:
            is_test {bool} -- indicate training or testing (default: {False})
            max_len {dict} -- max length for some namespace (default: {dict()})
        """

        self.file_path = file_path
        self.is_test = is_test
        self.max_len = dict(max_len)
        self.seq_lens = defaultdict(list)

    def __iter__(self):
        """Generator function"""

        with open(self.file_path, "r") as fin:
            for line in fin:
                line = json.loads(line)
                sentence = {}

                state, results = self.get_tokens(line)
                self.seq_lens["tokens"].append(len(results["tokens"]))
                if not state or (
                    "tokens" in self.max_len
                    and len(results["tokens"]) > self.max_len["tokens"]
                    and not self.is_test
                ):
                    if not self.is_test:
                        continue
                sentence.update(results)

                state, results = self.get_wordpiece_tokens(line)
                self.seq_lens["wordpiece_tokens"].append(
                    len(results["wordpiece_tokens"])
                )
                if not state or (
                    "wordpiece_tokens" in self.max_len
                    and len(results["wordpiece_tokens"])
                    > self.max_len["wordpiece_tokens"]
                ):
                    if not self.is_test:
                        continue
                sentence.update(results)

                line["text"] = " ".join(line["tokens"])
                line["articleId"], line["sentId"] = line["text"], line["text"]
                if len(sentence["tokens"]) != len(sentence["wordpiece_tokens_index"]):
                    logger.error(
                        "article id: {} sentence id: {} wordpiece_tokens_index length is not equal to tokens.".format(
                            line["articleId"], line["sentId"]
                        )
                    )
                    continue

                if len(sentence["wordpiece_tokens"]) != len(
                    sentence["wordpiece_segment_ids"]
                ):
                    logger.error(
                        "article id: {} sentence id: {} wordpiece_tokens length is not equal to wordpiece_segment_ids.".format(
                            line["articleId"], line["sentId"]
                        )
                    )
                    continue

                state, results = self.get_entity_relation_label(
                    line, len(sentence["tokens"])
                )
                if results is None:
                    continue
                for key, result in results.items():
                    self.seq_lens[key].append(len(result))
                    if key in self.max_len and len(result) > self.max_len[key]:
                        state = False
                if not state:
                    continue
                sentence.update(results)

                yield sentence

    def get_tokens(self, line):
        """This function splits text into tokens

        Arguments:
            line {dict} -- text

        Returns:
            bool -- execute state
            dict -- results: tokens
        """

        results = {}
        line["text"] = " ".join(line["tokens"])
        line["articleId"], line["sentId"] = line["text"], line["text"]

        if "text" not in line:
            logger.error(
                "article id: {} sentence id: {} doesn't contain 'text'.".format(
                    line["articleId"], line["sentId"]
                )
            )
            return False, results

        results["text"] = line["text"]

        if "tokens" in line:
            results["tokens"] = line["tokens"]
        else:
            results["tokens"] = line["text"].strip().split(" ")

        return True, results

    def get_wordpiece_tokens(self, line):
        """This function splits wordpiece text into wordpiece tokens

        Arguments:
            line {dict} -- text

        Returns:
            bool -- execute state
            dict -- results: tokens
        """

        results = {}
        line["text"] = " ".join(line["tokens"])
        line["articleId"], line["sentId"] = line["text"], line["text"]

        if (
            "wordpieceSentText" not in line
            or "wordpieceTokensIndex" not in line
            or "wordpieceSegmentIds" not in line
        ):
            logger.error(
                "article id: {} sentence id: {} doesn't contain 'wordpieceSentText' or 'wordpieceTokensIndex' or 'wordpieceSegmentIds'.".format(
                    line["articleId"], line["sentId"]
                )
            )
            return False, results

        wordpiece_tokens = line["wordpieceSentText"].strip().split(" ")
        results["wordpiece_tokens"] = wordpiece_tokens
        results["wordpiece_tokens_index"] = [
            span[0] for span in line["wordpieceTokensIndex"]
        ]
        results["wordpiece_segment_ids"] = list(line["wordpieceSegmentIds"])

        return True, results

    def get_entity_relation_label(self, line, sentence_length):
        """This function constructs mapping relation from span to entity label
        and span pair to relation label, and joint entity relation label matrix.

        Arguments:
            line {dict} -- text
            sentence_length {int} -- sentence length

        Returns:
            bool -- execute state
            dict -- ent2rel: entity span mapping to entity label,
            joint_label_matrix: joint entity relation label matrix
        """

        results = {}
        line["text"] = " ".join(line["tokens"])
        line["articleId"], line["sentId"] = line["text"], line["text"]

        if "entities" not in line:
            logger.error(
                "article id: {} sentence id: {} doesn't contain 'entities'.".format(
                    line["articleId"], line["sentId"]
                )
            )
            return False, results

        if "relations" not in line:
            logger.error(
                "article id: {} sentence id: {} doesn't contain 'relations'.".format(
                    line["articleId"], line["sentId"]
                )
            )
            return False, results

        if "jointLabelMatrix" not in line:
            logger.error(
                "article id: {} sentence id: {} doesn't contain 'jointLabelMatrix'.".format(
                    line["articleId"], line["sentId"]
                )
            )
            return False, results

        results["joint_label_matrix"] = line["jointLabelMatrix"]
        results["quintuplet_shape"] = line["quintupletMatrix"]["shape"]
        results["quintuplet_entries"] = line["quintupletMatrix"]["entries"]

        return True, results

    def get_seq_lens(self):
        return self.seq_lens


class Dataset:
    """This class constructs dataset for multiple date file"""

    def __init__(self, name, instance_dict=dict()):
        """This function initializes a dataset,
        define dataset name, this dataset contains multiple readers, as datafiles.

        Arguments:
            name {str} -- dataset name

        Keyword Arguments:
            instance_dict {dict} -- instance settings (default: {dict()})
        """

        self.dataset_name = name
        self.datasets = dict()
        self.instance_dict = dict(instance_dict)

    def add_instance(self, name, instance, reader, is_count=False, is_train=False):
        """This function adds a instance to dataset

        Arguments:
            name {str} -- intance name
            instance {Instance} -- instance
            reader {DatasetReader} -- reader correspond to instance

        Keyword Arguments:
            is_count {bool} -- instance paticipates in counting or not (default: {False})
            is_train {bool} -- instance is training data or not (default: {False})
        """

        self.instance_dict[name] = {
            "instance": instance,
            "reader": reader,
            "is_count": is_count,
            "is_train": is_train,
        }

    def build_dataset(
        self,
        vocab,
        counter=None,
        min_count=dict(),
        pretrained_vocab=None,
        intersection_namespace=dict(),
        no_pad_namespace=list(),
        no_unk_namespace=list(),
        contain_pad_namespace=dict(),
        contain_unk_namespace=dict(),
        tokens_to_add=None,
    ):
        """This function bulids dataset

        Arguments:
            vocab {Vocabulary} -- vocabulary

        Keyword Arguments:
            counter {dict} -- counter (default: {None})
            min_count {dict} -- min count for each namespace (default: {dict()})
            pretrained_vocab {dict} -- pretrained vocabulary (default: {None})
            intersection_namespace {dict} -- intersection vocabulary namespace correspond to
            pretrained vocabulary in case of too large pretrained vocabulary (default: {dict()})
            no_pad_namespace {list} -- no padding vocabulary namespace (default: {list()})
            no_unk_namespace {list} -- no unknown vocabulary namespace (default: {list()})
            contain_pad_namespace {dict} -- contain padding token vocabulary namespace (default: {dict()})
            contain_unk_namespace {dict} -- contain unknown token vocabulary namespace (default: {dict()})
            tokens_to_add {dict} -- tokens need to be added to vocabulary (default: {None})
        """

        # construct counter
        if counter is not None:
            for instance_name, instance_settting in self.instance_dict.items():
                if instance_settting["is_count"]:
                    instance_settting["instance"].count_vocab_items(
                        counter, instance_settting["reader"]
                    )

            # construct vocabulary from counter
            vocab.extend_from_counter(
                counter,
                min_count,
                no_pad_namespace,
                no_unk_namespace,
                contain_pad_namespace,
                contain_unk_namespace,
            )

        # add extra tokens, this operation should be executeed before adding pretrained_vocab
        if tokens_to_add is not None:
            for namespace, tokens in tokens_to_add.items():
                vocab.add_tokens_to_namespace(tokens, namespace)

        # construct vocabulary from pretained vocabulary
        if pretrained_vocab is not None:
            vocab.extend_from_pretrained_vocab(
                pretrained_vocab,
                intersection_namespace,
                no_pad_namespace,
                no_unk_namespace,
                contain_pad_namespace,
                contain_unk_namespace,
            )

        self.vocab = vocab

        for instance_name in self.instance_dict.keys():
            self.process_instance(instance_name)

    def process_instance(self, instance_name: str):
        instance_settting = self.instance_dict[instance_name]
        instance_settting["instance"].index(self.vocab, instance_settting["reader"])
        self.datasets[instance_name] = instance_settting["instance"].get_instance()
        self.instance_dict[instance_name]["size"] = instance_settting[
            "instance"
        ].get_size()
        self.instance_dict[instance_name]["vocab_dict"] = instance_settting[
            "instance"
        ].get_vocab_dict()

        logger.info(
            "{} dataset size: {}.".format(
                instance_name, self.instance_dict[instance_name]["size"]
            )
        )
        for key, seq_len in instance_settting["reader"].get_seq_lens().items():
            logger.info(
                "{} dataset's {}: max_len={}, min_len={}.".format(
                    instance_name, key, max(seq_len), min(seq_len)
                )
            )

    def make_quintuplet_batch(
        self, dataset: Dict[str, list], sorted_ids: List[int]
    ) -> dict:
        entries = [dataset["quintuplet_entries"][i] for i in sorted_ids]
        lengths = [dataset["quintuplet_shape"][i][0] for i in sorted_ids]
        size = max(lengths)
        matrix = np.zeros((len(sorted_ids), size, size, size))
        mask = np.zeros((len(sorted_ids), size, size, size))

        for index, lst in enumerate(entries):
            for i, j, k, value in lst:
                matrix[index, i, j, k] = value
            num = lengths[index]
            mask[index, :num, :num, :num] = 1

        return {"quintuplet_matrix": matrix, "quintuplet_matrix_mask": mask > 0}

    def get_batch(self, instance_name, batch_size, sort_namespace=None):
        """get_batch gets batch data and padding

        Arguments:
            instance_name {str} -- instance name
            batch_size {int} -- batch size

        Keyword Arguments:
            sort_namespace {str} -- sort samples key, meanwhile calculate sequence length if not None, while keep None means that no sorting (default: {None})

        Yields:
            int -- epoch
            dict -- batch data
        """

        if instance_name not in self.instance_dict:
            logger.error(
                "can not find instance name {} in datasets.".format(instance_name)
            )
            return

        dataset = self.datasets[instance_name]

        if sort_namespace is not None and sort_namespace not in dataset:
            logger.error(
                "can not find sort namespace {} in datasets instance {}.".format(
                    sort_namespace, instance_name
                )
            )

        size = self.instance_dict[instance_name]["size"]
        vocab_dict = self.instance_dict[instance_name]["vocab_dict"]
        ids = list(range(size))
        if self.instance_dict[instance_name]["is_train"]:
            random.shuffle(ids)
        epoch = 1
        cur = 0

        while True:
            if cur >= size:
                epoch += 1
                if not self.instance_dict[instance_name]["is_train"] and epoch > 1:
                    break
                random.shuffle(ids)
                cur = 0

            sample_ids = ids[cur : cur + batch_size]
            cur += batch_size

            if sort_namespace is not None:
                sample_ids = [
                    (idx, len(dataset[sort_namespace][idx])) for idx in sample_ids
                ]
                sample_ids = sorted(sample_ids, key=lambda x: x[1], reverse=True)
                sorted_ids = [idx for idx, _ in sample_ids]
            else:
                sorted_ids = sample_ids

            batch = {}

            q_info = self.make_quintuplet_batch(dataset, sorted_ids)
            batch.update(**q_info)
            for namespace in dataset:
                if "quintuplet" in namespace:
                    continue

                batch[namespace] = []

                if namespace in self.wo_padding_namespace:
                    for id in sorted_ids:
                        batch[namespace].append(dataset[namespace][id])
                else:
                    if namespace in vocab_dict:
                        padding_idx = self.vocab.get_padding_index(
                            vocab_dict[namespace]
                        )
                    else:
                        padding_idx = 0

                    batch_namespace_len = [
                        len(dataset[namespace][id]) for id in sorted_ids
                    ]
                    max_namespace_len = max(batch_namespace_len)
                    batch[namespace + "_lens"] = batch_namespace_len
                    batch[namespace + "_mask"] = []

                    if isinstance(dataset[namespace][0][0], list):
                        max_char_len = 0
                        for id in sorted_ids:
                            max_char_len = max(
                                max_char_len,
                                max(len(item) for item in dataset[namespace][id]),
                            )
                        for id in sorted_ids:
                            padding_sent = []
                            mask = []
                            for item in dataset[namespace][id]:
                                padding_sent.append(
                                    item + [padding_idx] * (max_char_len - len(item))
                                )
                                mask.append(
                                    [1] * len(item) + [0] * (max_char_len - len(item))
                                )
                            padding_sent = padding_sent + [
                                [padding_idx] * max_char_len
                            ] * (max_namespace_len - len(dataset[namespace][id]))
                            mask = mask + [[0] * max_char_len] * (
                                max_namespace_len - len(dataset[namespace][id])
                            )
                            batch[namespace].append(padding_sent)
                            batch[namespace + "_mask"].append(mask)
                    else:
                        for id in sorted_ids:
                            batch[namespace].append(
                                dataset[namespace][id]
                                + [padding_idx]
                                * (max_namespace_len - len(dataset[namespace][id]))
                            )
                            batch[namespace + "_mask"].append(
                                [1] * len(dataset[namespace][id])
                                + [0]
                                * (max_namespace_len - len(dataset[namespace][id]))
                            )

            yield epoch, batch

    def get_dataset_size(self, instance_name):
        """This function gets dataset size

        Arguments:
            instance_name {str} -- instance name

        Returns:
            int -- dataset size
        """

        return self.instance_dict[instance_name]["size"]

    def set_wo_padding_namespace(self, wo_padding_namespace):
        """set_wo_padding_namespace sets without paddding namespace

        Args:
            wo_padding_namespace (list): without padding namespace
        """

        self.wo_padding_namespace = wo_padding_namespace

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
