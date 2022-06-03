import json
import logging
import os
import pickle
import random
from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.functional import Tensor
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from data.q_process import RawPred, Sentence, SparseCube
from inputs.dataset_readers.q_reader import ACEReaderForJointDecoding
from inputs.datasets.q_dataset import Dataset
from inputs.fields.map_token_field import MapTokenField
from inputs.fields.raw_token_field import RawTokenField
from inputs.fields.token_field import TokenField
from inputs.instance import Instance
from inputs.vocabulary import Vocabulary
from models.joint_decoding.joint_decoder import \
    EntRelJointDecoder as TripletModel
from models.joint_decoding.q_decoder import EntRelJointDecoder
from models.joint_decoding.q_tagger import EntRelJointDecoder as Tagger
from scoring import EntityScorer, QuintupletScorer, StrictScorer
from utils.new_argparse import ConfigurationParer
from utils.nn_utils import get_n_trainable_parameters

logger = logging.getLogger(__name__)


def load_model(task: str, path: str = "", **kwargs):
    model_class = dict(
        quintuplet=EntRelJointDecoder, tagger=Tagger, triplet=TripletModel
    )[task]
    if path:
        return model_class.load(path)
    else:
        return model_class(**kwargs)


def run_eval(
    path: str = "ckpt/quintuplet/best_model",
    path_data="ckpt/quintuplet/dataset.pickle",
    data_split: str = "dev",
    task: str = "quintuplet",
    path_in: str = "",
):
    model = load_model(task, path)
    dataset = Dataset.load(path_data)
    cfg = model.cfg
    evaluate(cfg, dataset, model, data_split, path_in=path_in)


def score_preds(path_pred: str, path_gold: str) -> dict:
    with open(path_pred) as f:
        preds = [Sentence(**json.loads(line)) for line in f]
    with open(path_gold) as f:
        sents = [Sentence(**json.loads(line)) for line in f]

    results = {}
    for scorer in [EntityScorer(), StrictScorer(), QuintupletScorer()]:
        results[scorer.name] = scorer.run(preds, sents)
    print(json.dumps(results, indent=2))
    return results


def prepare_inputs(batch_inputs, device):
    for k, v in batch_inputs.items():
        device_id = device if device > -1 else None
        if k in ["joint_label_matrix_mask", "quintuplet_matrix_mask"]:
            batch_inputs[k] = torch.tensor(v, dtype=torch.bool, device=device_id)
        if k in [
            "tokens",
            "joint_label_matrix",
            "quintuplet_matrix",
            "wordpiece_tokens",
            "wordpiece_tokens_index",
            "wordpiece_segment_ids",
        ]:
            batch_inputs[k] = torch.tensor(v, dtype=torch.long, device=device_id)

    return batch_inputs


class DoubleModel(nn.Module):
    def __init__(self, kwargs: dict, kwargs_2: dict):
        super().__init__()
        self.extractor = TripletModel(**kwargs)
        self.tagger = Tagger(**kwargs_2)
        self.tagger.embedding_model = self.extractor.embedding_model


def train(cfg, dataset, dataset_2, model):
    logger.info("Training starting...")

    for name, param in model.named_parameters():
        logger.info(
            "{!r}: size: {} requires_grad: {}.".format(
                name, param.size(), param.requires_grad
            )
        )

    logger.info(
        "Trainable parameters size: {}.".format(get_n_trainable_parameters(model))
    )

    parameters = [
        (name, param) for name, param in model.named_parameters() if param.requires_grad
    ]
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    bert_layer_lr = {}
    base_lr = cfg.bert_learning_rate
    for i in range(11, -1, -1):
        bert_layer_lr["." + str(i) + "."] = base_lr
        base_lr *= cfg.lr_decay_rate

    optimizer_grouped_parameters = []
    for name, param in parameters:
        params = {"params": [param], "lr": cfg.learning_rate}
        if any(item in name for item in no_decay):
            params["weight_decay_rate"] = 0.0
        else:
            if "bert" in name:
                params["weight_decay_rate"] = cfg.adam_bert_weight_decay_rate
            else:
                params["weight_decay_rate"] = cfg.adam_weight_decay_rate

        for bert_layer_name, lr in bert_layer_lr.items():
            if bert_layer_name in name:
                params["lr"] = lr
                break

        optimizer_grouped_parameters.append(params)

    optimizer = AdamW(
        optimizer_grouped_parameters,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        lr=cfg.learning_rate,
        eps=cfg.adam_epsilon,
        weight_decay=cfg.adam_weight_decay_rate,
        correct_bias=False,
    )
    assert optimizer.step is not None

    total_train_steps = (
        (
            dataset.get_dataset_size("train")
            + cfg.train_batch_size * cfg.gradient_accumulation_steps
            - 1
        )
        / (cfg.train_batch_size * cfg.gradient_accumulation_steps)
        * cfg.epochs
    ) * 2
    num_warmup_steps = int(cfg.warmup_rate * total_train_steps) + 1
    num_batches = cfg.epochs * dataset.get_dataset_size("train") // cfg.train_batch_size
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_train_steps,
    )
    assert scheduler is not None

    best_score = -1e9
    best_score_2 = -1e9
    accumulation_steps = 0
    model.zero_grad()
    seen_epochs = set()
    epoch_info = dict(epoch=0.0)

    for (epoch, batch), (_, batch_2) in tqdm(
        zip(
            dataset.get_batch("train", cfg.train_batch_size, None),
            dataset_2.get_batch("train", cfg.train_batch_size, None),
        ),
        total=num_batches,
    ):
        if epoch > cfg.epochs:
            break

        if epoch not in seen_epochs:
            seen_epochs.add(epoch)
            if accumulation_steps != 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            dev_loss, dev_score = evaluate(
                model.extractor.cfg, dataset, model.extractor, data_split="dev"
            )
            dev_loss_2, dev_score_2 = evaluate(
                model.tagger.cfg, dataset_2, model.tagger, data_split="dev"
            )
            epoch_info.update(
                dev_loss=dev_loss,
                dev_loss_2=dev_loss_2,
                score=dev_score,
                best_score=best_score,
                score_2=dev_score_2,
                best_score_2=best_score_2,
            )
            if dev_score > best_score:
                best_score = dev_score
                path = model.extractor.cfg.best_model_path
                logger.info(str(dict(save=path)))
                model.extractor.save(path)
            if dev_score_2 > best_score_2:
                best_score_2 = dev_score_2
                path = model.tagger.cfg.best_model_path
                logger.info(str(dict(save=path)))
                model.tagger.save(path)
            logger.info(str(epoch_info))

        model.train()
        batch["epoch"] = epoch - 1
        batch_2["epoch"] = epoch - 1
        outputs = model.extractor(prepare_inputs(batch, cfg.device))
        outputs_2 = model.tagger(prepare_inputs(batch_2, cfg.device))
        loss = outputs["loss"] + outputs_2["loss"]
        epoch_info["loss"] = loss
        epoch_info.update(epoch=epoch)

        if cfg.gradient_accumulation_steps > 1:
            loss /= cfg.gradient_accumulation_steps

        loss.backward()

        accumulation_steps = (accumulation_steps + 1) % cfg.gradient_accumulation_steps
        if accumulation_steps == 0:
            nn.utils.clip_grad.clip_grad_norm_(
                parameters=model.parameters(), max_norm=cfg.gradient_clipping
            )
            optimizer.step()
            scheduler.step()
            model.zero_grad()


def process_outputs(
    batch_inputs: Dict[str, Tensor], batch_outputs: Dict[str, Tensor]
) -> List[dict]:
    all_outputs = []
    for i in range(len(batch_inputs["tokens_lens"])):
        output = dict()
        for k in set(batch_inputs.keys()).union(batch_outputs.keys()):
            v = batch_inputs.get(k)
            if v is None:
                v = batch_outputs[k]

            if k in ["quintuplet_preds"]:
                output[k] = SparseCube.from_numpy(v[i].cpu().numpy()).dict()
            if k in ["tokens", "joint_label_matrix", "joint_label_preds"]:
                output[k] = v[i].cpu().numpy()
            if k in [
                "span2ent",
                "span2rel",
                "seq_len",
                "separate_positions",
                "all_separate_position_preds",
                "all_ent_preds",
                "all_rel_preds",
                "all_q_preds",
            ]:
                output[k] = v[i]
        all_outputs.append(output)
    return all_outputs


def evaluate(
    cfg: Namespace,
    dataset: Dataset,
    model: nn.Module,
    data_split: str,
    path_in: str = "",
):
    model.zero_grad()
    losses = []
    all_outputs = []

    if path_in:
        data_split = "pred"
        max_len = {
            "tokens": cfg.max_sent_len,
            "wordpiece_tokens": cfg.max_wordpiece_len,
        }
        reader = ACEReaderForJointDecoding(path_in, False, max_len)
        fields = dataset.instance_dict["test"]["instance"].fields
        instance = Instance(fields)
        dataset.add_instance(
            data_split, instance, reader, is_count=True, is_train=False
        )
        dataset.process_instance(data_split)

    num_batches = dataset.get_dataset_size(data_split) // cfg.test_batch_size
    for _, batch in tqdm(
        dataset.get_batch(data_split, cfg.test_batch_size, None), total=num_batches
    ):
        model.eval()
        with torch.no_grad():
            inputs = prepare_inputs(batch, cfg.device)
            outputs = model(inputs)
            losses.append(outputs["loss"].cpu().item())
            all_outputs.extend(process_outputs(inputs, outputs))

    # Save raw outputs
    path = Path(cfg.save_dir) / f"raw_{data_split}.pkl"
    print(dict(path=path))
    with open(path, "wb") as f:
        pickle.dump(all_outputs, f)

    # Save processed sents
    path = Path(cfg.save_dir) / f"{data_split}.json"
    print(dict(path=path))
    with open(path, "w") as f:
        for r in all_outputs:
            sent = RawPred(**r).as_sentence(model.vocab)
            f.write(sent.json() + "\n")

    mapping = dict(train=cfg.train_file, dev=cfg.dev_file, test=cfg.test_file)
    results = score_preds(path_pred=str(path), path_gold=path_in or mapping[data_split])

    score = dict(
        quintuplet=results["quintuplet"]["f1"],
        triplet=results["entity"]["f1"] + results["strict triplet"]["f1"],
        tagger=results["entity"]["f1"],
    )[cfg.task]
    return np.mean(losses), score


def build_data_vocab(cfg: Namespace) -> Tuple[Dataset, Vocabulary, dict]:
    # define fields
    tokens = TokenField("tokens", "tokens", "tokens", True)
    separate_positions = RawTokenField("separate_positions", "separate_positions")
    span2ent = MapTokenField("span2ent", "ent_rel_id", "span2ent", False)
    span2rel = MapTokenField("span2rel", "ent_rel_id", "span2rel", False)
    joint_label_matrix = RawTokenField("joint_label_matrix", "joint_label_matrix")
    quintuplet_shape = RawTokenField("quintuplet_shape", "quintuplet_shape")
    quintuplet_entries = RawTokenField("quintuplet_entries", "quintuplet_entries")
    wordpiece_tokens = TokenField(
        "wordpiece_tokens", "wordpiece", "wordpiece_tokens", False
    )
    wordpiece_tokens_index = RawTokenField(
        "wordpiece_tokens_index", "wordpiece_tokens_index"
    )
    wordpiece_segment_ids = RawTokenField(
        "wordpiece_segment_ids", "wordpiece_segment_ids"
    )
    fields = [tokens, separate_positions, span2ent, span2rel, joint_label_matrix]
    fields.extend([quintuplet_shape, quintuplet_entries])

    if cfg.embedding_model in ["bert", "pretrained"]:
        fields.extend([wordpiece_tokens, wordpiece_tokens_index, wordpiece_segment_ids])

    # define counter and vocabulary
    counter = defaultdict(lambda: defaultdict(int))
    vocab = Vocabulary()

    # define instance
    train_instance = Instance(fields)
    dev_instance = Instance(fields)
    test_instance = Instance(fields)

    # define dataset reader
    max_len = {"tokens": cfg.max_sent_len, "wordpiece_tokens": cfg.max_wordpiece_len}
    ent_rel_file = json.load(open(cfg.ent_rel_file, "r", encoding="utf-8"))
    pretrained_vocab = {"ent_rel_id": ent_rel_file["id"]}
    if cfg.embedding_model == "bert":
        tokenizer = BertTokenizer.from_pretrained(cfg.bert_model_name)
        logger.info("Load bert tokenizer successfully.")
        pretrained_vocab["wordpiece"] = tokenizer.get_vocab()
    elif cfg.embedding_model == "pretrained":
        tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name)
        logger.info("Load {} tokenizer successfully.".format(cfg.pretrained_model_name))
        pretrained_vocab["wordpiece"] = tokenizer.get_vocab()
    else:
        raise ValueError()

    ace_train_reader = ACEReaderForJointDecoding(cfg.train_file, False, max_len)
    ace_dev_reader = ACEReaderForJointDecoding(cfg.dev_file, False, max_len)
    ace_test_reader = ACEReaderForJointDecoding(cfg.test_file, False, max_len)

    # define dataset
    ace_dataset = Dataset("ACE2005")
    ace_dataset.add_instance(
        "train", train_instance, ace_train_reader, is_count=True, is_train=True
    )
    ace_dataset.add_instance(
        "dev", dev_instance, ace_dev_reader, is_count=True, is_train=False
    )
    ace_dataset.add_instance(
        "test", test_instance, ace_test_reader, is_count=True, is_train=False
    )

    min_count = {"tokens": 1}
    no_pad_namespace = ["ent_rel_id"]
    no_unk_namespace = ["ent_rel_id"]
    contain_pad_namespace = {"wordpiece": tokenizer.pad_token}
    contain_unk_namespace = {"wordpiece": tokenizer.unk_token}
    ace_dataset.build_dataset(
        vocab=vocab,
        counter=counter,
        min_count=min_count,
        pretrained_vocab=pretrained_vocab,
        no_pad_namespace=no_pad_namespace,
        no_unk_namespace=no_unk_namespace,
        contain_pad_namespace=contain_pad_namespace,
        contain_unk_namespace=contain_unk_namespace,
    )
    wo_padding_namespace = ["separate_positions", "span2ent", "span2rel"]
    ace_dataset.set_wo_padding_namespace(wo_padding_namespace=wo_padding_namespace)
    ace_dataset.save(os.path.join(cfg.save_dir, "dataset.pickle"))
    vocab.save(cfg.vocabulary_file)
    return ace_dataset, vocab, ent_rel_file


def make_config_2(cfg: Namespace) -> Namespace:
    # Refer to new_argparse.parse_args
    cfg = deepcopy(cfg)
    cfg.task = cfg.task_2
    cfg.data_dir = cfg.data_dir_2
    cfg.save_dir = cfg.save_dir_2
    cfg.train_file = str(Path(cfg.data_dir) / Path(cfg.train_file).name)
    cfg.dev_file = str(Path(cfg.data_dir) / Path(cfg.dev_file).name)
    cfg.test_file = str(Path(cfg.data_dir) / Path(cfg.test_file).name)
    cfg.ent_rel_file = str(Path(cfg.data_dir) / Path(cfg.ent_rel_file).name)
    cfg.best_model_path = os.path.join(cfg.save_dir, "best_model")
    cfg.last_model_path = os.path.join(cfg.save_dir, "last_model")
    cfg.vocabulary_file = os.path.join(cfg.save_dir, "vocabulary.pickle")
    cfg.model_checkpoints_dir = os.path.join(cfg.save_dir, "model_ckpts")
    Path(cfg.model_checkpoints_dir).mkdir(parents=True)
    return cfg


def main():
    # config settings
    parser = ConfigurationParer()
    parser.add_save_cfgs()
    parser.add_data_cfgs()
    parser.add_model_cfgs()
    parser.add_optimizer_cfgs()
    parser.add_run_cfgs()

    cfg = parser.parse_args()
    logger.info(parser.format_values())

    # set random seed
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.device > -1 and not torch.cuda.is_available():
        logger.error("config conflicts: no gpu available, use cpu for training.")
        cfg.device = -1
    if cfg.device > -1:
        torch.cuda.manual_seed(cfg.seed)

    cfg_2 = make_config_2(cfg)
    data, vocab, labels = build_data_vocab(cfg)
    data_2, vocab_2, labels_2 = build_data_vocab(cfg_2)

    # joint model
    model = DoubleModel(
        kwargs=dict(cfg=cfg, vocab=vocab, ent_rel_file=labels),
        kwargs_2=dict(cfg=cfg_2, vocab=vocab_2, ent_rel_file=labels_2),
    )
    if cfg.device > -1:
        model.cuda(device=cfg.device)

    train(cfg, dataset=data, dataset_2=data_2, model=model)


"""

p double_main.py \
--save_dir ckpt/double/q10_triplet \
--save_dir_2 ckpt/double/q10_tags \
--data_dir data/q10 \
--data_dir_2 data/q10_tags \
--task triplet \
--task_2 tagger \
--config_file q_config.yml

'epoch': 29,
'dev_loss': 0.0,
'dev_loss_2': 0.07230172460041263,
'score': 1.6062650083125096,
'best_score': 1.606962118870837,
'score_2': 0.8534278959810875,
'best_score_2': 0.8677685950413223,
'loss': tensor(0.5845, device='cuda:0', grad_fn=<AddBackward0>)

"""


if __name__ == "__main__":
    main()
