import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from inputs.dataset_readers.q_reader import ACEReaderForJointDecoding
from inputs.datasets.q_dataset import Dataset
from inputs.fields.map_token_field import MapTokenField
from inputs.fields.raw_token_field import RawTokenField
from inputs.fields.token_field import TokenField
from inputs.instance import Instance
from inputs.vocabulary import Vocabulary
from models.joint_decoding.q_decoder import EntRelJointDecoder
from utils.new_argparse import ConfigurationParer
from utils.nn_utils import get_n_trainable_parameters

logger = logging.getLogger(__name__)


def prepare_inputs(batch_inputs, device):
    batch_inputs["tokens"] = torch.LongTensor(batch_inputs["tokens"])
    batch_inputs["joint_label_matrix"] = torch.LongTensor(
        batch_inputs["joint_label_matrix"]
    )
    batch_inputs["joint_label_matrix_mask"] = torch.BoolTensor(
        batch_inputs["joint_label_matrix_mask"]
    )
    batch_inputs["quintuplet_matrix"] = torch.LongTensor(
        batch_inputs["quintuplet_matrix"]
    )
    batch_inputs["quintuplet_matrix_mask"] = torch.BoolTensor(
        batch_inputs["quintuplet_matrix_mask"]
    )
    batch_inputs["wordpiece_tokens"] = torch.LongTensor(
        batch_inputs["wordpiece_tokens"]
    )
    batch_inputs["wordpiece_tokens_index"] = torch.LongTensor(
        batch_inputs["wordpiece_tokens_index"]
    )
    batch_inputs["wordpiece_segment_ids"] = torch.LongTensor(
        batch_inputs["wordpiece_segment_ids"]
    )

    if device > -1:
        batch_inputs["tokens"] = batch_inputs["tokens"].cuda(
            device=device, non_blocking=True
        )
        batch_inputs["joint_label_matrix"] = batch_inputs["joint_label_matrix"].cuda(
            device=device, non_blocking=True
        )
        batch_inputs["joint_label_matrix_mask"] = batch_inputs[
            "joint_label_matrix_mask"
        ].cuda(device=device, non_blocking=True)
        batch_inputs["quintuplet_matrix"] = batch_inputs["quintuplet_matrix"].cuda(
            device=device, non_blocking=True
        )
        batch_inputs["quintuplet_matrix_mask"] = batch_inputs[
            "quintuplet_matrix_mask"
        ].cuda(device=device, non_blocking=True)
        batch_inputs["wordpiece_tokens"] = batch_inputs["wordpiece_tokens"].cuda(
            device=device, non_blocking=True
        )
        batch_inputs["wordpiece_tokens_index"] = batch_inputs[
            "wordpiece_tokens_index"
        ].cuda(device=device, non_blocking=True)
        batch_inputs["wordpiece_segment_ids"] = batch_inputs[
            "wordpiece_segment_ids"
        ].cuda(device=device, non_blocking=True)

    return batch_inputs


def train(cfg, dataset, model):
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
    )
    num_warmup_steps = int(cfg.warmup_rate * total_train_steps) + 1
    num_batches = cfg.epochs * dataset.get_dataset_size("train") // cfg.train_batch_size
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_train_steps,
    )
    assert scheduler is not None

    best_loss = 1e9
    accumulation_steps = 0
    model.zero_grad()
    seen_epochs = set()
    epoch_info = dict(epoch=0)

    for epoch, batch in tqdm(
        dataset.get_batch("train", cfg.train_batch_size, None), total=num_batches
    ):
        if epoch > cfg.epochs:
            break

        if epoch not in seen_epochs:
            seen_epochs.add(epoch)
            if accumulation_steps != 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            dev_loss = dev(cfg, dataset, model)
            epoch_info.update(dev_loss=dev_loss, best_loss=best_loss)
            if dev_loss < best_loss:
                best_loss = dev_loss
                logger.info("Save model...")
                torch.save(model.state_dict(), cfg.best_model_path)

            logger.info(str(epoch_info))

        model.train()
        batch["epoch"] = epoch - 1
        outputs = model(prepare_inputs(batch, cfg.device))
        loss = outputs["loss"]
        for k, v in outputs.items():
            if "loss" in k:
                epoch_info[k] = round(v.cpu().item(), 4)
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


def dev(cfg, dataset, model, data_split: str = "dev"):
    model.zero_grad()
    losses = []
    for _, batch in dataset.get_batch(data_split, cfg.test_batch_size, None):
        model.eval()
        with torch.no_grad():
            outputs = model(prepare_inputs(batch, cfg.device))
            losses.append(outputs["loss"].cpu().item())

    return np.mean(losses)


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

    if cfg.test:
        vocab = Vocabulary.load(cfg.vocabulary_file)
    else:
        vocab.save(cfg.vocabulary_file)

    # joint model
    model = EntRelJointDecoder(cfg=cfg, vocab=vocab, ent_rel_file=ent_rel_file)
    if cfg.device > -1:
        model.cuda(device=cfg.device)

    if not Path(cfg.best_model_path).exists():
        train(cfg, ace_dataset, model)
    raw_predict(cfg, ace_dataset, model)


def raw_predict(cfg, dataset, model, data_split: str = "test"):
    print(dict(load=cfg.best_model_path))
    model.load_state_dict(torch.load(cfg.best_model_path))
    model.eval()
    outputs = []

    num_batches = dataset.get_dataset_size(data_split) // cfg.test_batch_size
    for _, batch in tqdm(
        dataset.get_batch(data_split, cfg.test_batch_size, None), total=num_batches
    ):
        with torch.no_grad():
            batch = prepare_inputs(batch, cfg.device)
            for raw in model.raw_predict(batch):
                outputs.append(raw)

    path = Path(cfg.save_dir) / f"pred_{data_split}_raw.npy"
    np.save(path, outputs)


"""

python q_main.py \
--ent_rel_file label_vocab.json \
--train_batch_size 16 \
--gradient_accumulation_steps 2 \
--config_file config.yml \
--save_dir ckpt/quintuplet \
--data_dir data/quintuplet \
--fine_tune \
--max_sent_len 80 \
--max_wordpiece_len 80 \
--epochs 50 \
--pretrain_epochs 0 \
--device 0

"""


if __name__ == "__main__":
    main()
