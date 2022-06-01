import json
import logging
import pickle
import random
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.functional import Tensor
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from data.q_process import (Sentence, SparseCube, load_raw_preds,
                            match_sent_preds)
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


def score_preds(path_pred: str, path_gold: str, path_vocab: str) -> dict:
    raw_preds = load_raw_preds(path_pred)
    vocab = Vocabulary.load(path_vocab)
    with open(path_gold) as f:
        sents = [Sentence(**json.loads(line)) for line in f]

    print(dict(preds=len(raw_preds), sents=len(sents)))
    preds = match_sent_preds(sents, raw_preds, vocab)
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
    best_score = -1e9
    accumulation_steps = 0
    model.zero_grad()
    seen_epochs = set()
    epoch_info = dict(epoch=0.0)

    for epoch, batch in tqdm(
        dataset.get_batch("train", cfg.train_batch_size, None), total=num_batches
    ):
        if epoch > cfg.epochs:
            model.save(cfg.last_model_path)
            break

        if epoch not in seen_epochs:
            seen_epochs.add(epoch)
            if accumulation_steps != 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            dev_loss, dev_score = evaluate(cfg, dataset, model, data_split="dev")
            epoch_info.update(
                dev_loss=dev_loss,
                best_loss=best_loss,
                score=dev_score,
                best_score=best_score,
            )
            if dev_loss < best_loss:
                best_loss = dev_loss
            if dev_score > best_score:
                best_score = dev_score
                logger.info(str(dict(save=cfg.best_model_path)))
                model.save(cfg.best_model_path)

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

    mapping = dict(train=cfg.train_file, dev=cfg.dev_file, test=cfg.test_file)
    results = score_preds(
        path_pred=str(path),
        path_gold=path_in or mapping[data_split],
        path_vocab=cfg.vocabulary_file,
    )

    score = dict(
        quintuplet=results["quintuplet"]["f1"],
        triplet=results["entity"]["f1"] + results["strict triplet"]["f1"],
        tagger=results["entity"]["f1"],
    )[cfg.task]
    return np.mean(losses), score


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
    model = load_model(cfg.task, cfg=cfg, vocab=vocab, ent_rel_file=ent_rel_file)
    if cfg.device > -1:
        model.cuda(device=cfg.device)

    if cfg.load_weight_path:
        model.load_state_dict(
            EntRelJointDecoder.load(cfg.load_weight_path).state_dict()
        )

    path_data = str(Path(cfg.save_dir) / "dataset.pickle")
    ace_dataset.save(path_data)
    train(cfg, ace_dataset, model)
    run_eval(
        path=cfg.best_model_path, path_data=path_data, data_split="test", task=cfg.task
    )


"""
################################################################################
Distant supervised + filtered dev/test (10)

p q_main.py \
--save_dir ckpt/q10_pair2_fix_q_loss_prune_0 \
--data_dir data/q10 \
--prune_topk 0 \
--use_pair2_mlp \
--fix_q_loss \
--config_file q_config.yml

"precision": 0.6625742574257426,                                         
"recall": 0.6447013487475916,                                            
"f1": 0.653515625

p q_main.py \
--save_dir ckpt/q10_pair2_fix_q_loss_prune_10 \
--data_dir data/q10 \
--prune_topk 10 \
--use_pair2_mlp \
--fix_q_loss \
--config_file q_config.yml
                                                                                                                       
"precision": 0.6751721344673957,                                                                                                                          "recall": 0.6423892100192679,                                                                                                                         
"f1": 0.6583728278041073

p q_main.py \
--save_dir ckpt/q10_pair2_fix_q_loss_prune_20 \
--data_dir data/q10 \
--prune_topk 20 \
--use_pair2_mlp \
--fix_q_loss \
--config_file q_config.yml

"precision": 0.6719291490180979,
"recall": 0.6724470134874759,
"f1": 0.6721879815100154

p q_main.py \
--save_dir ckpt/q10_pair2_fix_q_loss_prune_30 \
--data_dir data/q10 \
--prune_topk 30 \
--use_pair2_mlp \
--fix_q_loss \
--config_file q_config.yml

"precision": 0.6750499001996008,
"recall": 0.6516377649325626,  
"f1": 0.6631372549019608

p q_main.py \
--save_dir ckpt/q10_pair2_fix_q_loss_prune_40 \
--data_dir data/q10 \
--prune_topk 40 \
--use_pair2_mlp \
--fix_q_loss \
--config_file q_config.yml
                                                                                                                      
"precision": 0.6715686274509803,
"recall": 0.6335260115606937,     
"f1": 0.6519928613920285

p q_main.py \
--save_dir ckpt/q10_pair2_fix_q_loss \
--data_dir data/q10 \
--use_pair2_mlp \
--fix_q_loss \
--config_file q_config.yml

"precision": 0.6717373899119295,                                                                                                                          "recall": 0.6466281310211947,                                                                                                                         
"f1": 0.6589436481445121

p q_main.py \
--save_dir ckpt/q10r_pair2_fix_q_loss \
--data_dir data/q10r \
--embedding_model pretrained \
--pretrained_model_name roberta-base \
--use_pair2_mlp \
--fix_q_loss \
--config_file q_config.yml

"precision": 0.6874216464688675,
"recall": 0.6339113680154143,
"f1": 0.6595829991980754

p q_main.py \
--save_dir ckpt/q10_pair2_fix_q_loss_labeled_train_transfer \
--data_dir data/q10_labeled_train \
--load_weight_path ckpt/q10_pair2_fix_q_loss/best_model \
--use_pair2_mlp \
--fix_q_loss \
--config_file q_config.yml

"precision": 0.696078431372549,
"recall": 0.7029702970297029,
"f1": 0.6995073891625616

################################################################################
Triplet task (10)

p q_main.py \
--save_dir ckpt/q10_triplet \
--data_dir data/q10 \
--task triplet \
--config_file q_config.yml

"precision": 0.7585431654676259,
"recall": 0.7296712802768166,
"f1": 0.7438271604938272

################################################################################
Tagger task (10)

p q_main.py \
--task tagger \
--ent_rel_file label.json \
--train_batch_size 32 \
--config_file config.yml \
--save_dir ckpt/q10_tagger \
--data_dir data/q10_tagger \
--fine_tune \
--max_sent_len 90 \
--max_wordpiece_len 90 \
--epochs 30 \
--pretrain_epochs 0 \
--device 0

"""


if __name__ == "__main__":
    main()
