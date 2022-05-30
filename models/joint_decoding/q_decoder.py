import json
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from models.embedding_models.bert_embedding_model import BertEmbedModel
from models.embedding_models.pretrained_embedding_model import \
    PretrainedEmbedModel
from modules.token_embedders.bert_encoder import BertLinear
from utils.nn_utils import batched_index_select

logger = logging.getLogger(__name__)


def check_adjacent(
    i: int,
    j: int,
    k: int,
    bounds: Tuple[int, int, int, int, int, int],
) -> bool:
    return (
        (bounds[0] - 1 <= i < bounds[1] + 1)
        and (bounds[2] - 1 <= j < bounds[3] + 1)
        and (bounds[4] - 1 <= k < bounds[5] + 1)
    )


def update_bounds(
    i: int, j: int, k: int, bounds: Tuple[int, int, int, int, int, int]
) -> Tuple[int, int, int, int, int, int]:
    return (
        min(i, bounds[0]),
        max(i + 1, bounds[1]),
        min(j, bounds[2]),
        max(j + 1, bounds[3]),
        min(k, bounds[4]),
        max(k + 1, bounds[5]),
    )


def decode_nonzero_cuboids(
    x: torch.Tensor,
) -> List[Tuple[int, int, int, int, int, int]]:
    cuboids = []
    for coordinates in x.nonzero():  # Assumes lexicographic sorting
        i, j, k = map(int, coordinates)
        for idx, bounds in enumerate(cuboids):
            if check_adjacent(i, j, k, bounds):
                cuboids[idx] = update_bounds(i, j, k, bounds)
                break
        else:
            cuboids.append((i, i + 1, j, j + 1, k, k + 1))
    return cuboids


def prune_matrix(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 4
    assert indices.ndim == 2
    bs, topk = indices.shape
    assert x.shape[0] == bs
    samples = []
    for i in range(bs):
        y = x[i]
        y = y[indices[i], :, :]
        y = y[:, indices[i], :]
        y = y[:, :, indices[i]]
        assert y.shape == (topk, topk, topk)
        samples.append(y)

    # x = batched_index_select(x, indices)
    # x = batched_index_select(x.permute(0, 3, 1, 2), indices)
    # x = batched_index_select(x.permute(0, 3, 1, 2), indices)
    # x = x.permute(0, 3, 1, 2)
    # assert x.shape == (bs, topk, topk, topk)
    # return x

    return torch.stack(samples, dim=0)


class EntRelJointDecoder(nn.Module):
    def __init__(self, cfg, vocab, ent_rel_file):
        """__init__ constructs `EntRelJointDecoder` components and
        sets `EntRelJointDecoder` parameters. This class adopts a joint
        decoding algorithm for entity relation joint decoing and facilitates
        the interaction between entity and relation.

        Args:
            cfg (dict): config parameters for constructing multiple models
            vocab (Vocabulary): vocabulary
            ent_rel_file (dict): entity and relation file (joint id, entity id, relation id)
        """

        super().__init__()
        self.cfg = cfg
        self.ent_rel_file = ent_rel_file
        self.vocab = vocab
        self.max_span_length = cfg.max_span_length
        self.activation = nn.GELU()
        self.device = cfg.device
        self.separate_threshold = cfg.separate_threshold
        print(json.dumps(vars(self.cfg), indent=2))

        if cfg.embedding_model == "bert":
            self.embedding_model = BertEmbedModel(cfg, vocab)
        elif cfg.embedding_model == "pretrained":
            self.embedding_model = PretrainedEmbedModel(cfg, vocab)
        self.encoder_output_size = self.embedding_model.get_hidden_size()

        self.head_mlp = BertLinear(
            input_size=self.encoder_output_size,
            output_size=cfg.mlp_hidden_size,
            activation=self.activation,
            dropout=cfg.dropout,
        )
        self.tail_mlp = BertLinear(
            input_size=self.encoder_output_size,
            output_size=cfg.mlp_hidden_size,
            activation=self.activation,
            dropout=cfg.dropout,
        )

        self.pair_mlp = BertLinear(
            input_size=(self.encoder_output_size) * 2,
            output_size=cfg.mlp_hidden_size,
            activation=self.activation,
            dropout=cfg.dropout,
        )

        if self.get_config("use_pair2_mlp"):
            self.pair2_mlp = BertLinear(
                input_size=(self.encoder_output_size) * 2,
                output_size=cfg.mlp_hidden_size,
                activation=self.activation,
                dropout=cfg.dropout,
            )

        if self.get_config("use_entity_loss"):
            self.entity_mlp = BertLinear(
                input_size=self.embedding_model.get_hidden_size(),
                output_size=ent_rel_file["q_num_logits"],
                activation=nn.Identity(),
                dropout=0.0,
            )

        self.final_mlp = BertLinear(
            input_size=cfg.mlp_hidden_size,
            output_size=self.vocab.get_vocab_size("ent_rel_id"),
            activation=nn.Identity(),
            dropout=0.0,
        )

        self.value_mlp = BertLinear(
            input_size=self.encoder_output_size,
            output_size=cfg.mlp_hidden_size,
            activation=self.activation,
            dropout=cfg.dropout,
        )

        self.U = nn.parameter.Parameter(
            torch.FloatTensor(
                ent_rel_file["q_num_logits"],
                cfg.mlp_hidden_size,
                cfg.mlp_hidden_size,
            )
        )
        self.U.data.zero_()

        if cfg.logit_dropout > 0:
            self.logit_dropout = nn.Dropout(p=cfg.logit_dropout)
        else:
            self.logit_dropout = lambda x: x

        self.none_idx = self.vocab.get_token_index("None", "ent_rel_id")
        self.ent_label = np.array(self.ent_rel_file["entity"])
        self.rel_label = np.array(self.ent_rel_file["relation"])
        self.q_label = np.array(self.ent_rel_file["qualifier"])
        self.element_loss = nn.CrossEntropyLoss()
        self.quintuplet_loss = nn.CrossEntropyLoss()
        self.prune_topk = self.get_config("prune_topk") or 0

    def get_config(self, key: str):
        return getattr(self.cfg, key, None)

    def forward(self, batch_inputs):
        """forward

        Arguments:
            batch_inputs {dict} -- batch input data

        Returns:
            dict -- results: ent_loss, ent_pred
        """

        self.embedding_model(batch_inputs)
        batch_seq_tokens_encoder_repr = batch_inputs["seq_encoder_reprs"]

        batch_size, seq_len, hidden_size = batch_seq_tokens_encoder_repr.shape
        head = batch_seq_tokens_encoder_repr.unsqueeze(dim=2).expand(
            -1, -1, seq_len, -1
        )
        tail = batch_seq_tokens_encoder_repr.unsqueeze(dim=1).expand(
            -1, seq_len, -1, -1
        )
        pair = torch.cat([head, tail], dim=-1)
        pair = self.pair_mlp(pair)
        batch_joint_score = self.final_mlp(pair)

        if self.prune_topk > 0:
            topk = min(seq_len, self.prune_topk)
            assert self.get_config("use_pair2_mlp")
            seq_mask = batch_inputs["joint_label_matrix_mask"].diagonal(dim1=1, dim2=2)
            seq_score = batch_joint_score.diagonal(dim1=1, dim2=2).permute(0, 2, 1)
            seq_score = seq_score[:, :, list(self.ent_label)].max(dim=-1).values
            assert seq_mask.shape == seq_score.shape
            seq_score = torch.where(seq_mask, seq_score, seq_score.min())
            bs, _ = seq_score.shape
            indices = seq_score.topk(k=topk, dim=1).indices
            assert indices.shape == (bs, topk)
            pruned = batched_index_select(batch_seq_tokens_encoder_repr, indices)
            head = pruned.unsqueeze(dim=2).expand(-1, -1, topk, -1)
            tail = pruned.unsqueeze(dim=1).expand(-1, topk, -1, -1)
            batch_seq_tokens_encoder_repr = pruned
            for k in ["quintuplet_matrix_mask", "quintuplet_matrix"]:
                batch_inputs[k] = prune_matrix(batch_inputs[k], indices)
        else:
            indices = None

        if self.get_config("use_pair2_mlp"):
            # Don't share representations with table/triplets
            pair = self.pair2_mlp(torch.cat([head, tail], dim=-1))

        value = self.value_mlp(batch_seq_tokens_encoder_repr)
        q_score = torch.einsum("bxyi, oij, bzj -> bxyzo", pair, self.U, value)
        mask = batch_inputs["quintuplet_matrix_mask"]
        assert q_score.shape[:-1] == mask.shape
        q_loss = self.quintuplet_loss(
            q_score.softmax(-1)[mask], batch_inputs["quintuplet_matrix"][mask]
        )

        if self.get_config("fix_q_loss"):
            # Don't softmax before crossentropy and add logit dropout
            q_loss = self.quintuplet_loss(
                self.logit_dropout(q_score[mask]),
                batch_inputs["quintuplet_matrix"][mask],
            )

        results = {}
        if not self.training:
            batch_normalized_joint_score = (
                torch.softmax(batch_joint_score, dim=-1)
                * batch_inputs["joint_label_matrix_mask"].unsqueeze(-1).float()
            )
            batch_normalized_q_score = (
                torch.softmax(q_score, dim=-1)
                * batch_inputs["quintuplet_matrix_mask"].unsqueeze(-1).float()
            )

            results["joint_label_preds"] = torch.argmax(
                batch_normalized_joint_score, dim=-1
            )
            results["quintuplet_preds"] = torch.argmax(batch_normalized_q_score, dim=-1)

            batch_seq_tokens_lens = batch_inputs["tokens_lens"]
            decode_preds = self.soft_joint_decoding(
                batch_normalized_joint_score,
                batch_seq_tokens_lens,
                batch_normalized_q_score,
                prune_indices=indices,
            )
            results.update(decode_preds)

        results["element_loss"] = self.element_loss(
            self.logit_dropout(
                batch_joint_score[batch_inputs["joint_label_matrix_mask"]]
            ),
            batch_inputs["joint_label_matrix"][batch_inputs["joint_label_matrix_mask"]],
        )
        results["q_loss"] = q_loss
        results["loss"] = results["element_loss"] + results["q_loss"]
        results["joint_score"] = batch_joint_score
        results["q_score"] = q_score

        if self.get_config("use_entity_loss"):
            entity_score = self.entity_mlp(batch_seq_tokens_encoder_repr)
            entity_mask = batch_inputs["joint_label_matrix_mask"].diagonal(
                dim1=1, dim2=2
            )
            entity_labels = batch_inputs["joint_label_matrix"].diagonal(dim1=1, dim2=2)
            results["entity_loss"] = self.element_loss(
                self.logit_dropout(entity_score[entity_mask]),
                entity_labels[entity_mask],
            )
            results["loss"] += results["entity_loss"]

        return results

    def soft_joint_decoding(
        self,
        batch_normalized_joint_score,
        batch_seq_tokens_lens,
        batch_normalized_q_score,
        prune_indices: Optional[torch.Tensor],
    ) -> dict:
        """soft_joint_decoding extracts entity and relation at the same time,
        and consider the interconnection of entity and relation.

        Args:
            batch_normalized_joint_score (tensor): batch normalized joint score
            batch_seq_tokens_lens (list): batch sequence length

        Returns:
            tuple: predicted entity and relation
        """

        separate_position_preds = []
        ent_preds = []
        rel_preds = []
        q_preds = []

        ent_label = self.ent_label
        rel_label = self.rel_label
        q_label = self.q_label

        for idx, seq_len in enumerate(batch_seq_tokens_lens):
            separate_position_preds.append([])
            ent_pred = {}
            rel_pred = {}
            q_pred = {}

            if prune_indices is not None:
                indices = [i for i in prune_indices[idx].tolist() if i < seq_len]
                pairs = [(i, j) for i, j in enumerate(indices)]
                topk, _, _, num_logits = batch_normalized_q_score[idx].shape
                d = batch_normalized_joint_score.device
                temp = torch.full((seq_len, topk, topk, num_logits), -1e9, device=d)
                for i, j in pairs:
                    temp[j, :, :] = batch_normalized_q_score[idx][i, :, :]
                temp2 = torch.full((seq_len, seq_len, topk, num_logits), -1e9, device=d)
                for i, j in pairs:
                    temp2[:, j, :] = temp[:, i, :]
                temp3 = torch.full(
                    (seq_len, seq_len, seq_len, num_logits), -1e9, device=d
                )
                for i, j in pairs:
                    temp3[:, :, j] = temp2[:, :, i]
                q_score = temp3
            else:
                q_score = batch_normalized_q_score[idx][:seq_len, :seq_len, :seq_len, :]

            joint_score = batch_normalized_joint_score[idx][:seq_len, :seq_len, :]
            cuboids = decode_nonzero_cuboids(q_score.argmax(-1))
            for i_start, i_end, j_start, j_end, k_start, k_end in cuboids:
                score = q_score[
                    i_start:i_end,
                    j_start:j_end,
                    k_start:k_end,
                ].mean((0, 1, 2))
                pred = q_label[score.cpu().numpy()[q_label].argmax()].item()
                pred_label = self.vocab.get_token_from_index(pred, "ent_rel_id")
                spans = ((i_start, i_end), (j_start, j_end), (k_start, k_end))
                q_pred[spans] = pred_label

                score = joint_score[i_start:i_end, j_start:j_end].mean((0, 1))
                pred = rel_label[score.cpu().numpy()[rel_label].argmax()].item()
                pred_label = self.vocab.get_token_from_index(pred, "ent_rel_id")
                rel_pred[spans[:2]] = pred_label

                for sp in spans:
                    pred = ent_label[0].item()
                    ent_pred[sp] = self.vocab.get_token_from_index(pred, "ent_rel_id")

            ent_preds.append(ent_pred)
            rel_preds.append(rel_pred)
            q_preds.append(q_pred)

        return dict(
            all_separate_position_preds=separate_position_preds,
            all_ent_preds=ent_preds,
            all_rel_preds=rel_preds,
            all_q_preds=q_preds,
        )

    def save(self, path: str):
        device = self.device
        info = dict(
            state_dict=self.cpu().state_dict(),
            cfg=self.cfg,
            vocab=self.vocab,
            ent_rel_file=self.ent_rel_file,
        )
        torch.save(info, path)
        self.to(device)
        print(dict(save=path))

    @classmethod
    def load(cls, path):
        print(dict(load=path))
        info = torch.load(path)
        state_dict = info.pop("state_dict")
        model = cls(**info)
        model.load_state_dict(state_dict)
        if model.cfg.device > -1:
            model.cuda(device=model.cfg.device)
            print(dict(cuda=model.cfg.device))
        return model
