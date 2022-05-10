import logging

import numpy as np
import torch
import torch.nn as nn
from models.embedding_models.bert_embedding_model import BertEmbedModel
from models.embedding_models.pretrained_embedding_model import \
    PretrainedEmbedModel
from modules.token_embedders.bert_encoder import BertLinear

logger = logging.getLogger(__name__)


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
        self.vocab = vocab
        self.max_span_length = cfg.max_span_length
        self.activation = nn.GELU()
        self.device = cfg.device
        self.separate_threshold = cfg.separate_threshold

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
        self.ent_label = torch.LongTensor(ent_rel_file["entity"])
        self.rel_label = torch.LongTensor(ent_rel_file["relation"])
        self.q_label = torch.LongTensor(ent_rel_file["qualifier"])
        self.element_loss = nn.CrossEntropyLoss()
        self.quintuplet_loss = nn.CrossEntropyLoss()

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

        value = self.value_mlp(batch_seq_tokens_encoder_repr)
        q_score = torch.einsum("bxyi, oij, bzj -> bxyzo", pair, self.U, value)
        mask = batch_inputs["quintuplet_matrix_mask"]
        assert q_score.shape[:-1] == mask.shape
        q_loss = self.quintuplet_loss(
            q_score.softmax(-1)[mask], batch_inputs["quintuplet_matrix"][mask]
        )

        results = {}
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
        return results

    def soft_joint_decoding(self, batch_normalized_joint_score, batch_seq_tokens_lens):
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

        batch_normalized_joint_score = batch_normalized_joint_score.cpu().numpy()
        ent_label = self.ent_label.cpu().numpy()
        rel_label = self.rel_label.cpu().numpy()

        for idx, seq_len in enumerate(batch_seq_tokens_lens):
            ent_pred = {}
            rel_pred = {}
            joint_score = batch_normalized_joint_score[idx][:seq_len, :seq_len, :]
            joint_score_feature = joint_score.reshape(seq_len, -1)
            transposed_joint_score_feature = joint_score.transpose((1, 0, 2)).reshape(
                seq_len, -1
            )
            separate_pos = (
                (
                    np.linalg.norm(
                        joint_score_feature[0 : seq_len - 1]
                        - joint_score_feature[1:seq_len],
                        axis=1,
                    )
                    + np.linalg.norm(
                        transposed_joint_score_feature[0 : seq_len - 1]
                        - transposed_joint_score_feature[1:seq_len],
                        axis=1,
                    )
                )
                * 0.5
                > self.separate_threshold
            ).nonzero()[0]
            separate_position_preds.append([pos.item() for pos in separate_pos])
            if len(separate_pos) > 0:
                spans = [
                    (0, separate_pos[0].item() + 1),
                    (separate_pos[-1].item() + 1, seq_len),
                ] + [
                    (separate_pos[idx].item() + 1, separate_pos[idx + 1].item() + 1)
                    for idx in range(len(separate_pos) - 1)
                ]
            else:
                spans = [(0, seq_len)]

            ents = []
            for span in spans:
                score = np.mean(
                    joint_score[span[0] : span[1], span[0] : span[1], :], axis=(0, 1)
                )
                if not (np.max(score[ent_label]) < score[self.none_idx]):
                    pred = ent_label[np.argmax(score[ent_label])].item()
                    pred_label = self.vocab.get_token_from_index(pred, "ent_rel_id")
                    ents.append(span)
                    ent_pred[span] = pred_label

            for ent1 in ents:
                for ent2 in ents:
                    if ent1 == ent2:
                        continue
                    score = np.mean(
                        joint_score[ent1[0] : ent1[1], ent2[0] : ent2[1], :],
                        axis=(0, 1),
                    )
                    if not (np.max(score[rel_label]) < score[self.none_idx]):
                        pred = rel_label[np.argmax(score[rel_label])].item()
                        pred_label = self.vocab.get_token_from_index(pred, "ent_rel_id")
                        rel_pred[(ent1, ent2)] = pred_label

            ent_preds.append(ent_pred)
            rel_preds.append(rel_pred)

        return separate_position_preds, ent_preds, rel_preds
