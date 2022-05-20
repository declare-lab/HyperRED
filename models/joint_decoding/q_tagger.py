import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from models.embedding_models.bert_embedding_model import BertEmbedModel
from models.embedding_models.pretrained_embedding_model import \
    PretrainedEmbedModel
from modules.token_embedders.bert_encoder import BertLinear

logger = logging.getLogger(__name__)


def decode_nonzero_spans(labels: List[int]) -> List[Tuple[int, int]]:
    i = -1
    spans = []

    for j, x in enumerate(labels):
        assert isinstance(x, int)
        if x == 0 and i != -1:
            assert 0 <= i < j
            assert j <= len(labels)
            spans.append((i, j))
            i = -1
        elif x != 0 and i == -1:
            i = j

    if i != -1:
        spans.append((i, len(labels)))
    assert len(set(spans)) == len(spans)
    return spans


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

        if cfg.embedding_model == "bert":
            self.embedding_model = BertEmbedModel(cfg, vocab)
        elif cfg.embedding_model == "pretrained":
            self.embedding_model = PretrainedEmbedModel(cfg, vocab)

        self.final_mlp = BertLinear(
            input_size=self.embedding_model.get_hidden_size(),
            output_size=ent_rel_file["q_num_logits"],
            activation=nn.Identity(),
            dropout=0.0,
        )

        if cfg.logit_dropout > 0:
            self.logit_dropout = nn.Dropout(p=cfg.logit_dropout)
        else:
            self.logit_dropout = lambda x: x

        self.none_idx = self.vocab.get_token_index("None", "ent_rel_id")
        self.q_label = np.array(self.ent_rel_file["qualifier"])
        self.element_loss = nn.CrossEntropyLoss()

    def forward(self, batch_inputs):
        """forward

        Arguments:
            batch_inputs {dict} -- batch input data

        Returns:
            dict -- results: ent_loss, ent_pred
        """

        self.embedding_model(batch_inputs)
        batch_seq_tokens_encoder_repr = batch_inputs["seq_encoder_reprs"]
        batch_joint_score = self.final_mlp(batch_seq_tokens_encoder_repr)
        batch_mask = batch_inputs["joint_label_matrix_mask"].diagonal(dim1=1, dim2=2)

        results = {}
        if not self.training:
            batch_normalized_joint_score = (
                torch.softmax(batch_joint_score, dim=-1)
                * batch_mask.unsqueeze(-1).float()
            )

            bs, seq_len, num_labels = batch_normalized_joint_score.shape
            results["joint_label_preds"] = torch.diag_embed(
                batch_normalized_joint_score.argmax(-1)
            )
            results["quintuplet_preds"] = torch.zeros(bs, seq_len, seq_len, seq_len)

            batch_seq_tokens_lens = batch_inputs["tokens_lens"]
            decode_preds = self.soft_joint_decoding(
                batch_normalized_joint_score, batch_seq_tokens_lens
            )
            results.update(decode_preds)

        batch_labels = batch_inputs["joint_label_matrix"].diagonal(dim1=1, dim2=2)
        results["loss"] = self.element_loss(
            self.logit_dropout(batch_joint_score[batch_mask]),
            batch_labels[batch_mask],
        )
        results["joint_score"] = batch_joint_score
        return results

    def soft_joint_decoding(
        self, batch_normalized_joint_score, batch_seq_tokens_lens
    ) -> dict:
        """soft_joint_decoding extracts entity and relation at the same time,
        and consider the interconnection of entity and relation.

        Args:
            batch_normalized_joint_score (tensor): batch normalized joint score
            batch_seq_tokens_lens (list): batch sequence length

        Returns:
            tuple: predicted entity and relation
        """

        ent_preds = []
        batch_normalized_joint_score = batch_normalized_joint_score.cpu().numpy()
        ent_label = self.q_label

        for idx, seq_len in enumerate(batch_seq_tokens_lens):
            ent_pred = {}
            joint_score = batch_normalized_joint_score[idx][:seq_len, :]
            joint_preds = joint_score.argmax(-1)
            assert joint_preds.shape == (seq_len,)
            spans = decode_nonzero_spans([int(x) for x in joint_preds])

            for span in spans:
                score = np.mean(joint_score[span[0] : span[1], :], axis=0)
                pred = ent_label[np.argmax(score[ent_label])].item()
                pred_label = self.vocab.get_token_from_index(pred, "ent_rel_id")
                ent_pred[span] = pred_label
            ent_preds.append(ent_pred)

        assert len(ent_preds) == len(batch_seq_tokens_lens)
        return dict(
            all_separate_position_preds=[[] for _ in ent_preds],
            all_ent_preds=ent_preds,
            all_rel_preds=[{} for _ in ent_preds],
            all_q_preds=[{} for _ in ent_preds],
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
