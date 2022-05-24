import logging

import torch
import torch.nn as nn
from modules.token_embedders.bert_encoder import BertLinear
from transformers import AutoModel
from utils.nn_utils import gelu

logger = logging.getLogger(__name__)


class PretrainedEncoder(nn.Module):
    """This class using pre-trained model to encode token,
    then fine-tuning the pre-trained model
    """

    def __init__(
        self,
        pretrained_model_name,
        trainable=False,
        output_size=0,
        activation=gelu,
        dropout=0.0,
    ):
        """This function initialize pertrained model

        Arguments:
            pretrained_model_name {str} -- pre-trained model name

        Keyword Arguments:
            output_size {float} -- output size (default: {None})
            activation {nn.Module} -- activation function (default: {gelu})
            dropout {float} -- dropout rate (default: {0.0})
        """

        super().__init__()
        self.pretrained_model = AutoModel.from_pretrained(pretrained_model_name)
        logger.info(
            "Load pre-trained model {} successfully.".format(pretrained_model_name)
        )

        self.output_size = output_size

        if trainable:
            logger.info(
                "Start fine-tuning pre-trained model {}.".format(pretrained_model_name)
            )
        else:
            logger.info(
                "Keep fixed pre-trained model {}.".format(pretrained_model_name)
            )

        for param in self.pretrained_model.parameters():
            param.requires_grad = trainable

        if self.output_size > 0:
            self.mlp = BertLinear(
                input_size=self.pretrained_model.config.hidden_size,
                output_size=self.output_size,
                activation=activation,
            )
        else:
            self.output_size = self.pretrained_model.config.hidden_size
            self.mlp = lambda x: x

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

        self.pad_id = self.pretrained_model.config.pad_token_id
        self.model_type = self.pretrained_model.config.model_type
        logger.info(str(dict(pad_id=self.pad_id, model_type=self.model_type)))

    def get_output_dims(self):
        return self.output_size

    def forward(self, seq_inputs, token_type_inputs=None):
        """forward calculates forward propagation results, get token embedding

        Args:
            seq_inputs {tensor} -- sequence inputs (tokenized)
            token_type_inputs (tensor, optional): token type inputs. Defaults to None.

        Returns:
            tensor: bert output for tokens
        """

        if token_type_inputs is None:
            token_type_inputs = torch.zeros_like(seq_inputs)
        if self.model_type == "roberta":
            token_type_inputs = None
        mask_inputs = (seq_inputs != self.pad_id).long()

        if self.model_type == "distilbert":
            outputs = self.pretrained_model(
                input_ids=seq_inputs,
                attention_mask=mask_inputs,
            )
            last_hidden_state = outputs[0]
            pooled_output = outputs[0].mean(dim=1)  # distilbert has no pooled output
        else:
            outputs = self.pretrained_model(
                input_ids=seq_inputs,
                token_type_ids=token_type_inputs,
                attention_mask=mask_inputs,
            )
            last_hidden_state = outputs[0]
            pooled_output = outputs[1]

        return self.dropout(self.mlp(last_hidden_state)), self.dropout(
            self.mlp(pooled_output)
        )
