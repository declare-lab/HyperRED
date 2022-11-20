import logging

import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import BertModel

from utils.nn_utils import batched_index_select
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


class BertEncoder(nn.Module):
    """This class using pretrained `Bert` model to encode token,
    then fine-tuning  `Bert` model
    """

    def __init__(
        self,
        bert_model_name,
        trainable=False,
        output_size=0,
        activation=gelu,
        dropout=0.0,
    ):
        """This function initialize pertrained `Bert` model

        Arguments:
            bert_model_name {str} -- bert model name

        Keyword Arguments:
            output_size {float} -- output size (default: {None})
            activation {nn.Module} -- activation function (default: {gelu})
            dropout {float} -- dropout rate (default: {0.0})
        """

        super().__init__()
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        logger.info("Load bert model {} successfully.".format(bert_model_name))

        self.output_size = output_size

        if trainable:
            logger.info("Start fine-tuning bert model {}.".format(bert_model_name))
        else:
            logger.info("Keep fixed bert model {}.".format(bert_model_name))

        for param in self.bert_model.parameters():
            param.requires_grad = trainable

        if self.output_size > 0:
            self.mlp = BertLinear(
                input_size=self.bert_model.config.hidden_size,
                output_size=self.output_size,
                activation=activation,
            )
        else:
            self.output_size = self.bert_model.config.hidden_size
            self.mlp = lambda x: x

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

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
        mask_inputs = (seq_inputs != 0).long()

        outputs = self.bert_model(
            input_ids=seq_inputs,
            attention_mask=mask_inputs,
            token_type_ids=token_type_inputs,
        )
        last_hidden_state = outputs[0]
        pooled_output = outputs[1]

        return self.dropout(self.mlp(last_hidden_state)), self.dropout(
            self.mlp(pooled_output)
        )


class BertLayerNorm(nn.Module):
    """This class is LayerNorm model for Bert"""

    def __init__(self, hidden_size, eps=1e-12):
        """This function sets `BertLayerNorm` parameters

        Arguments:
            hidden_size {int} -- input size

        Keyword Arguments:
            eps {float} -- epsilon (default: {1e-12})
        """

        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        """This function propagates forwardly

        Arguments:
            x {tensor} -- input tesor

        Returns:
            tensor -- LayerNorm outputs
        """

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertLinear(nn.Module):
    """This class is Linear model for Bert"""

    def __init__(self, input_size, output_size, activation=gelu, dropout=0.0):
        """This function sets `BertLinear` model parameters

        Arguments:
            input_size {int} -- input size
            output_size {int} -- output size

        Keyword Arguments:
            activation {function} -- activation function (default: {gelu})
            dropout {float} -- dropout rate (default: {0.0})
        """

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)
        self.linear.weight.data.normal_(mean=0.0, std=0.02)
        self.linear.bias.data.zero_()
        self.activation = activation
        self.layer_norm = BertLayerNorm(self.output_size)

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

    def get_input_dims(self):
        return self.input_size

    def get_output_dims(self):
        return self.output_size

    def forward(self, x):
        """This function propagates forwardly

        Arguments:
            x {tensor} -- input tensor

        Returns:
            tenor -- Linear outputs
        """

        output = self.activation(self.linear(x))
        return self.dropout(self.layer_norm(output))


class BertEmbedModel(nn.Module):
    """This class acts as an embeddding layer with bert model"""

    def __init__(self, cfg, vocab):
        """This function constructs `BertEmbedModel` components and
        sets `BertEmbedModel` parameters

        Arguments:
            cfg {dict} -- config parameters for constructing multiple models
            vocab {Vocabulary} -- vocabulary
        """

        super().__init__()
        self.activation = gelu
        self.bert_encoder = BertEncoder(
            bert_model_name=cfg.bert_model_name,
            trainable=cfg.fine_tune,
            output_size=cfg.bert_output_size,
            activation=self.activation,
            dropout=cfg.bert_dropout,
        )
        self.encoder_output_size = self.bert_encoder.get_output_dims()

    def forward(self, batch_inputs):
        """This function propagetes forwardly

        Arguments:
            batch_inputs {dict} -- batch input data
        """

        if "wordpiece_segment_ids" in batch_inputs:
            batch_seq_bert_encoder_repr, batch_cls_repr = self.bert_encoder(
                batch_inputs["wordpiece_tokens"], batch_inputs["wordpiece_segment_ids"]
            )
        else:
            batch_seq_bert_encoder_repr, batch_cls_repr = self.bert_encoder(
                batch_inputs["wordpiece_tokens"]
            )

        batch_seq_tokens_encoder_repr = batched_index_select(
            batch_seq_bert_encoder_repr, batch_inputs["wordpiece_tokens_index"]
        )

        batch_inputs["seq_encoder_reprs"] = batch_seq_tokens_encoder_repr
        batch_inputs["seq_cls_repr"] = batch_cls_repr

    def get_hidden_size(self):
        """This function returns embedding dimensions

        Returns:
            int -- embedding dimensitons
        """

        return self.encoder_output_size


class PretrainedEmbedModel(nn.Module):
    """This class acts as an embeddding layer with pre-trained model"""

    def __init__(self, cfg, vocab):
        """This function constructs `PretrainedEmbedModel` components and
        sets `PretrainedEmbedModel` parameters

        Arguments:
            cfg {dict} -- config parameters for constructing multiple models
            vocab {Vocabulary} -- vocabulary
        """

        super().__init__()
        self.activation = gelu
        self.pretrained_encoder = PretrainedEncoder(
            pretrained_model_name=cfg.pretrained_model_name,
            trainable=cfg.fine_tune,
            output_size=cfg.bert_output_size,
            activation=self.activation,
            dropout=cfg.bert_dropout,
        )
        self.encoder_output_size = self.pretrained_encoder.get_output_dims()

    def forward(self, batch_inputs):
        """This function propagetes forwardly

        Arguments:
            batch_inputs {dict} -- batch input data
        """

        if "wordpiece_segment_ids" in batch_inputs:
            batch_seq_pretrained_encoder_repr, batch_cls_repr = self.pretrained_encoder(
                batch_inputs["wordpiece_tokens"], batch_inputs["wordpiece_segment_ids"]
            )
        else:
            batch_seq_pretrained_encoder_repr, batch_cls_repr = self.pretrained_encoder(
                batch_inputs["wordpiece_tokens"]
            )

        batch_seq_tokens_encoder_repr = batched_index_select(
            batch_seq_pretrained_encoder_repr, batch_inputs["wordpiece_tokens_index"]
        )

        batch_inputs["seq_encoder_reprs"] = batch_seq_tokens_encoder_repr
        batch_inputs["seq_cls_repr"] = batch_cls_repr

    def get_hidden_size(self):
        """This function returns embedding dimensions

        Returns:
            int -- embedding dimensitons
        """

        return self.encoder_output_size
