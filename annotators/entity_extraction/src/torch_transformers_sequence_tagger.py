# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from logging import getLogger
from pathlib import Path
from typing import List, Union, Dict, Optional

import numpy as np
import torch
from torch import nn
from overrides import overrides
from transformers import AutoModelForTokenClassification, AutoConfig, AutoModel

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel

log = getLogger(__name__)


def token_from_subtoken(units: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ Assemble token level units from subtoken level units

    Args:
        units: torch.Tensor of shape [batch_size, SUBTOKEN_seq_length, n_features]
        mask: mask of token beginnings. For example: for tokens

                [[``[CLS]`` ``My``, ``capybara``, ``[SEP]``],
                [``[CLS]`` ``Your``, ``aar``, ``##dvark``, ``is``, ``awesome``, ``[SEP]``]]

            the mask will be

                [[0, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0]]

    Returns:
        word_level_units: Units assembled from ones in the mask. For the
            example above this units will correspond to the following

                [[``My``, ``capybara``],
                [``Your`, ``aar``, ``is``, ``awesome``,]]

            the shape of this tensor will be [batch_size, TOKEN_seq_length, n_features]
    """
    shape = units.size()
    batch_size = shape[0]
    nf = shape[2]
    nf_int = units.size()[-1]

    # number of TOKENS in each sentence
    token_seq_lengths = torch.sum(mask, 1).to(torch.int64)
    # for a matrix m =
    # [[1, 1, 1],
    #  [0, 1, 1],
    #  [1, 0, 0]]
    # it will be
    # [3, 2, 1]

    n_words = torch.sum(token_seq_lengths)
    # n_words -> 6

    max_token_seq_len = torch.max(token_seq_lengths)
    # max_token_seq_len -> 3

    idxs = torch.stack(torch.nonzero(mask, as_tuple=True), dim=1)
    # for the matrix mentioned above
    # tf.where(mask) ->
    # [[0, 0],
    #  [0, 1]
    #  [0, 2],
    #  [1, 1],
    #  [1, 2]
    #  [2, 0]]

    sample_ids_in_batch = torch.nn.functional.pad(input=idxs[:, 0], pad=[1, 0])
    # for indices
    # [[0, 0],
    #  [0, 1]
    #  [0, 2],
    #  [1, 1],
    #  [1, 2],
    #  [2, 0]]
    # it is
    # [0, 0, 0, 0, 1, 1, 2]
    # padding is for computing change from one sample to another in the batch

    a = torch.logical_not(torch.eq(sample_ids_in_batch[1:], sample_ids_in_batch[:-1]).to(torch.int64))
    # for the example above the result of this statement equals
    # [0, 0, 0, 1, 0, 1]
    # so data samples begin in 3rd and 5th positions (the indexes of ones)

    # transforming sample start masks to the sample starts themselves
    q = a * torch.arange(n_words).to(torch.int64)
    # [0, 0, 0, 3, 0, 5]
    count_to_substract = torch.nn.functional.pad(torch.masked_select(q, q.to(torch.bool)), [1, 0])
    # [0, 3, 5]

    new_word_indices = torch.arange(n_words).to(torch.int64) - torch.gather(
        count_to_substract, dim=0, index=torch.cumsum(a, 0))
    # tf.range(n_words) -> [0, 1, 2, 3, 4, 5]
    # tf.cumsum(a) -> [0, 0, 0, 1, 1, 2]
    # tf.gather(count_to_substract, tf.cumsum(a)) -> [0, 0, 0, 3, 3, 5]
    # new_word_indices -> [0, 1, 2, 3, 4, 5] - [0, 0, 0, 3, 3, 5] = [0, 1, 2, 0, 1, 0]
    # new_word_indices is the concatenation of range(word_len(sentence))
    # for all sentences in units

    n_total_word_elements = (batch_size * max_token_seq_len).to(torch.int32)
    word_indices_flat = (idxs[:, 0] * max_token_seq_len + new_word_indices).to(torch.int64)
    x_mask = torch.sum(torch.nn.functional.one_hot(word_indices_flat, n_total_word_elements), 0)
    x_mask = x_mask.to(torch.bool)
    # to get absolute indices we add max_token_seq_len:
    # idxs[:, 0] * max_token_seq_len -> [0, 0, 0, 1, 1, 2] * 2 = [0, 0, 0, 3, 3, 6]
    # word_indices_flat -> [0, 0, 0, 3, 3, 6] + [0, 1, 2, 0, 1, 0] = [0, 1, 2, 3, 4, 6]
    # total number of words in the batch (including paddings)
    # batch_size * max_token_seq_len -> 3 * 3 = 9
    # tf.one_hot(...) ->
    # [[1. 0. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 1. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 1. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0. 0. 1. 0. 0.]]
    #  x_mask -> [1, 1, 1, 1, 1, 0, 1, 0, 0]

    full_range = torch.arange(batch_size * max_token_seq_len).to(torch.int64)
    # full_range -> [0, 1, 2, 3, 4, 5, 6, 7, 8]
    nonword_indices_flat = torch.masked_select(full_range, torch.logical_not(x_mask))

    # # y_idxs -> [5, 7, 8]

    # get a sequence of units corresponding to the start subtokens of the words
    # size: [n_words, n_features]
    def gather_nd(params, indices):
        assert type(indices) == torch.Tensor
        return params[indices.transpose(0, 1).long().numpy().tolist()]

    elements = gather_nd(units, idxs)

    # prepare zeros for paddings
    # size: [batch_size * TOKEN_seq_length - n_words, n_features]
    sh = tuple(torch.stack([torch.sum(max_token_seq_len - token_seq_lengths), torch.tensor(nf)], 0).numpy())
    paddings = torch.zeros(sh, dtype=torch.float64)

    def dynamic_stitch(indices, data):
        # https://discuss.pytorch.org/t/equivalent-of-tf-dynamic-partition/53735/2
        n = sum(idx.numel() for idx in indices)
        res = [None] * n
        for i, data_ in enumerate(data):
            idx = indices[i].view(-1)
            if idx.numel() > 0:
                d = data_.view(idx.numel(), -1)
                k = 0
                for idx_ in idx:
                    res[idx_] = d[k].to(torch.float64)
                    k += 1
        return res

    tensor_flat = torch.stack(dynamic_stitch([word_indices_flat, nonword_indices_flat], [elements, paddings]))
    # tensor_flat -> [x, x, x, x, x, 0, x, 0, 0]

    tensor = torch.reshape(tensor_flat, (batch_size, max_token_seq_len.item(), nf_int))
    # tensor -> [[x, x, x],
    #            [x, x, 0],
    #            [x, 0, 0]]

    return tensor


def token_labels_to_subtoken_labels(labels, y_mask, input_mask):
    subtoken_labels = []
    labels_ind = 0
    n_tokens_with_special = int(np.sum(input_mask))

    for el in y_mask[1:n_tokens_with_special - 1]:
        if el == 1:
            subtoken_labels += [labels[labels_ind]]
            labels_ind += 1
        else:
            subtoken_labels += [labels[labels_ind - 1]]

    subtoken_labels = [0] + subtoken_labels + [0] * (len(input_mask) - n_tokens_with_special + 1)
    return subtoken_labels


class AutoModelForTwoHeadTokenClassification(nn.Module):
    def __init__(self, pretrained_bert: str, config: AutoConfig, num_tags_seq: int, num_tags_ent: int,
                       encoder_path: str, seq_linear_path: str, ent_linear_path: str, device: str = "gpu",
                       using_custom_types: bool = False, classifier_dropout: float = 0.1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.pretrained_bert = pretrained_bert
        self.encoder = AutoModel.from_pretrained(self.pretrained_bert, config=config).to(self.device)
        self.dropout = nn.Dropout(classifier_dropout)
        self.num_tags_seq = num_tags_seq
        self.num_tags_ent = num_tags_ent
        self.config = config
        self.fc_seq = nn.Linear(self.config.hidden_size, self.num_tags_seq).to(self.device)
        self.fc_ent = nn.Linear(self.config.hidden_size, self.num_tags_ent).to(self.device)
        self.encoder_path = encoder_path
        self.seq_linear_path = seq_linear_path
        self.ent_linear_path = ent_linear_path
        self.using_custom_types = using_custom_types
    
    def forward(self, input_ids, attention_mask, seq_labels=None, ent_labels=None):
        transf_output = self.encoder(input_ids, attention_mask, output_attentions=True)
        output = transf_output.last_hidden_state
        output = self.dropout(output)
        bs, seq_len, emb = output.size()
        
        seq_logits = self.fc_seq(output)
        ent_logits = self.fc_ent(output)
        
        loss = None
        if seq_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            if attention_mask is not None:
                seq_active_loss = attention_mask.view(-1) == 1
                seq_active_logits = seq_logits.view(-1, self.num_tags_seq)
                seq_active_labels = torch.where(
                    seq_active_loss, seq_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(seq_labels)
                )
                seq_loss = loss_fct(seq_active_logits, seq_active_labels)
                
                ent_active_loss = ent_labels.view(-1) > 0
                ent_labels = ent_labels - 1
                ent_active_logits = ent_logits.view(-1, self.num_tags_ent)
                ent_active_labels = torch.where(
                    ent_active_loss, ent_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(ent_labels)
                )
                ent_loss = loss_fct(ent_active_logits, ent_active_labels)
            if self.using_custom_types:
                loss = ent_loss
            else:
                loss = seq_loss + ent_loss
            return loss
        else:
            return seq_logits, ent_logits
    
    def load(self):
        encoder_path = Path(self.encoder_path).expanduser().resolve()
        encoder_path = encoder_path.with_suffix(f".pth.tar")
        if encoder_path.exists():
            log.info(f"Load path {encoder_path} exists.")
            log.info(f"Initializing `{self.__class__.__name__}` from saved.")
            log.info(f"Loading weights from {encoder_path}.")
            checkpoint = torch.load(encoder_path, map_location=self.device)
            self.encoder.load_state_dict(checkpoint["model_state_dict"])
        else:
            log.info(f"Init from scratch. Load path {encoder_path} does not exist.")
        
        seq_linear_path = Path(self.seq_linear_path).expanduser().resolve()
        seq_linear_path = seq_linear_path.with_suffix(f".pth.tar")
        if seq_linear_path.exists():
            log.info(f"Load path {seq_linear_path} exists.")
            log.info(f"Initializing `{self.__class__.__name__}` from saved.")
            log.info(f"Loading weights from {seq_linear_path}.")
            checkpoint = torch.load(seq_linear_path, map_location=self.device)
            self.fc_seq.load_state_dict(checkpoint["model_state_dict"])
        else:
            log.info(f"Init from scratch. Load path {seq_linear_path} does not exist.")
        
        try:
            ent_linear_path = Path(self.ent_linear_path).expanduser().resolve()
            ent_linear_path = ent_linear_path.with_suffix(f".pth.tar")
            if ent_linear_path.exists():
                log.info(f"Load path {ent_linear_path} exists.")
                log.info(f"Initializing `{self.__class__.__name__}` from saved.")
                log.info(f"Loading weights from {ent_linear_path}.")
                checkpoint = torch.load(ent_linear_path, map_location=self.device)
                self.fc_ent.load_state_dict(checkpoint["model_state_dict"])
            else:
                log.info(f"Init from scratch. Load path {ent_linear_path} does not exist.")
        except:
            self.fc_ent = nn.Linear(self.config.hidden_size, self.num_tags_ent).to(self.device)
    
    def save(self):
        encoder_path = Path(self.encoder_path).expanduser().resolve()
        encoder_path = encoder_path.with_suffix(f".pth.tar")
        log.info(f"Saving model to {encoder_path}.")
        torch.save({"model_state_dict": self.encoder.cpu().state_dict()}, encoder_path)
        self.encoder.to(self.device)
        
        if not self.using_custom_types:
            seq_linear_path = Path(self.seq_linear_path).expanduser().resolve()
            seq_linear_path = seq_linear_path.with_suffix(f".pth.tar")
            log.info(f"Saving model to {seq_linear_path}.")
            torch.save({"model_state_dict": self.fc_seq.cpu().state_dict()}, seq_linear_path)
            self.fc_seq.to(self.device)
        
        ent_linear_path = Path(self.ent_linear_path).expanduser().resolve()
        ent_linear_path = ent_linear_path.with_suffix(f".pth.tar")
        log.info(f"Saving model to {ent_linear_path}.")
        torch.save({"model_state_dict": self.fc_ent.cpu().state_dict()}, ent_linear_path)
        self.fc_ent.to(self.device)


@register('torch_transformers_sequence_tagger')
class TorchTransformersSequenceTagger(TorchModel):
    """Transformer-based model on PyTorch for text tagging. It predicts a label for every token (not subtoken)
    in the text. You can use it for sequence labeling tasks, such as morphological tagging or named entity recognition.

    Args:
        n_tags: number of distinct tags
        pretrained_bert: pretrained Bert checkpoint path or key title (e.g. "bert-base-uncased")
        return_probas: set this to `True` if you need the probabilities instead of raw answers
        bert_config_file: path to Bert configuration file, or None, if `pretrained_bert` is a string name
        attention_probs_keep_prob: keep_prob for Bert self-attention layers
        hidden_keep_prob: keep_prob for Bert hidden layers
        optimizer: optimizer name from `torch.optim`
        optimizer_parameters: dictionary with optimizer's parameters,
                              e.g. {'lr': 0.1, 'weight_decay': 0.001, 'momentum': 0.9}
        learning_rate_drop_patience: how many validations with no improvements to wait
        learning_rate_drop_div: the divider of the learning rate after `learning_rate_drop_patience` unsuccessful
            validations
        load_before_drop: whether to load best model before dropping learning rate or not
        clip_norm: clip gradients by norm
        min_learning_rate: min value of learning rate if learning rate decay is used
    """

    def __init__(self,
                 n_tags: int,
                 pretrained_bert: str,
                 encoder_path: str,
                 seq_linear_path: str,
                 ent_linear_path: str,
                 using_custom_types: bool = False,
                 n_tags_ent: int = None,
                 bert_config_file: Optional[str] = None,
                 return_probas: bool = False,
                 attention_probs_keep_prob: Optional[float] = None,
                 hidden_keep_prob: Optional[float] = None,
                 optimizer: str = "AdamW",
                 optimizer_parameters: dict = {"lr": 1e-3, "weight_decay": 1e-6},
                 learning_rate_drop_patience: int = 20,
                 learning_rate_drop_div: float = 2.0,
                 load_before_drop: bool = True,
                 clip_norm: Optional[float] = None,
                 min_learning_rate: float = 1e-07,
                 two_heads: bool = False,
                 device: str = "gpu",
                 **kwargs) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.n_classes = n_tags
        self.n_tags_ent = n_tags_ent - 1
        self.return_probas = return_probas
        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob
        self.clip_norm = clip_norm
        self.two_heads = two_heads

        self.pretrained_bert = pretrained_bert
        self.encoder_path = encoder_path
        self.seq_linear_path = seq_linear_path
        self.ent_linear_path = ent_linear_path
        self.using_custom_types = using_custom_types
        self.bert_config_file = bert_config_file

        super().__init__(optimizer=optimizer,
                         optimizer_parameters=optimizer_parameters,
                         learning_rate_drop_patience=learning_rate_drop_patience,
                         learning_rate_drop_div=learning_rate_drop_div,
                         load_before_drop=load_before_drop,
                         min_learning_rate=min_learning_rate,
                         **kwargs)
        self.model.to(self.device)

    def train_on_batch(self,
                       input_ids: Union[List[List[int]], np.ndarray],
                       input_masks: Union[List[List[int]], np.ndarray],
                       y_masks: Union[List[List[int]], np.ndarray],
                       seq_y: List[List[int]],
                       ent_y: List[List[int]],
                       *args, **kwargs) -> Dict[str, float]:
        """

        Args:
            input_ids: batch of indices of subwords
            input_masks: batch of masks which determine what should be attended
            args: arguments passed  to _build_feed_dict
                and corresponding to additional input
                and output tensors of the derived class.
            kwargs: keyword arguments passed to _build_feed_dict
                and corresponding to additional input
                and output tensors of the derived class.

        Returns:
            dict with fields 'loss', 'head_learning_rate', and 'bert_learning_rate'
        """
        b_input_ids = torch.from_numpy(input_ids).to(self.device)
        b_input_masks = torch.from_numpy(input_masks).to(self.device)
        seq_subtoken_labels = [token_labels_to_subtoken_labels(y_el, y_mask, input_mask)
                                for y_el, y_mask, input_mask in zip(seq_y, y_masks, input_masks)]
        seq_b_labels = torch.from_numpy(np.array(seq_subtoken_labels)).to(torch.int64).to(self.device)

        self.optimizer.zero_grad()
        if self.two_heads:
            ent_subtoken_labels = [token_labels_to_subtoken_labels(y_el, y_mask, input_mask)
                                for y_el, y_mask, input_mask in zip(ent_y, y_masks, input_masks)]
            ent_b_labels = torch.from_numpy(np.array(ent_subtoken_labels)).to(torch.int64).to(self.device)
            loss = self.model(input_ids=b_input_ids, attention_mask=b_input_masks,
                              seq_labels=seq_b_labels, ent_labels=ent_b_labels)
        else:
            loss = self.model(input_ids=b_input_ids, attention_mask=b_input_masks, labels=seq_b_labels).loss
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {'loss': loss.item()}

    def __call__(self,
                 input_ids: Union[List[List[int]], np.ndarray],
                 input_masks: Union[List[List[int]], np.ndarray],
                 y_masks: Union[List[List[int]], np.ndarray]) -> Union[List[List[int]], List[np.ndarray]]:
        """ Predicts tag indices for a given subword tokens batch

        Args:
            input_ids: indices of the subwords
            input_masks: mask that determines where to attend and where not to
            y_masks: mask which determines the first subword units in the the word

        Returns:
            Label indices or class probabilities for each token (not subtoken)

        """
        b_input_ids = torch.from_numpy(input_ids).to(self.device)
        b_input_masks = torch.from_numpy(input_masks).to(self.device)

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = self.model(b_input_ids, attention_mask=b_input_masks)

            # Move logits and labels to CPU and to numpy arrays
            seq_logits = token_from_subtoken(logits[0].detach().cpu(), torch.from_numpy(y_masks))
            if self.two_heads:
                ent_logits = token_from_subtoken(logits[1].detach().cpu(), torch.from_numpy(y_masks))

        seq_lengths = np.sum(y_masks, axis=1)
        
        seq_probas = torch.nn.functional.softmax(seq_logits, dim=-1)
        seq_probas = seq_probas.detach().cpu().numpy()
        seq_logits = seq_logits.detach().cpu().numpy()
        seq_pred = np.argmax(seq_logits, axis=-1).tolist()
        seq_pred = [p[:l] for l, p in zip(seq_lengths, seq_pred)]
        
        if self.two_heads:
            ent_pred = torch.nn.functional.softmax(ent_logits, dim=-1)
            ent_pred = ent_pred.detach().cpu().numpy()
            ent_pred = [p[:l] for l, p in zip(seq_lengths, ent_pred)]
            return seq_pred, ent_pred
        else:
            if self.return_probas:
                return seq_probas
            else:
                return seq_pred

    @overrides
    def load(self, fname=None):
        if fname is not None:
            self.load_path = fname

        if self.pretrained_bert:
            config = AutoConfig.from_pretrained(self.pretrained_bert, num_labels=self.n_classes,
                                                output_attentions=False, output_hidden_states=False)
            if self.two_heads:
                self.model = AutoModelForTwoHeadTokenClassification(self.pretrained_bert, config, self.n_classes,
                    self.n_tags_ent, self.encoder_path, self.seq_linear_path, self.ent_linear_path, self.using_custom_types)
            else:
                self.model = AutoModelForTokenClassification.from_pretrained(self.pretrained_bert, config=config)
        elif self.bert_config_file and Path(self.bert_config_file).is_file():
            self.bert_config = AutoConfig.from_json_file(str(expand_path(self.bert_config_file)))

            if self.attention_probs_keep_prob is not None:
                self.bert_config.attention_probs_dropout_prob = 1.0 - self.attention_probs_keep_prob
            if self.hidden_keep_prob is not None:
                self.bert_config.hidden_dropout_prob = 1.0 - self.hidden_keep_prob
            self.model = AutoModelForTokenClassification(config=self.bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")

        self.model.to(self.device)

        self.optimizer = getattr(torch.optim, self.optimizer_name)(
            self.model.parameters(), **self.optimizer_parameters)
        if self.lr_scheduler_name is not None:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                self.optimizer, **self.lr_scheduler_parameters)

        self.model.load()
    
    def save(self, fname=None):
        self.model.save()
        self.model.to(self.device)
