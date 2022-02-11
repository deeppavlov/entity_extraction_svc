import itertools
from pathlib import Path
from logging import getLogger
from typing import List, Optional, Dict, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
# from apex import amp

from deeppavlov.core.commands.utils import expand_path
from transformers import AutoConfig, AutoTokenizer, AutoModel
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel
from src.torch_transformers_preprocessor import TorchTransformersEntityRankerPreprocessor

log = getLogger(__name__)


@register('torch_transformers_el_ranker')
class TorchTransformersElRanker(TorchModel):

    def __init__(
            self,
            model_name: str,
            encoder_save_path: str,
            bilinear_save_path: str,
            pretrained_bert: str = None,
            bert_config_file: Optional[str] = None,
            criterion: str = "CrossEntropyLoss",
            optimizer: str = "AdamW",
            optimizer_parameters: Dict = {"lr": 5e-5, "weight_decay": 0.01, "eps": 1e-6},
            return_probas: bool = False,
            attention_probs_keep_prob: Optional[float] = None,
            hidden_keep_prob: Optional[float] = None,
            clip_norm: Optional[float] = None,
            threshold: Optional[float] = None,
            **kwargs
    ):
        self.encoder_save_path = encoder_save_path
        self.bilinear_save_path = bilinear_save_path
        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file
        self.return_probas = return_probas
        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob
        self.clip_norm = clip_norm

        super().__init__(
            model_name=model_name,
            optimizer=optimizer,
            criterion=criterion,
            optimizer_parameters=optimizer_parameters,
            return_probas=return_probas,
            **kwargs)

    def train_on_batch(self, q_features: List[Dict],
                             c_features: List[Dict],
                             entity_tokens_pos: List[int],
                             labels: List[int]) -> float:

        _input = {'labels': labels}
        _input['entity_tokens_pos'] = entity_tokens_pos
        for elem in ['input_ids', 'attention_mask']:
            inp_elem = [getattr(f, elem) for f in q_features]
            _input[f"q_{elem}"] = torch.LongTensor(inp_elem).to(self.device)
        for elem in ['input_ids', 'attention_mask']:
            inp_elem = [getattr(f, elem) for f in c_features]
            _input[f"c_{elem}"] = torch.LongTensor(inp_elem).to(self.device)

        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()      # zero the parameter gradients

        loss, softmax_scores = self.model(**_input)
        loss.backward()
        self.optimizer.step()

        # Clip the norm of the gradients to prevent the "exploding gradients" problem
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item()

    def __call__(self, q_features: List[Dict],
                       c_features: List[Dict],
                       entity_tokens_pos: List[int]) -> Union[List[int], List[np.ndarray]]:

        self.model.eval()

        _input = {'entity_tokens_pos': entity_tokens_pos}
        for elem in ['input_ids', 'attention_mask']:
            inp_elem = [getattr(f, elem) for f in q_features]
            _input[f"q_{elem}"] = torch.LongTensor(inp_elem).to(self.device)
        for elem in ['input_ids', 'attention_mask']:
            inp_elem = [getattr(f, elem) for f in c_features]
            _input[f"c_{elem}"] = torch.LongTensor(inp_elem).to(self.device)

        with torch.no_grad():
            softmax_scores = self.model(**_input)
            if self.return_probas:
                pred = softmax_scores
            else:
                pred = torch.argmax(softmax_scores, dim=1).cpu().numpy()
            
        return pred

    def siamese_ranking_el_model(self, **kwargs) -> nn.Module:
        return SiameseBertElModel(
            pretrained_bert=self.pretrained_bert,
            encoder_save_path=self.encoder_save_path,
            bilinear_save_path=self.bilinear_save_path,
            bert_tokenizer_config_file=self.pretrained_bert,
            device=self.device
        )
        
    def save(self, fname: Optional[str] = None, *args, **kwargs) -> None:
        if fname is None:
            fname = self.save_path
        if not fname.parent.is_dir():
            raise ConfigError("Provided save path is incorrect!")
        weights_path = Path(fname).with_suffix(f".pth.tar")
        log.info(f"Saving model to {weights_path}.")
        torch.save({
            "model_state_dict": self.model.cpu().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epochs_done": self.epochs_done
        }, weights_path)
        self.model.to(self.device)
        self.model.save()


class TextEncoder(nn.Module):
    def __init__(self, pretrained_bert: str = None,
                 bert_tokenizer_config_file: str = None,
                 bert_config_file: str = None,
                 resize: bool = False,
                 device: str = "gpu"
                 ):
        super().__init__()
        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file
        self.encoder, self.config, self.bert_config = None, None, None
        self.device = device
        self.load()
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_bert)
        self.encoder.resize_token_embeddings(len(self.tokenizer) + 1)
        
    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor,
                entity_tokens_pos: List[int] = None
    ) -> Union[Tuple[Any, Tensor], Tuple[Tensor]]:

        if entity_tokens_pos is not None:
            q_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            q_hidden_states = q_outputs.last_hidden_state
            
            entity_emb = []
            for i in range(len(entity_tokens_pos)):
                pos = entity_tokens_pos[i]
                entity_emb.append(q_hidden_states[i, pos])
            
            entity_emb = torch.stack(entity_emb, dim=0)
            return entity_emb
        else:
            c_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            c_cls_emb = c_outputs.last_hidden_state[:,:1,:].squeeze(1)
            return c_cls_emb
            
    def load(self) -> None:
        if self.pretrained_bert:
            log.info(f"From pretrained {self.pretrained_bert}.")
            self.config = AutoConfig.from_pretrained(
                self.pretrained_bert, output_hidden_states=True
            )
            self.encoder = AutoModel.from_pretrained(self.pretrained_bert, config=self.config)

        elif self.bert_config_file and Path(self.bert_config_file).is_file():
            self.config = AutoConfig.from_json_file(str(expand_path(self.bert_config_file)))
            self.encoder = AutoModel.from_config(config=self.bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")
        self.encoder.to(self.device)


class BilinearRanking(nn.Module):
    def __init__(self, n_classes: int = 2, emb_size: int = 768, block_size: int = 8):
        super().__init__()
        self.n_classes = n_classes
        self.emb_size = emb_size
        self.block_size = block_size
        self.bilinear = nn.Linear(self.emb_size * self.block_size, self.n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text1: Tensor, text2: Tensor):
        b1 = text1.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = text2.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)
        softmax_logits = self.softmax(logits)
        log_softmax = F.log_softmax(logits, dim=-1)
        return softmax_logits, log_softmax


class SiameseBertElModel(nn.Module):

    def __init__(
            self,
            encoder_save_path: str,
            bilinear_save_path: str,
            pretrained_bert: str = None,
            bert_tokenizer_config_file: str = None,
            bert_config_file: str = None,
            device: str = "gpu"
    ):
        super().__init__()
        self.pretrained_bert = pretrained_bert
        self.encoder_save_path = encoder_save_path
        self.bilinear_save_path = bilinear_save_path
        self.bert_config_file = bert_config_file
        self.device = device

        # initialize parameters that would be filled later
        self.encoder = TextEncoder(pretrained_bert=self.pretrained_bert, device=self.device)
        self.bilinear_ranker = BilinearRanking(emb_size=264, block_size=6)

    def forward(
            self,
            q_input_ids: Tensor,
            q_attention_mask: Tensor,
            c_input_ids: Tensor,
            c_attention_mask: Tensor,
            entity_tokens_pos: List,
            labels: List[int] = None
    ) -> Union[Tuple[Any, Tensor], Tuple[Tensor]]:

        entity_emb = self.encoder(input_ids=q_input_ids, attention_mask=q_attention_mask, entity_tokens_pos=entity_tokens_pos)
        c_cls_emb = self.encoder(input_ids=c_input_ids, attention_mask=c_attention_mask)
        softmax_scores, log_softmax = self.bilinear_ranker(entity_emb, c_cls_emb)
        
        if labels is not None:
            labels_one_hot = [[0.0, 0.0] for _ in labels]
            for i in range(len(labels)):
                labels_one_hot[i][labels[i]] = 1.0
            labels_one_hot = torch.Tensor(labels_one_hot).to(self.device)
            
            bs, dim = labels_one_hot.shape
            per_sample_loss = -torch.bmm(labels_one_hot.view(bs, 1, dim), log_softmax.view(bs, dim, 1)).squeeze(2).squeeze(1)
            loss = torch.mean(per_sample_loss)
            return loss, softmax_scores
        else:
            return softmax_scores
        
    def save(self) -> None:
        encoder_weights_path = expand_path(self.encoder_save_path).with_suffix(f".pth.tar")
        log.info(f"Saving encoder to {encoder_weights_path}.")
        torch.save({"model_state_dict": self.encoder.cpu().state_dict()}, encoder_weights_path)
        bilinear_weights_path = expand_path(self.bilinear_save_path).with_suffix(f".pth.tar")
        log.info(f"Saving bilinear weights to {bilinear_weights_path}.")
        torch.save({"model_state_dict": self.bilinear_ranker.cpu().state_dict()}, bilinear_weights_path)
        self.encoder.to(self.device)
        self.bilinear_ranker.to(self.device)


@register('torch_transformers_entity_ranker_infer')
class TorchTransformersEntityRankerInfer:
    def __init__(self, pretrained_bert,
                       encoder_weights_path,
                       bilinear_weights_path,
                       special_token_id: int,
                       do_lower_case: bool = False,
                       batch_size: int = 5,
                       emb_size: int = 300,
                       block_size: int = 8,
                       device: str = "cpu", **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.pretrained_bert = str(expand_path(pretrained_bert))
        self.preprocessor = TorchTransformersEntityRankerPreprocessor(vocab_file=self.pretrained_bert,
                                                                      do_lower_case=do_lower_case,
                                                                      special_tokens=["[ENT]"])
        self.encoder, self.config = None, None
        self.config = AutoConfig.from_pretrained(self.pretrained_bert, output_hidden_states=True)
        self.emb_size = emb_size
        self.block_size = block_size
        self.encoder = TextEncoder(pretrained_bert=self.pretrained_bert, device=self.device)
        self.encoder_weights_path = str(expand_path(encoder_weights_path))
        self.bilinear_weights_path = str(expand_path(bilinear_weights_path))
        encoder_checkpoint = torch.load(self.encoder_weights_path, map_location=self.device)
        self.encoder.load_state_dict(encoder_checkpoint["model_state_dict"])
        self.encoder.to(self.device)
        self.bilinear_ranking = BilinearRanking(emb_size=self.emb_size, block_size=self.block_size)
        bilinear_checkpoint = torch.load(self.bilinear_weights_path, map_location=self.device)
        self.bilinear_ranking.load_state_dict(bilinear_checkpoint["model_state_dict"])
        self.bilinear_ranking.to(self.device)
        self.special_token_id = special_token_id
        self.batch_size = batch_size

    def __call__(self, contexts_batch: List[str],
                       candidate_entities_batch: List[List[str]],
                       candidate_entities_descr_batch: List[List[str]]):
        entity_emb_batch = []
        
        num_batches = len(contexts_batch) // self.batch_size + int(len(contexts_batch) % self.batch_size > 0)
        for ii in range(num_batches):
            contexts_list = contexts_batch[ii*self.batch_size:(ii+1)*self.batch_size]
            context_features = self.preprocessor(contexts_list)
            context_input_ids = context_features["input_ids"]
            context_attention_mask = context_features["attention_mask"]
            special_tokens_pos = []
            for input_ids_list in context_input_ids:
                found_n = -1
                for n, input_id in enumerate(input_ids_list):
                    if input_id == self.special_token_id:
                        found_n = n
                        break
                if found_n == -1:
                    found_n = 0
                special_tokens_pos.append(found_n)
            
            cur_entity_emb_batch = self.encoder(input_ids=context_input_ids,
                                                attention_mask=context_attention_mask,
                                                entity_tokens_pos=special_tokens_pos)
        
            entity_emb_batch += cur_entity_emb_batch.detach().cpu().numpy().tolist()
            
        scores_batch = []
        for entity_emb, candidate_entities_list, candidate_entities_descr_list in \
                zip(entity_emb_batch, candidate_entities_batch, candidate_entities_descr_batch):
            if candidate_entities_list:
                entity_emb = [entity_emb for _ in candidate_entities_list]
                entity_emb = torch.Tensor(entity_emb).to(self.device)
                descr_features = self.preprocessor(candidate_entities_descr_list)
                descr_input_ids = descr_features["input_ids"]
                descr_attention_mask = descr_features["attention_mask"]
                candidate_entities_emb = self.encoder(input_ids=descr_input_ids,
                                                      attention_mask=descr_attention_mask)
                scores_list, _ = self.bilinear_ranking(entity_emb, candidate_entities_emb)
                scores_list = scores_list.detach().cpu().numpy()
                scores_list = [score[1] for score in scores_list]
                entities_with_scores = [(entity, score) for entity, score in zip(candidate_entities_list, scores_list)]
                entities_with_scores = sorted(entities_with_scores, key=lambda x: x[1], reverse=True)
                scores_batch.append(entities_with_scores)
            else:
                scores_batch.append([])

        return scores_batch
