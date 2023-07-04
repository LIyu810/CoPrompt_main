from argparse import ArgumentParser
from json import decoder
from logging import debug
import pytorch_lightning as pl
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
# Hide lines below until Lab 5
import wandb
import numpy as np
# Hide lines above until Lab 5
import os
from .base import BaseLitModel
from .util import f1_eval, compute_f1, acc, f1_score
from transformers.optimization import get_linear_schedule_with_warmup

from functools import partial

import random

def mask_hook(grad_input, st, ed):
    mask = torch.zeros((grad_input.shape[0], 1)).type_as(grad_input)
    mask[st: ed] += 1.0  # 只优化id为1～8的token
    # for the speaker unused token12
    mask[1:3] += 1.0
    return grad_input * mask

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()




class BertLitModel(BaseLitModel):
    """
    use AutoModelForMaskedLM, and select the output by another layer in the lit model
    """
    def __init__(self, model, args, tokenizer):
        super().__init__(model, args)
        self.tokenizer = tokenizer
        
        with open(f"{args.data_dir}/rel2id.json","r") as file:
            rel2id = json.load(file)
        
        Na_num = 0
        for k, v in rel2id.items():
            if k == "NA" or k == "no_relation" or k == "unanswerable":
                Na_num = v
                break
        num_relation = len(rel2id)
        # init loss function
        self.loss_fn = multilabel_categorical_crossentropy if "dialogue" in args.data_dir else nn.CrossEntropyLoss()
        # self.loss_fn = AMSoftmax(self.model.config.hidden_size, num_relation)
        # ignore the no_relation class to compute the f1 score
        self.eval_fn = f1_eval if "dialogue" in args.data_dir else partial(f1_score, rel_num=num_relation, na_num=Na_num)
        self.best_f1 = 0
        self.t_lambda = args.t_lambda
        
        self.label_st_id = tokenizer("[class1]", add_special_tokens=False)['input_ids'][0]
        self.tokenizer = tokenizer
    
        self._init_label_word()
        
        # with torch.no_grad():
        #     self.loss_fn.fc.weight = nn.Parameter(self.model.get_output_embeddings().weight[self.label_st_id:self.label_st_id+num_relation])
            # self.loss_fn.fc.bias = nn.Parameter(self.model.get_output_embeddings().bias[self.label_st_id:self.label_st_id+num_relation])

    def _init_label_word(self, ):
        args = self.args
        # ./dataset/dataset_name
        dataset_name = args.data_dir.split("/")[1]
        model_name_or_path = args.model_name_or_path.split("/")[-1]
        label_path = f"./dataset/{os.path.split(model_name_or_path)[-1]}_{dataset_name}.pt"
        # [num_labels, num_tokens], ignore the unanswerable
        if "dialogue" in args.data_dir:
            label_word_idx = torch.load(label_path)[:30]
        else:
            label_word_idx = torch.load(label_path)
        
        num_labels = len(label_word_idx)
        
        self.model.resize_token_embeddings(len(self.tokenizer))         # 加入label_word Resize一下大小
        
        with torch.no_grad():
            word_embeddings = self.model.get_input_embeddings()
            # add label_word to vocab.txt
            continous_label_word = [a[0] for a in self.tokenizer([f"[class{i}]" for i in range(1, num_labels+1)], add_special_tokens=False)['input_ids']]       # label_word_idx --> [36,11]    word_embedding ---> (21215,768)
            
            # for abaltion study
            if self.args.init_answer_words:   # none
                if self.args.init_answer_words_by_one_token:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = word_embeddings.weight[idx][-1]
                else:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = torch.mean(word_embeddings.weight[idx], dim=0)
                # word_embeddings.weight[continous_label_word[i]] = self.relation_embedding[i]


            if self.args.init_type_words:       # none
                so_word = [a[0] for a in self.tokenizer(["[obj]","[sub]"], add_special_tokens=False)['input_ids']]
                meaning_word = [a[0] for a in self.tokenizer(["person","organization", "location", "date", "country"], add_special_tokens=False)['input_ids']]
            
                for i, idx in enumerate(so_word):
                    word_embeddings.weight[so_word[i]] = torch.mean(word_embeddings.weight[meaning_word], dim=0)




            assert torch.equal(self.model.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.model.get_input_embeddings().weight, self.model.get_output_embeddings().weight)
        print(continous_label_word)
        self.word2label = continous_label_word # a continous list
            
                
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, so = batch
        result = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        logits = result.logits
        output_embedding = result.hidden_states[-1]
        logits = self.pvp(logits, input_ids)
        # logits = self.model.roberta(input_ids, attention_mask).last_hidden_state
        # loss = self.get_loss(logits, input_ids, labels)

        ke_loss = self.ke_loss(output_embedding, labels, so, input_ids)
        loss = self.loss_fn(logits, labels) + self.t_lambda * ke_loss
        self.log("Train/loss", loss)
        self.log("Train/ke_loss", loss)
        return loss
    
    def get_loss(self, logits, input_ids, labels):
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        
        loss = self.loss_fn(mask_output, labels)
        return loss


    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, _ = batch
        logits = self.model(input_ids, attention_mask, return_dict=True).logits
        # logits = self.model.roberta(input_ids, attention_mask).last_hidden_state
        # loss = self.loss_fn(logits, labels)
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, _ = batch
        logits = self.model(input_ids, attention_mask, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)
        


    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--t_lambda", type=float, default=0.01, help="")
        parser.add_argument("--t_gamma", type=float, default=0.3, help="")
        return parser
        
    def pvp(self, logits, input_ids):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        #! hard coded
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"
        final_output = mask_output[:,self.word2label]
        
        return final_output


    def ke_loss(self, logits, labels, input_ids):
        bsz = logits.shape[0]
        labels_id = labels.cpu().numpy()
        labels_id = np.argmax(labels_id, axis=1)
        for i in range(bsz):
            labels_id[i] = labels_id[i] + self.label_st_id
        rand_id = np.random.randint(21128, 21157, bsz)
        for i in range(bsz):
            if labels_id[i] == rand_id[i] and labels_id[i] == 21128:
                rand_id[i] = rand_id[i]+1
            elif labels_id[i] == rand_id[i]:
                rand_id[i] = rand_id[i]-1

        labels_id = torch.from_numpy(labels_id)
        rand_id = torch.from_numpy(rand_id)

        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)        # mask_idx 对应[mask]在input中的位置
        mask_output = logits[torch.arange(bsz), mask_idx]
        mask_relation_embedding = mask_output

        real_relation_embedding = self.model.get_output_embeddings().weight[labels_id]
        rand_relation_embedding = self.model.get_output_embeddings().weight[rand_id]

        d_1 = F.cosine_similarity(real_relation_embedding,mask_relation_embedding,dim=1)
        d_1 = sum(d_1) / bsz
        d_2 = F.cosine_similarity(real_relation_embedding,rand_relation_embedding,dim=1)
        d_2 = sum(d_2) / bsz
        f = torch.nn.LogSigmoid()
        loss = -1. * f(d_1 - self.args.t_gamma) - f(self.args.t_gamma - d_2)
        return loss




    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        if not self.args.two_steps: 
            parameters = self.model.named_parameters()
        else:
            # model.bert.embeddings.weight
            parameters = [next(self.model.named_parameters())]
        # only optimize the embedding parameters
        optimizer_group_parameters = [
            {"params": [p for n, p in parameters if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in parameters if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }

class TransformerLitModelTwoSteps(BertLitModel):
    def configure_optimizers(self):
        no_decay_param = ["bais", "LayerNorm.weight"]
        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.args.lr_2, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }


# CRECIL
class DialogueLitModel(BertLitModel):

    def training_step(self, batch, batch_idx):                                  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids , labels = batch
        result = self.model(input_ids, attention_mask, token_type_ids, return_dict=True, output_hidden_states=True)
        logits = result.logits
        logits = self.pvp(logits, input_ids)
        output_embedding = result.hidden_states[-1]
        ke_loss = self.ke_loss(output_embedding, labels, input_ids)
        loss = self.loss_fn(logits, labels) + self.t_lambda * ke_loss
        self.log("Train/loss", loss)
        return loss



    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument

        input_ids, attention_mask, token_type_ids , labels = batch              # labels ---> [6,30]
        logits = self.model(input_ids, attention_mask, token_type_ids, return_dict=True).logits      # [6,512,21258]

        logits = self.pvp(logits, input_ids)                # [6,30]
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}

    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, token_type_ids , labels = batch
        logits = self.model(input_ids, attention_mask, token_type_ids, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)



    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--t_lambda", type=float, default=0.01, help="")
        parser.add_argument("--t_gamma", type=float, default=0.3, help="")

        return parser

    def pvp(self, logits, input_ids):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        #! hard coded
        _, mask_idx = (input_ids == 103).nonzero(as_tuple=True)
        bs = input_ids.shape[0]

        mask_output = logits[torch.arange(bs), mask_idx]                    # （6，512，21209） --> （6,21209）

        assert mask_idx.shape[0] == bs, "only one mask in sequence!"        # bs --> 6

        final_output = mask_output[:,self.word2label]       # （6，30） 截取最后30个表示类别
        # final_output = mask_output
        return final_output             # [6,30]




def decode(tokenizer, output_ids):
    return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output_ids]

class GPTLitModel(BaseLitModel):
    def __init__(self, model, args , data_config):
        super().__init__(model, args)
        # self.num_training_steps = data_config["num_training_steps"]
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = multilabel_categorical_crossentropy
        self.best_f1 = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, cls_idx , labels = batch
        logits = self.model(input_ids, attention_mask=attention_mask, mc_token_ids=cls_idx)
        if not isinstance(logits, torch.Tensor):
            logits = logits.mc_logits

        loss = self.loss_fn(logits, labels)
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, cls_idx , labels = batch
        logits = self.model(input_ids, attention_mask=attention_mask, mc_token_ids=cls_idx)
        if not isinstance(logits, torch.Tensor):
            logits = logits.mc_logits
        loss = self.loss_fn(logits, labels)
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}
    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        # f1 = compute_f1(logits, labels)["f1"]
        f1 = f1_score(logits, labels)
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argumenT
        input_ids, attention_mask, cls_idx , labels = batch
        logits = self.model(input_ids, attention_mask=attention_mask, mc_token_ids=cls_idx)
        if not isinstance(logits, torch.Tensor):
            logits = logits.mc_logits
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = f1_score(logits, labels)
        # f1 = acc(logits, labels)
        self.log("Test/f1", f1)

from models.trie import get_trie
class BartRELitModel(BaseLitModel):
    def __init__(self, model, args, tokenizer=None):
        super().__init__(model, args)
        self.best_f1 = 0
        self.first = True

        with open(f"{args.data_dir}/rel2id.json","r") as file:
            rel2id = json.load(file)

        Na_num = 0
        for k, v in rel2id.items():
            if k == "NA" or k == "no_relation" or k == "unanswerable":
                Na_num = v
                break
        num_relation = len(rel2id)
        # init loss function
        self.loss_fn = multilabel_categorical_crossentropy if "dialogue" in args.data_dir else nn.CrossEntropyLoss()
        # ignore the no_relation class to compute the f1 score
        self.eval_fn = f1_eval if "dialogue" in args.data_dir else partial(f1_score, rel_num=num_relation, na_num=Na_num)
        
        self.tokenizer = tokenizer
        self.trie, self.rel2id = get_trie(args, tokenizer=tokenizer)
        
        self.decode = partial(decode, tokenizer=self.tokenizer)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        real_label  = batch.pop("label")
        loss = self.model(**batch).loss
        self.log("Train/loss", loss)
        return loss
        
        

    def validation_step(self, batch, batch_idx):
        real_label = batch.pop("label")
        labels = batch.pop("labels")
        batch.pop("decoder_input_ids")
        topk = 1
        outputs = self.model.generate(**batch, 
            prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
            num_beams=topk, num_return_sequences=topk,
            output_scores=True,
            min_length=0,
            max_length=32,
        ).cpu()
        # calculate the rank in the decoder output 

        pad_id = self.tokenizer.pad_token_id
        outputs = self.decode(output_ids=outputs)
        labels = self.decode(output_ids=labels)
        
        preds = torch.tensor([self.rel2id[o] for o in outputs])
        true = real_label


        return {"eval_logits": preds.detach().cpu().numpy(), "eval_labels": true.detach().cpu().numpy()}


    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1 and not self.first:
            self.best_f1 = f1
        self.first = False
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)
       

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        real_label = batch.pop("label")
        labels = batch.pop("labels")
        batch.pop("decoder_input_ids")
        topk = 1
        outputs = self.model.generate(**batch, 
            prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
            num_beams=topk, num_return_sequences=topk,
            output_scores=True,
            min_length=0,
            max_length=32,
        ).cpu()
        # calculate the rank in the decoder output 

        pad_id = self.tokenizer.pad_token_id
        outputs = self.decode(output_ids=outputs)
        labels = self.decode(output_ids=labels)
        
        preds = torch.tensor([self.rel2id[o] for o in outputs])
        true = real_label


        return {"test_logits": preds.detach().cpu().numpy(), "test_labels": true.detach().cpu().numpy()}

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)
        

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }
