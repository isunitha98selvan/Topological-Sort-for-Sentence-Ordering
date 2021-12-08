# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import csv
import codecs
import functools
import torch.nn as nn
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                    BertModel, BertTokenizer)

from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import DataProcessor, InputExample, InputFeatures
torch.set_printoptions(threshold=1000)

torch.cuda.set_device(0)
logger = logging.getLogger(__name__)

#ALL_MODELS = sum((tuple(
#    conf.pretrained_config_archive_map.keys()) for conf in (
#        BertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer)
}

class LinearClassifier(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.fcc = nn.Linear(hidden_dim*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        l1 = self.fcc(inp)
        return self.sigmoid(l1)

class AdditiveAttention(nn.Module):
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim) 
        self.tanh = nn.Tanh()
        self.v = nn.Parameter(torch.Tensor(1, hidden_dim))
        self.softmax = nn.Softmax(dim=1)
        nn.init.normal_(self.v, 0, 0.1)
     
    def forward(self, h_i):
      u_i = self.tanh(self.W(h_i))
      alpha_i = self.softmax(self.v @ u_i.T)
      result = alpha_i @ h_i
      return result

hidden_dim = 768



class EncoderBlock(nn.Module):
  def __init__(self, hidden_dim, num_heads, dim_feedforward):
    super().__init__()
    
    self.hidden_dim = hidden_dim
    self.num_heads = num_heads
    self.layer_norm = nn.LayerNorm(hidden_dim)
    # Two-layer MLP
    dropout=0.0
    self.ffn = nn.Sequential(
        nn.Linear(hidden_dim, dim_feedforward),
        nn.Dropout(dropout),
        nn.ReLU(inplace=True),
        nn.Linear(dim_feedforward, hidden_dim)
    )
    self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads)


  def forward(self, X):
    attn_output, attn_output_weights = self.multihead_attn(X, X, X)
    X_capL = self.layer_norm(X + attn_output)
    X_L = self.layer_norm(X_capL + self.ffn(X_capL))

    return X_L

class ParagraphEncoder(nn.Module):
  def __init__(self, num_layers, hidden_dim, num_heads, dim_feedforward):
      super().__init__()
      self.layers = nn.ModuleList([EncoderBlock(hidden_dim, num_heads, dim_feedforward) for _ in range(num_layers)])

  def forward(self, x):
      for l in self.layers:
          x = l(x)
      return x
    
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def getSent1TokenRange(inputId):
    startEndIdxList = []
    startIdxs1 = (inputId == 101).nonzero()[0] + 1
    endIdxs1 = (inputId == 102).nonzero()[0]
    startIdxs2 = (inputId == 102).nonzero()[0] + 1
    endIdxs2 = (inputId == 102).nonzero()[1]
    startEndIdxList.append((startIdxs1, endIdxs1))
    startEndIdxList.append((startIdxs2, endIdxs2))
    return startEndIdxList

def attendToTokenEmbeddings(args, a, tokenEmbeddingAfterBertLayer, sentTokenRange):
    sentenceRepresentation = []
    a.to(args.device)
    tokenRanges1 = sentTokenRange[0]
    tokenRanges2 = sentTokenRange[1]
    s1Id = tokenRanges1[0]
    e1Id = tokenRanges1[1]
    s2Id = tokenRanges2[0]
    e2Id = tokenRanges2[1]

    firstSentenceTokenRep = tokenEmbeddingAfterBertLayer[s1Id : e1Id]
    # firstSentenceTokenRep.to(args.device)
    
    sentence_rep_s1 = a.forward(firstSentenceTokenRep)
    sentenceRepresentation.append(sentence_rep_s1)

    secondSentenceTokenRep = tokenEmbeddingAfterBertLayer[s2Id : e2Id]
    # secondSentenceTokenRep.to(args.device)
    # a.to(args.device)
    sentence_rep_s2 = a.forward(secondSentenceTokenRep)
    sentenceRepresentation.append(sentence_rep_s2)

    return sentenceRepresentation


def attendToSentencePairs(args, cattention, pairwiseSentenceRepresentation):
    hidden_dim = 768

    if len(pairwiseSentenceRepresentation) > 0:
        pairwiseSentenceRepresentation[0] = pairwiseSentenceRepresentation[0].reshape(1,-1)
        pairSentReprTensor = pairwiseSentenceRepresentation[0]
        # pairSentReprTensor = pairwiseSentenceRepresentation[0].to(device=args.device)
        # pairwiseSentenceRepresentation[0].to(args.device)
        
    for i,spair in enumerate(pairwiseSentenceRepresentation):
        spair = spair.reshape(1,-1)
        # spair.to(args.device)
        pairSentReprTensor = torch.cat((pairSentReprTensor, spair), dim=0)

    # pairSentReprTensor.to(args.device)
    # cattention.to(args.device)
    # print("device: ", args.device)
    sentReprTensor = cattention.forward(pairSentReprTensor)
    # print("Sent Rep: ", sentReprTensor)
    return sentReprTensor


def train(args, model, tokenizer):
    """ Train the model """
    cattention = AdditiveAttention(hidden_dim).to(device=args.device)
    a = AdditiveAttention(hidden_dim).to(device=args.device)
    lc = LinearClassifier(hidden_dim).to(device=args.device)
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    
    processor = PairProcessor()
    output_mode = 'classification'

   

    label_list = processor.get_labels()
    # examples  = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
    examples  =  processor.get_train_examples(args.data_dir)
     # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
        'weight_decay': args.weight_decay
        },
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
        'weight_decay': 0.0
        }
        ]
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate, 
        eps=args.adam_epsilon
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
            )

    set_seed(args)
    
    paragraphEncoder = ParagraphEncoder(12, 768, 6, 32).to(device=args.device)
    loss = nn.BCELoss()
    acc_loss = 0.0
    global_step = 0
    for epoch in range(1):
        tr_loss = 0.0
        print("Starting epoch: ", epoch)
        para_id = 0
        start_idx = 0
        s1_id = []
        s2_id = []
        for i,example in enumerate(examples):
            # print("Batch: ", i)
            ids = example.guid.split(":")
            if i!=0 and ids[0]!=para_id:
                features = convert_examples_to_features(examples[start_idx:i],
                                    tokenizer,
                                    label_list=label_list,
                                    max_length=args.max_seq_length,
                                    output_mode=output_mode,
                                    pad_on_left=False,                 # pad on the left for xlnet
                                    pad_token=tokenizer.convert_tokens_to_ids(
                                        [tokenizer.pad_token])[0],
                                    pad_token_segment_id=0,
                                    )

                all_input_ids = torch.tensor(
                    [f.input_ids for f in features], dtype=torch.long)
                all_attention_mask = torch.tensor(
                    [f.attention_mask for f in features], dtype=torch.long)
                all_token_type_ids = torch.tensor(
                    [f.token_type_ids for f in features], dtype=torch.long)
                all_labels = torch.tensor(
                    [f.label for f in features], dtype=torch.long)
                all_s1 = torch.tensor([int(s) for s in s1_id], dtype=torch.int32)
                all_s2 = torch.tensor([int(s) for s in s2_id], dtype=torch.int32)

                dataset = TensorDataset(
                    all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_s1, all_s2)

                start_idx = i
                train_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
                
                if args.max_steps > 0:
                    t_total = args.max_steps
                    # args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
                else:
                    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
                
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=args.warmup_steps, 
                    num_training_steps=t_total
                    )
                logger.info("***** Running training *****")
                logger.info("  Num examples = %d", len(dataset))
                logger.info("  Num Epochs = %d", args.num_train_epochs)
                # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
                # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                #             args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
                # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
                logger.info("  Total optimization steps = %d", t_total)

                n_sent = max(all_s2) + 1
                pairwiseSentenceRepresentation_1 = torch.zeros(n_sent, n_sent, 768).to(device=args.device)
                pairwiseSentenceRepresentation_2 = torch.zeros(n_sent, n_sent, 768).to(device=args.device)

                # train_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
                # for step, batch in enumerate(train_iterator):
                gen = iter(train_dataloader)
                batch = next(gen)
                model.train()
                batch_1 = tuple(t.to(args.device) for t in batch)

                inputs = {'input_ids':      batch_1[0],
                        'attention_mask': batch_1[1],
                        'token_type_ids': batch_1[2]
                        }
                labels = (batch_1[3]).to(torch.float32)
                s1 = batch_1[4]
                s2 = batch_1[5]
                outputs = model(**inputs)
                hidden_state = outputs[0]
                # model.to(args.device)
                # hidden_state.to(args.device)
                for i in range(hidden_state.shape[0]):
                    sentTokenRange = getSent1TokenRange(inputs["input_ids"][i])
                    res = attendToTokenEmbeddings(args, a, hidden_state[i], sentTokenRange)
                    pairwiseSentenceRepresentation_1[s1[i]][s2[i]] = res[0]
                    pairwiseSentenceRepresentation_2[s1[i]][s2[i]] = res[1]
            
                allpairsSentenceRepresentation = torch.zeros(n_sent, 768).to(device=args.device)
            
                for sent in range(n_sent):
                    #get all 2n - 2 representations
                    allReps = []
                    for ri in range(n_sent):
                        for ci in range(n_sent):
                            if ri == sent or ci == sent:
                                if pairwiseSentenceRepresentation_1[ri][ci] is not None:
                                    allReps.append(pairwiseSentenceRepresentation_1[ri][ci])
                                if pairwiseSentenceRepresentation_2[ri][ci] is not None:
                                    allReps.append(pairwiseSentenceRepresentation_2[ri][ci])
                

                allpairsSentenceRepresentation[sent] = attendToSentencePairs(args, cattention, allReps)
        
                allpairsSentenceRepresentation = allpairsSentenceRepresentation.unsqueeze(0)
                op = paragraphEncoder(allpairsSentenceRepresentation)


                for i in range(len(s1)):
                    sent1_id, sent2_id = s1[i], s2[i]
                    label = labels[i]
            
                    x = torch.cat((op[0][sent1_id].reshape(1,-1), op[0][sent2_id].reshape(1,-1)), dim = -1)
                    output = lc(x)
                    output[0].to(args.device)
                    tr_loss=loss(output[0].reshape(1,-1), label.reshape(1,-1))
                    if args.gradient_accumulation_steps > 1:
                        tr_loss = tr_loss / args.gradient_accumulation_steps
                    print("Loss", tr_loss)
                    tr_loss.backward(retain_graph=True)
                    if (i + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        global_step += 1
                        model.zero_grad()
                    acc_loss+=tr_loss

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                s1_id = []
                s2_id = []
                del pairwiseSentenceRepresentation_1
                del pairwiseSentenceRepresentation_2
                del allpairsSentenceRepresentation
                del batch_1
                torch.cuda.empty_cache()


            para_id = ids[0]
            s1_id.append(ids[1])
            s2_id.append(ids[2])
        
        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)

        # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
        # Log metrics
        # if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
    results = evaluate(args, model, tokenizer)
    
    # for key, value in results.items():
    #     tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
    tb_writer.add_scalar('loss', results)
        # tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
        # print('loss: ' + str((tr_loss - logging_loss)/args.logging_steps) + ' step: ' + str(global_step))
        # logging_loss = tr_loss

        # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
            # Save model checkpoint
    

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

class MyTestDataset(Dataset):
    """Dataset during test time"""

    def __init__(self, tensor_data, sents):
        assert len(tensor_data) == len(sents)
        self.tensor_data = tensor_data
        self.rows = sents
        
    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        return (self.tensor_data[idx], self.rows[idx])
    
def evaluate_test(args, model, tokenizer, prefix=""):
    
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = PairProcessor()
    output_mode = "classification"
    
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'test',
        'bert',
        str(args.max_seq_length),
        'pair_order'))
    
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        lines = torch.load(cached_features_file + '_lines')
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()

        examples, lines = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=False,                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
        )
        if args.local_rank in [-1, 0]:
            logger.info(
                "Saving features into cached file %s", 
                cached_features_file)
            torch.save(features, cached_features_file)
            torch.save(lines, cached_features_file + '_lines')
        
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier() 

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, 
        all_attention_mask, 
        all_token_type_ids, 
        all_labels
        )

    
    eval_outputs_dirs = (args.output_dir,)
    file_h = codecs.open(args.data_dir + "test_results.tsv", "w", "utf-8")
    outF = csv.writer(file_h, delimiter='\t')

    results = {}
    for eval_output_dir in eval_outputs_dirs:
        eval_dataset = MyTestDataset(dataset, lines)
        
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        
        eval_dataloader = DataLoader(
            eval_dataset, 
            sampler=eval_sampler, 
            batch_size=args.eval_batch_size
            )

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            row = batch[1]
            rows = {
                'guid': row[0],
                'text_a': row[1],
                'text_b': row[2],
                'labels': row[3],
                'pos_a': row[4],
                'pos_b':row[5]
            }
            del row
            batch = tuple(t.to(args.device) for t in batch[0])

            with torch.no_grad():
                inputs = {
                        'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2],
                        'labels':         batch[3]}

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            logits = logits.detach().cpu().numpy()
            tmp_pred = np.argmax(logits, axis=1)
            for widx in range(logits.shape[0]):
                outF.writerow([rows['guid'][widx], rows['text_a'][widx], \
                rows['text_b'][widx], rows['labels'][widx], \
                rows['pos_a'][widx], rows['pos_b'][widx], \
                logits[widx][0], logits[widx][1], tmp_pred[widx]])
            if preds is None:
                preds = logits
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits, axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)

        result = compute_metrics("mnli", preds, out_label_ids)
        results.update(result)

        file_h.close()
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    
    eval_outputs_dirs = (args.output_dir)

    results = {}
    paragraphEncoder = ParagraphEncoder(12, 768, 6, 32).to(device=args.device)
    for eval_output_dir in eval_outputs_dirs:
        # eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        para_id = 0
        start_idx = 0
        s1_id = []
        s2_id = []
        # output_file = os.path.join(eval_output_dir, "results.txt")
        # print("output file: ", output_file)
        f = open("results.txt", "w")

        processor = PairProcessor()
        output_mode = 'classification'
        label_list = processor.get_labels()
        examples = processor.get_test_examples(args.data_dir)
        loss = nn.BCELoss()
        global_step = 0
        tr_loss = 0
        preds = []
        all_labels = []
        out_label_ids = []
        for i, example in enumerate(examples[0]):
            # print("Inner: ")
            # break
            ids = example.guid.split(":")
            if i!=0 and ids[0]!=para_id:
                features = convert_examples_to_features(examples[0][start_idx:i],
                                    tokenizer,
                                    label_list=label_list,
                                    max_length=args.max_seq_length,
                                    output_mode=output_mode,
                                    pad_on_left=False,                 # pad on the left for xlnet
                                    pad_token=tokenizer.convert_tokens_to_ids(
                                        [tokenizer.pad_token])[0],
                                    pad_token_segment_id=0,
                                    )

                all_input_ids = torch.tensor(
                    [f.input_ids for f in features], dtype=torch.long)
                all_attention_mask = torch.tensor(
                    [f.attention_mask for f in features], dtype=torch.long)
                all_token_type_ids = torch.tensor(
                    [f.token_type_ids for f in features], dtype=torch.long)
                all_labels = torch.tensor(
                    [f.label for f in features], dtype=torch.long)
                all_s1 = torch.tensor([int(s) for s in s1_id], dtype=torch.int32)
                all_s2 = torch.tensor([int(s) for s in s2_id], dtype=torch.int32)
                eval_dataset = TensorDataset(
                    all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_s1, all_s2)

                start_idx = i

                eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset))

                # Eval!
                logger.info("***** Running evaluation {} *****".format(prefix))
                logger.info("  Num examples = %d", len(eval_dataset))
                logger.info("  Batch size = %d", args.eval_batch_size)
                n_sent = max(max(all_s1), max(all_s2))+ 1
                
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    model.eval()
                    batch = tuple(t.to(args.device) for t in batch)

                    with torch.no_grad():
                        inputs = {'input_ids':      batch[0],
                                'attention_mask': batch[1],
                                'token_type_ids': batch[2],
                                }
                        outputs = model(**inputs)
                        hidden_state = outputs[0]
                        labels = (batch[3]).to(torch.float32)
                        s1 = batch[4]
                        s2 = batch[5]
                        
                        pairwiseSentenceRepresentation_1 = torch.zeros(n_sent, n_sent, 768).to(device=args.device)
                        pairwiseSentenceRepresentation_2 = torch.zeros(n_sent, n_sent, 768).to(device=args.device)

                        for i in range(hidden_state.shape[0]):
                            sentTokenRange = getSent1TokenRange(inputs["input_ids"][i])
                            res = attendToTokenEmbeddings(args, hidden_state[i], sentTokenRange)
                            pairwiseSentenceRepresentation_1[s1[i]][s2[i]] = res[0]
                            pairwiseSentenceRepresentation_2[s1[i]][s2[i]] = res[1]
                    
                    allpairsSentenceRepresentation = torch.zeros(n_sent, 768).to(device=args.device)
              
                    for sent in range(n_sent):
                        #get all 2n - 2 representations
                        allReps = []
                        for ri in range(n_sent):
                            for ci in range(n_sent):
                                if ri == sent or ci == sent:
                                    if pairwiseSentenceRepresentation_1[ri][ci] is not None:
                                        allReps.append(pairwiseSentenceRepresentation_1[ri][ci])
                                    if pairwiseSentenceRepresentation_2[ri][ci] is not None:
                                        allReps.append(pairwiseSentenceRepresentation_2[ri][ci])
                    

                    allpairsSentenceRepresentation[sent] = attendToSentencePairs(args, allReps)
            
                    allpairsSentenceRepresentation = allpairsSentenceRepresentation.unsqueeze(0)
                    op = paragraphEncoder(allpairsSentenceRepresentation)
                    for i in range(len(s1)):
                        sent1_id, sent2_id = s1[i], s2[i]
                        label = labels[i]
                
                        x = torch.cat((op[0][sent1_id].reshape(1,-1), op[0][sent2_id].reshape(1,-1)), dim = -1)
                        output = lc(x)
                        output[0].to(args.device)
                        tr_loss+=loss(output[0].reshape(1,-1), label.reshape(1,-1))
                        # preds.append(output[0].item())
                        res = 0.0
                        if output[0]>0.5:
                            res = 1.0  
                        else:
                            res = 0.0
                        preds.append(res)

                        out_label_ids.append(label.item())
                        global_step += 1
                        
                        f.write(str(para_id) + " " + str(sent1_id.item()) + " " + str(sent2_id.item()) + " " + str(res) + " " + str(label.item()))
                        f.write("\n")
                s1_id = []
                s2_id = []
            para_id = ids[0]
            s1_id.append(ids[1])
            s2_id.append(ids[2])

            
                # if preds is None:
                #     preds = logits.detach().cpu().numpy()
                #     out_label_ids = inputs['labels'].detach().cpu().numpy()
                # else:
                #     preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                #     out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = tr_loss / global_step
        print("Eval loss: ", eval_loss)
    return eval_loss
        # preds = np.argmax(preds, axis=1)
        # print(preds)
        # print(out_label_ids)
        # print("Printing dtype:")
        # print(type(preds[0]))
        # print(type(out_label_ids[0]))
    #     result = compute_metrics("mnli", preds, out_label_ids)
    #     results.update(result)

    #     output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    #     with open(output_eval_file, "w") as writer:
    #         logger.info("***** Eval results {} *****".format(prefix))
    #         for key in sorted(result.keys()):
    #             logger.info("  %s = %s", key, str(result[key]))
    #             writer.write("%s = %s\n" % (key, str(result[key])))

    # return results


class PairProcessor(DataProcessor):
    """Pair Processor for the pair ordering task."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_test_examples(self, data_dir):
        return self._create_test_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
    
    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[1].lower()
                text_b = line[2].lower()
                label = line[3]
            except IndexError:
                print('cannot read the line: ' + line)
                continue
            examples.append(InputExample(
                                        guid=guid, 
                                        text_a=text_a, 
                                        text_b=text_b, 
                                        label=label
                                    ))
        return examples
    
    def _create_test_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples, rows = [], []
        for (_, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[1].lower()
                text_b = line[2].lower()
                label = line[3]
            except IndexError:
                print('cannot read the line: ' + line)
                continue
            examples.append(InputExample(
                                    guid=guid, 
                                    text_a=text_a, 
                                    text_b=text_b, 
                                    label=label
                            ))
            rows.append(line)
        return examples, rows


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, 
                        type=str, required=True,
                        help="The input data dir. Should contain the .tsv "
                        "files (or other data files) for the task.")
    parser.add_argument("--output_dir", default='run_glue_test', 
                        type=str, required=True,
                        help="The output directory where the model "
                        "predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length", default=105, type=int,
                        help="The maximum total input sequence length "
                        "after tokenization. Sequences longer than this "
                        "will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate "
                        "before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps "
                        "to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=2000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the "
                        "same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision "
                        "(through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level "
                        "selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    args.output_mode = "classification"

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  

    args.model_type = 'bert'
    _, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')
    model = model_class.from_pretrained('bert-base-uncased')

    if args.local_rank == 0:
        torch.distributed.barrier()  

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:
        # train_dataset = load_and_cache_examples(
        #                 args, tokenizer, evaluate=False)
        global_step, tr_loss = train(
                        args, model, tokenizer)
        logger.info(
            " global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)


    # Evaluation
    results = {}
    if (args.do_eval or args.do_test) and args.local_rank in [-1, 0]:
        #tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            # if args.do_test:
                # result = evaluate_test(args, model, tokenizer, prefix=global_step)
            # elif args.do_eval:
                # result = evaluate(args, model, tokenizer, prefix=global_step)
            # result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            # results.update(result)

    return None

if __name__ == "__main__":
    main()
