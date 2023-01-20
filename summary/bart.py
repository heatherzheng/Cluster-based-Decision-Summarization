import pandas as pd
from fastai.text.all import *
from transformers import *
from blurr.data.all import *
from blurr.modeling.all import *

import os
import json
import torch
from random import sample

import argparse

parser = argparse.ArgumentParser('svm model')
parser.add_argument('--cross_split',type=int,default=1,help='split number')

args = parser.parse_args()

split_num = args.cross_split

print("cross split", str(split_num))
train = open("bart_from_step2_sum/cluster_"+str(split_num)+"_train.txt.src","r")
train = [i.replace("\n","") for i in train.readlines()]

summf = open("bart_from_step2_sum/cluster_"+str(split_num)+"_train.txt.tgt.tagged","r")
summf = [i.replace("\n","") for i in summf.readlines()]

val = open("bart_from_step2_sum/cluster_"+str(split_num)+"_val.txt.src","r")
val = [i.replace("\n","") for i in val.readlines()]

val_summ = open("bart_from_step2_sum/cluster_"+str(split_num)+"_val.txt.tgt.tagged","r")
val_summ  = [i.replace("\n","") for i in val_summ.readlines()]


df = pd.DataFrame(
    {'text': train,
     'summary': summf,
    })

pretrained_model_name = "facebook/bart-large-cnn"
hf_arch, hf_config, hf_tokenizer, hf_model = BLURR.get_hf_objects(pretrained_model_name,
                                                                  model_cls=BartForConditionalGeneration)

#Create mini-batch and define parameters
hf_batch_tfm = HF_Seq2SeqBeforeBatchTransform(hf_arch, hf_config, hf_tokenizer, hf_model,
    task='summarization',max_tgt_length=60, min_tgt_length=5,
    text_gen_kwargs=
 {'max_length': 500, 'min_length': 5,'do_sample': False, 'early_stopping': True, 'num_beams': 6, 'temperature': 1.0,
  'top_k': 50, 'top_p': 1.0, 'repetition_penalty': 1.0, 'bad_words_ids': None, 'bos_token_id': 0, 'pad_token_id': 1,
 'eos_token_id': 2, 'length_penalty': 2.0, 'no_repeat_ngram_size': 3, 'encoder_no_repeat_ngram_size': 0,
 'num_return_sequences': 1, 'decoder_start_token_id': 2, 'use_cache': True, 'num_beam_groups': 1,
 'diversity_penalty': 0.0, 'output_attentions': False, 'output_hidden_states': False, 'output_scores': False,
 'return_dict_in_generate': False, 'forced_bos_token_id': 0, 'forced_eos_token_id': 2, 'remove_invalid_values': False})


#Prepare data for training
blocks = (HF_Seq2SeqBlock(before_batch_tfm=hf_batch_tfm), noop)
dblock = DataBlock(blocks=blocks, get_x=ColReader('text'), get_y=ColReader('summary'), splitter=RandomSplitter())
dls = dblock.dataloaders(df, batch_size = 2)

from collections import defaultdict
from rouge import Rouge

def evaluation(max_len):
  rouge = Rouge()

  score_avg = defaultdict(int)
  count = 0
  result = []
  for i in range(len(val)):
      outputs = learn.blurr_generate(val[i], num_return_sequences=1, early_stopping = True, num_beams=10, max_length=max_len)
      #long 150
      result.append([outputs,val_summ[i]])
      gold = val_summ[i]
      scores = rouge.get_scores(outputs[0], gold)
      count += 1
      for metric in scores[0]:
          for t in scores[0][metric]:
              score_avg[metric+t] += scores[0][metric][t]

  with open('bart_short_test_'+str(split_num)+'.json', 'w') as f:
      json.dump(result, f)
  for i in score_avg:
      score_avg[i] = score_avg[i] / count
  print(score_avg)

seq2seq_metrics = {
        'rouge': {
            'compute_kwargs': { 'rouge_types': ["rouge1", "rouge2", "rougeL"], 'use_stemmer': True },
            'returns': ["rouge1", "rouge2", "rougeL"]
        }}

#Model
model = HF_BaseModelWrapper(hf_model)
learn_cbs = [HF_BaseModelCallback]
fit_cbs = [HF_Seq2SeqMetricsCallback(custom_metrics=seq2seq_metrics)]

#Specify training
learn = Learner(dls, model,
                opt_func=ranger,loss_func=CrossEntropyLossFlat(),
                cbs=learn_cbs,splitter=partial(seq2seq_splitter, arch=hf_arch)).to_fp16()

#Create optimizer with default hyper-parameters
learn.create_opt()
learn.freeze()

#Training
lr = 3e-5
for epoch in range(7):
    learn.fit_one_cycle(1, lr_max=lr, cbs=fit_cbs)
    if epoch % 1==0:
        evaluation(15)
        evaluation(12)
        learn.save('cluster_from_step2_summ/bart_'+str(split_num)+"_"+str(epoch))

