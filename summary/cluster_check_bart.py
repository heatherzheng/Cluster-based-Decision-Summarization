import pandas as pd
import nltk
from fastai.text.all import *
from transformers import *
from blurr.data.all import *
from blurr.modeling.all import *

import os
import json
import torch
from random import sample
from nltk.stem import WordNetLemmatizer

import argparse

parser = argparse.ArgumentParser('model')
parser.add_argument('--cross_split',type=int,default=1,help='split number')
parser.add_argument('--save_epoch',type=int,default=100,help='load epoch')
parser.add_argument('--bart',type=str,default='',help='load epoch')
parser.add_argument('--robert',type=str,default='',help='load epoch')

args = parser.parse_args()
split_num = args.cross_split
robert_ckpt = args.robert
bart_ckpt = args.bart
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.sigmoid(self.classifier(pooler))
        return output.squeeze()

cmodel = RobertaClass()
cmodel.load_state_dict(torch.load('robert_ckpt/'+robert_ckpt))
cmodel.to(device)

cmodel.eval()
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", truncation=True, do_lower_case=True)

def convert_data(text):
    text = " ".join(text.split())
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=500,
        pad_to_max_length=True,
        return_token_type_ids=True
    )
    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    token_type_ids = inputs["token_type_ids"]

    new_dict = {'ids': torch.tensor([ids], dtype=torch.long),
        'mask': torch.tensor([mask], dtype=torch.long),
        'token_type_ids': torch.tensor([token_type_ids], dtype=torch.long)}
    return new_dict

args = parser.parse_args()

split_num = args.cross_split
save_epoch = args.save_epoch

with open("/home/sz488/Neural_Topic_Models/test_output_simple/fix_"+str(split_num)+"test_cluster.json") as json_file:
    data = json.load(json_file)
val = data[0]
val_f = data[1]

f = open ('test_sum_gold.json', "r")
val_gold= json.loads(f.read())

train = open("bart_data/cluster_1_train.txt.src","r")
train = [i.replace("\n","") for i in train.readlines()]

summf = open("bart_data/cluster_1_train.txt.tgt.tagged","r")
summf = [i.replace("\n","") for i in summf.readlines()]

df = pd.DataFrame(
    {'text': train,
     'summary': summf,
    })

pretrained_model_name = "facebook/bart-large-cnn"
hf_arch, hf_config, hf_tokenizer, hf_model = BLURR.get_hf_objects(pretrained_model_name,
                                                                  model_cls=BartForConditionalGeneration)

#Create mini-batch and define parameters
hf_batch_tfm = HF_Seq2SeqBeforeBatchTransform(hf_arch, hf_config, hf_tokenizer, hf_model,
    task='summarization',max_tgt_length=60, min_tgt_length=3,
    text_gen_kwargs=
 {'max_length': 500, 'min_length': 5,'do_sample': False, 'early_stopping': True, 'num_beams': 10, 'temperature': 1.0,
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
def old_delete_rep(lst):
    all_word = set()
    sents = []
    for i in lst:
        rep = 0
        for j in i.split():
            if j in all_word:
                rep +=1
            all_word.add(j)
        if rep/len(i.split())>0.9:
            continue
        else:
            sents.append(i)
    return sents

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def delete_rep(inputs):
    topic = open("/home/sz488/Neural_Topic_Models/word_"+str(split_num)+".json")
    seed_cluser_map = json.load(topic)

    lst = []
    for s in inputs:
        tmp = []
        s = word_tokenize(s)
        s_pos_tag = pos_tag(s)
        s = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else lemmatizer.lemmatize(i) for i,j in s_pos_tag]

        for w in s:
            if w in seed_cluser_map:
                tmp.append(w.lower())

        lst.append(tmp)


    all_word = set()
    sents = []
    remove = set()
    for i in range(len(lst)):
        max_rep = 0
        max_rep_sent = None
        for j in range(len(lst)):
            if i!=j:
                count = 0
                for word in lst[i]:
                    if word in lst[j]:
                        count +=1
                max_rep = count
                max_rep_sent = j
                if max_rep>len(lst[i])*0.5:
                    if len(inputs[i].split())<len(inputs[max_rep_sent].split()) or (len(inputs[i].split())==len(inputs[max_rep_sent].split()) and i<max_rep_sent):
                        remove.add(i)
    ans = []
    print(remove)
    for idx,sent in enumerate(inputs):
        if idx not in remove and lst[idx]!=[]:
            ans.append(sent[1:])
            print(sent[1:])




    return " ".join(ans)

def evaluation(max_len):
  rouge = Rouge()

  score_avg = defaultdict(int)
  count = 0
  toutput,tgold =[],[]
  prev = val_f[0]
  for i in range(len(val)):
      d = convert_data(val[i])

      ids = d['ids'].to(device, dtype = torch.long)
      mask = d['mask'].to(device, dtype = torch.long)
      token_type_ids = d['token_type_ids'].to(device, dtype = torch.long)
      thre = cmodel(ids, mask, token_type_ids)

      if val_f[i]==prev:
          if thre>0.5:
              outputs = learn.blurr_generate(val[i], early_stopping=False, num_return_sequences=1, num_beams=10, max_length=max_len)[0]
              toutput.append(outputs)
          #print("len",str(len(outputs[0].split())))
      else:
          #pred = " ".join(toutput)
          pred=delete_rep(toutput)
          ref = val_gold[prev.split(".")[0]]
          if ref=="":
              print("empty ref")

          if ref!="":
              scores = rouge.get_scores(pred, val_gold[prev.split(".")[0]])
              #print(pred)
              #print(val_gold[prev.split(".")[0]])
              #print("____")
              count += 1
              for metric in scores[0]:
                  for t in scores[0][metric]:
                      score_avg[metric+t] += scores[0][metric][t]
          toutput,tgold = [],[]
          if thre>0.5:
              outputs = learn.blurr_generate(val[i], early_stopping=False, num_return_sequences=1, num_beams=10, max_length=max_len)[0]
              toutput.append(outputs)

      prev = val_f[i]

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

learn.load('cluster_from_step2_summ/'+bart_ckpt)
evaluation(15)
