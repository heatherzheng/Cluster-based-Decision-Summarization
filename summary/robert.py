import argparse

parser = argparse.ArgumentParser('model')
parser.add_argument('--cross_split',type=int,default=0,help='split number')

args = parser.parse_args()
split_num = args.cross_split
print(str(split_num))
file1 = open('bart_from_step2/cluster_'+str(split_num)+'_train.txt.src', 'r')
train = [i.replace("\n","") for i in file1.readlines()]

file2 = open('bart_from_step2/cluster_'+str(split_num)+'_train.txt.tgt.tagged', 'r')
tmp = [i.replace("\n","") for i in file2.readlines()]

label = []
for i in tmp:
  if i=="empty":
    label.append(0)
  else:
    label.append(1)

file1 = open('bart_from_step2/cluster_'+str(split_num)+'_val.txt.src', 'r')
val = [i.replace("\n","") for i in file1.readlines()]

file2 = open('bart_from_step2/cluster_'+str(split_num)+'_val.txt.tgt.tagged', 'r')
tmp = [i.replace("\n","") for i in file2.readlines()]

val_label = []
for i in tmp:
  if i=="empty":
    val_label.append(0.0)
  else:
    val_label.append(1.0)

# Importing the libraries needed
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import transformers
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import logging
logging.basicConfig(level=logging.ERROR)

from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import random
class CustomDataset(Dataset):
    def __init__(self, input, target, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = input
        self.targets = target
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

tokenizer = RobertaTokenizer.from_pretrained("roberta-base", truncation=True, do_lower_case=True)
MAX_LEN = 500
temp = list(zip(train, label))
random.shuffle(temp)
res1, res2 = zip(*temp)
res1, res2 = list(res1), list(res2)

train_data = CustomDataset(res1,res2, tokenizer, MAX_LEN)
train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)

test_data = CustomDataset(val,val_label, tokenizer, MAX_LEN)
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=False)

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

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
model = RobertaClass()
model.to(device)
loss_function = torch.nn.BCELoss()
LEARNING_RATE = 1e-05
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

# Defining the training function on the 80% of the dataset for tuning the distilbert model

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in tqdm(enumerate(train_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)
        bceweight = torch.where(targets > 0, 2, 1).to(device)
        criterion = torch.nn.BCELoss(weight = bceweight)
        criterion.to(device)
        try:
            loss = criterion(outputs, targets)
        except:
            continue
        tr_loss += loss.item()

        predicted_classes = torch.round(outputs)
        n_correct += calcuate_accuracy(predicted_classes, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)

        if _%5000==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return

def evaluate():
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0

    model.eval()
    predict_true,target_true,correct_true=0,0,0
    for _,data in tqdm(enumerate(test_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)
        try:
            loss = loss_function(outputs, targets)
        except:
            continue
        tr_loss += loss.item()

        predicted_classes = torch.round(outputs)
        #n_correct += calcuate_accuracy(predicted_classes, targets)

        predict_true += torch.sum(predicted_classes == 1).float()
        target_true += torch.sum(targets==1).float()
        correct_true += torch.sum(
            (predicted_classes == targets).float() * (predicted_classes == 1)).float()
    p = correct_true/predict_true
    r = correct_true/target_true
    f1 = 2*p*r/(p+r)
    print(str(p),str(r))
    print(f'Test set Epoch {epoch}: {f1}')

    return

EPOCHS = 7
for epoch in range(EPOCHS):
    train(epoch)
    torch.save(model.state_dict(), 'robert_ckpt/'+str(split_num)+"_"+str(epoch)+'.ckpt')
    evaluate()
