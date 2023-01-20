import os
import re
import torch
import json
import pickle
import argparse
import logging
import time
import merge
from multiprocessing import cpu_count
from random import sample
import bcubed

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import torch.optim as optim
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')
sentence_bert= SentenceTransformer('all-MiniLM-L6-v2')

parser = argparse.ArgumentParser('topic model')
arser.add_argument('--cross_split',type=int,default=1,help='split number')
parser.add_argument('--topic_file',type=str,default="",help='topic file')
parser.add_argument('--save_model',type=str,default="",help='check point')
args = parser.parse_args()

from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
from nltk import pos_tag
from nltk.tokenize import word_tokenize
file = open(args.topic_file)
print(args.topic_file)
seed_cluser_map = json.load(file)

save_model_dir = args.save_model
print(save_model_dir)
topic_len = 0
for key in seed_cluser_map:
    topic_len = max(topic_len, int(seed_cluser_map[key]))
class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

def get_feature_vec(sent1,sent2,cont1,cont2):
    s1 = word_tokenize(sent1)
    s_pos_tag = pos_tag(s1)
    s = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else lemmatizer.lemmatize(i) for i,j in s_pos_tag]
    tmp_lst=[0]*(topic_len +1)
    for word in s:
        if word in seed_cluser_map:
            tmp_lst[int(seed_cluser_map[word])]=1

    s1 = word_tokenize(sent2)
    s_pos_tag = pos_tag(s1)
    s = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else lemmatizer.lemmatize(i) for i,j in s_pos_tag]
    tmp_lst2=[0]*(topic_len +1)
    for word in s:
        if word in seed_cluser_map:
            tmp_lst2[int(seed_cluser_map[word])]=1
    rep = 0
    search_lst = sent1.split()
    check_lst = sent2.split()
    for i in search_lst:
        if i in check_lst:
            rep += 1

    search_lst = cont1.split()
    check_lst = cont2.split()
    crep = 0
    for i in search_lst:
        if i in check_lst:
            crep += 1

    return tmp_lst+tmp_lst2+list(sentence_bert.encode(sent1)) + list(sentence_bert.encode(sent2))+[float(rep),float(rep/min(len(search_lst),len(check_lst))),float(crep)]

def most_frequent(List):
    return max(set(List), key = List.count)

def get_topic(sent1):
    topic = []
    s1 = word_tokenize(sent1)
    s_pos_tag = pos_tag(s1)
    s = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else lemmatizer.lemmatize(i) for i,j in s_pos_tag]
    tmp_lst=[0]*(topic_len +1)
    for word in s:
        if word in seed_cluser_map:
            tmp_lst[int(seed_cluser_map[word])]=1
            topic.append(int(seed_cluser_map[word]))
    return topic

def read_data(dir):
    train = []
    label = []
    all_fname = []
    pos,neg = 0,0
    for file in os.listdir(dir):
        if "json" in file:
            all_fname.append(file)
            tmp_sentences = []
            tmp_cluster = []
            with open(dir+file) as json_file:
                data = json.load(json_file)
                count = 0


                for sent in range(len(data)):
                  for pair in range(sent+1,len(data)):
                    #embed = [data[sent][1],data[pair][1],data[sent][0],data[pair][0],data[sent][3],data[pair][3],data[pair][4]]
                    #train.append(embed)
                    #print(data[sent][2],data[pair][2])
                    label.append(data[sent][3]==data[pair][3])

                    #sentence_pos,time, sentence, group, pos over passage, context sentence
                    train.append([data[sent][2],data[pair][2],data[sent][5],data[pair][5],abs(data[sent][1]-data[pair][1]),
                    abs(data[sent][4]-data[pair][4])])

                    if data[sent][3]==data[pair][3]:
                        pos +=1
                    else:
                        neg +=1
    print("pos ", str(pos) +" "+"neg "+str(neg))
    return train, label

class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(384*2+2+3+2*(topic_len +1), 200)
        self.linear2 = torch.nn.Linear(200,1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x1 = self.relu((self.linear1(x)))
        return self.sigmoid(self.linear2(x1))


def evaluation(net, val_loader,device,mode):
    net.eval()
    count = 0
    target_true, correct_true, predicted_true,acc = 0,0,0,0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_train = []
            for i in range(len(x_batch[0])):
                embed = get_feature_vec(x_batch[0][i],x_batch[1][i],x_batch[2][i],x_batch[3][i])
                embed.extend([x_batch[4][i],x_batch[5][i]])

                x_train.append(embed)
            inputs = torch.tensor(x_train).float().to(device)
            # forward + backward + optimize
            outputs = net(inputs).squeeze()

            predicted_classes = torch.round(outputs)
            #predicted_classes = torch.argmax(torch.softmax(outputs,dim=1),dim=1)
            target_classes = (y_batch).float().to(device)
            target_true += torch.sum(target_classes == 1).float()
            predicted_true += torch.sum(predicted_classes==1).float()
            correct_true += torch.sum(
                (predicted_classes == target_classes).float() * (predicted_classes == 1)).float()
            acc += torch.sum(predicted_classes==(y_batch).float().to(device))
            count += len(y_batch)

        recall = correct_true / target_true
        precision = correct_true / predicted_true
        f1_score = 2 * precision * recall / (precision + recall)

        target_true, correct_true, predicted_true = 0,0,0
        print(mode)
        f1 = f1_score.cpu().numpy()
        print("recall",recall.cpu().numpy(),"precision",precision.cpu().numpy(),"f1",f1)
        print("acc", acc.cpu().numpy()/count)
        print(count)
        print(acc.cpu().numpy())
        return f1

def test_cluster(net, device,path,epoch):
    print(epoch)
    f,r,p=0,0,0
    bprecision,brecall,bfscore = 0,0,0
    count =0
    test_text,test_fname = [],[]
    voi = 0
    for file in os.listdir(path):
        if not "json" in file:
            continue

        true_cluster = {}
        lookindex = []
        group_num = 0
        val_dic = {}
        no_topic_index = []
        with open(path+file) as json_file:
            data = json.load(json_file)
            for sent in range(len(data)):
                true_cluster[sent] = set(str(data[sent][3]))
                group_num = max(group_num,data[sent][3])
                sent_topic = get_topic(data[sent][2])
                #if sent_topic != []:
                    #lookindex.append([sent])
                lookindex.append([sent])
                #else:
                    #no_topic_index.append([sent])
                for pair in range(sent+1,len(data)):
                    tmp = [data[sent][2],data[pair][2],data[sent][5],data[pair][5],abs(data[sent][1]-data[pair][1]),
                    abs(data[sent][4]-data[pair][4])]
                    embed = get_feature_vec(tmp[0],tmp[1],tmp[2],tmp[3])
                    embed.extend([tmp[4],tmp[5]])
                    inputs = torch.tensor([embed]).float().to(device)
                    val_dic[(sent,pair)] = net(inputs).squeeze().item()
        gold = [[] for i in range(group_num+1)]
        for key in true_cluster:
            for g in list(true_cluster[key]):
                gold[int(g)].append(key)
        predict = merge.merge_corpus(lookindex,val_dic)

        predict_cluter = {}
        for i in range(len(predict)):
            for ele in predict[i]:
                predict_cluter[ele] = set(str(i))
        tmp_bp = bcubed.precision(predict_cluter, true_cluster)
        bprecision += tmp_bp

        tmp_br = bcubed.recall(predict_cluter, true_cluster)
        brecall += tmp_br
        bfscore += bcubed.fscore(tmp_bp,tmp_br)

        pt,rt, ft= merge.get_pair_score(predict, gold)
        p += pt
        r += rt
        f += ft
        count += 1


        voi += merge.variation_of_information(predict, gold)
        for i in predict:
            order_sent = sorted(i)
            tmp_text=""
            for j in order_sent:
                if tmp_text!="":
                    tmp_text+=" "+data[j][2]
                else:
                    tmp_text = data[j][2]
            test_text.append(tmp_text)
            test_fname.append(file)
    with open("test_output_simple/fix_"+epoch+"_cluster.json", 'w') as file:
        json.dump([test_text,test_fname], file)
    print("precision:",str(p/count))
    print("recall",str(r/count))
    print("f1",str(f/count))

    print("bcubed precision",str(bprecision/count))
    print("bcubed recall",str(brecall/count))
    print("bcubed f1",str(bfscore/count))
    print("voi",str(voi/count))

def main():
    global args
    taskname = args.taskname
    no_below = args.no_below
    no_above = args.no_above
    num_epochs = args.num_epochs
    n_topic = args.n_topic
    n_cpu = cpu_count()-2 if cpu_count()>2 else 2
    bkpt_continue = args.bkpt_continue
    use_tfidf = args.use_tfidf
    rebuild = args.rebuild
    batch_size = args.batch_size
    criterion = args.criterion
    auto_adj = args.auto_adj
    ckpt = args.ckpt
    lang = args.lang
    split_num = args.cross_split
    topic_file = args.topic_file

    device = torch.device('cuda')

    net = Net().to(device)
    net.load_state_dict(torch.load('ckpt/'+save_model_dir))

    print("split number: 1")
    #train, label = read_data('Neural_Topic_Models/pair_ntm_sentence_2/train/')
    #val, val_label = read_data('Neural_Topic_Models/pair_ntm_sentence_2/val/')
    train, label = read_data('pair_sentence_1/train/')
    val, val_label = read_data('pair_sentence_1/val/')
    test, test_label = read_data('pair_sentence_1/test/')
    train_data = CustomDataset(train, label)
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

    val_data = CustomDataset(val, val_label)
    val_loader = DataLoader(dataset=val_data, batch_size=8,shuffle=True)

    test_data = CustomDataset(test, test_label)
    test_loader = DataLoader(dataset=test_data, batch_size=8,shuffle=True)
    #print(len(label))
    #print(len(val_label))

    optimizer = optim.Adam(net.parameters())
    for epoch in range(0):  # loop over the dataset multiple times
        total_acc=0
        running_loss = 0.0
        count = 0
        target_true, correct_true, predicted_true,acc = 0,0,0,0
        net.train()
        for x_batch, y_batch in train_loader:
            x_train = []

            for i in range(len(x_batch[0])):
                embed = get_feature_vec(x_batch[0][i],x_batch[1][i],x_batch[2][i],x_batch[3][i])
                embed.extend([x_batch[4][i],x_batch[5][i]])
                x_train.append(embed)

            inputs = torch.tensor(x_train).float().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs).squeeze()

            bceweight = torch.where(y_batch > 0, 3, 1).to(device)
            criterion = torch.nn.BCELoss(weight = bceweight)
            criterion.to(device)

            loss = criterion(outputs, (y_batch).float().to(device))

            loss.backward()
            optimizer.step()

            predicted_classes = torch.round(outputs)
            target_classes = (y_batch).float().to(device)

            target_true += torch.sum(target_classes == 1).float()
            predicted_true += torch.sum(predicted_classes==1).float()
            correct_true += torch.sum(
                (predicted_classes == target_classes).float() * (predicted_classes == 1)).float()
            acc += torch.sum(predicted_classes==(y_batch).float().to(device))
            count += len(y_batch)


            running_loss += loss.item()

        print("train set")
        recall = correct_true / target_true
        precision = correct_true / predicted_true
        f1_score = 2 * precision * recall / (precision + recall)

        target_true, correct_true, predicted_true = 0,0,0
        print("epoch",str(epoch),"recall",recall.cpu().numpy(),"precision",precision.cpu().numpy(),"f1",f1_score.cpu().numpy())
        print("acc", acc.cpu().numpy()/count)
        print("loss",running_loss/count)
        evaluation(net,val_loader,device,'valid')
        evaluation(net,test_loader,device,'test')
        test_cluster(net,device,'pair_sentence_filter_1/test/',epoch)
    test_cluster(net,device,'pair_sentence_filter_'+str(split_num)+'/test/',str(split_num)+'test')
    test_cluster(net,device,'pair_sentence_filter_'+str(split_num)+'/val/',str(split_num)+'val')
    #print('train')
    test_cluster(net,device,'pair_sentence_filter_'+str(split_num)+'/train/',str(split_num)+'train')
    path = 'pair_net_simple_new.ckpt'
    #torch.save(net.state_dict(), path)
    print('Finished Training')

if __name__ == "__main__":
    main()
