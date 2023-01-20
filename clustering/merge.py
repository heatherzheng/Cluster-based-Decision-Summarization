from scipy import spatial
import bcubed
from math import log

def variation_of_information(X, Y):
  n = float(sum([len(x) for x in X]))
  sigma = 0.0
  for x in X:
    p = len(x) / n
    for y in Y:
      q = len(y) / n
      r = len(set(x) & set(y)) / n
      if r > 0.0:
        sigma += r * (log(r / p, 2) + log(r / q, 2))
  return abs(sigma)

def sim_LDA(lst1,lst2,val_dic,bar):
    sim = 0
    for i in lst1:
        for j in lst2:
            #sim += val_dic[(min(i,j),max(i,j))]
            if val_dic[(min(i,j),max(i,j))]>=bar:
              sim += 1
    return sim/len(lst1)/len(lst2)

def merge_corpus(corpus_index,val_dict,bar=0.5):
    matrices = corpus_index

    thre = None
    while thre==None or thre >= bar:
        max_val = -1
        max_pos = None

        for m in range(len(matrices)):
            for n in range(len(matrices)):
                if m >= n:
                    continue
                val = sim_LDA(matrices[m], matrices[n],val_dict,bar)

                if val>max_val or max_pos==None:
                    max_val = val
                    max_pos = (m,n)
                if val==max_val and max_pos!=None:
                  if abs(m-n)<abs(max_pos[0]-max_pos[1]):
                    max_val = val
                    max_pos = (m,n)

        thre = max_val

        if max_val >= bar:
            tmp = matrices[max_pos[0]]
            tmp.extend(matrices[max_pos[1]])
            matrices[max_pos[0]] = tmp
            del matrices[max_pos[1]]

    return matrices


def lst_to_pair(lst):
    pair = []
    for i in lst:
        if len(i)==1:
            continue

        for j in range(len(i)):
            for k in range(j+1, len(i)):
                pair.append((min(i[j],i[k]),max(i[j],i[k])))

    return pair

def get_pair_score(predict,expect):
    predict = lst_to_pair(predict)
    expect = lst_to_pair(expect)


    tp = 0
    for i in predict:
        if i in expect:
            tp +=1
    if expect:
        r = tp/len(expect)
    else:
        r = 1

    if predict:
        p = tp/len(predict)
    else:
        p= 1

    if p==0 and r==0:
        return 0,0,0
    #print("pairwise score:","precision",str(p),"recall",str(r),"f1",str(2*r*p/(r+p)))
    return p,r,2*r*p/(r+p)
