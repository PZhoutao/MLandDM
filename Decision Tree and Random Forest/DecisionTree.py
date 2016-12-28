# Implementaion of decision tree classifier
# python 3.5

import sys
import random
import math

def readfile(file):
    ret = []
    f = open(file, 'r')
    for line in f.readlines():
        line = line.strip().split()
        if len(line) == 0:
            continue
        label = eval(line.pop(0))
        line = [eval(x.split(':')[1]) for x in line]
        record = [label] + line
        ret.append(record)
    return ret

def giniIndex(data):
    labels = [x[0] for x in data]
    uniq_label = set(labels)
    pi2 = 0
    for lb in uniq_label:
        pi2 += (labels.count(lb))**2
    return 1-pi2/(len(labels))**2

def best_split_feature(data, feature_split):
    cur_best_feature = 0
    cur_best_gain = float('-inf')
    cur_best_data_split = {}
    gini_before = giniIndex(data)
    D = len(data)
    for f in feature_split:
        data_split = {}
        for record in data:
            if record[f] in data_split:
                data_split[record[f]].append(record)
            else:
                data_split[record[f]] = [record]
        if len(data_split) == 1:
            continue
        gini = 0
        splitInfo = 0
        for k,v in data_split.items():
            gini += giniIndex(v)*len(v)/D
            splitInfo -= len(v)/D*(math.log2(len(v)/D))
        gain = gini_before - gini
        gain_ratio = gain / splitInfo
        if gain_ratio > cur_best_gain:
            cur_best_gain = gain_ratio
            cur_best_data_split = data_split
            cur_best_feature = f
    return (cur_best_feature, cur_best_data_split)

def isLeaf(data, feature_split):
    if len(feature_split) == 0:
        return True
    labels = [x[0] for x in data]
    if len(set(labels)) == 1:
        return True
    sameVal = True
    for f in feature_split:
        val = [x[f] for x in data]
        if len(set(val)) != 1:
            sameVal = False
            break
    return sameVal

def createLeaf(data):
    labels = [x[0] for x in data]
    pred = -1
    major_count = -1
    for i in set(labels):
        val_count = labels.count(i)
        if val_count > major_count:
            major_count = val_count
            pred = i
    leaf = {
        'splitting_feature' : None,
        'children' : None,
        'is_leaf': True,
        'pred' : pred,
        'default' : None
    }
    return leaf

def build_tree(data, feature_split, p, cur_dep, max_dep=5):
    if (len(feature_split) == 0):
        return createLeaf(data)
    n_to_choose = max(1, round(p*len(feature_split)))
    sampled_feature = sorted(random.sample(feature_split, n_to_choose))
    if (cur_dep == max_dep or isLeaf(data, sampled_feature)):
        return createLeaf(data)
    split_feature, data_split = best_split_feature(data, sampled_feature)
    remain_feature = feature_split[:]
    remain_feature.remove(split_feature)
    children = {}
    major_child = -1
    major_child_size = -1
    for k,v in data_split.items():
        child = build_tree(v, remain_feature[:], p, cur_dep+1, max_dep)
        children[k] = child
        if len(v) > major_child_size:
            major_child = k
            major_child_size = len(v)
    return {
        'splitting_feature' : split_feature,
        'children' : children,
        'is_leaf' : False,
        'pred' : None,
        'default' : major_child
    }

def make_pred(record, dtree):
    if not dtree['is_leaf']:
        split_feature = dtree['splitting_feature']
        val = record[split_feature]
        if val not in dtree['children']:
            val = dtree['default']
        return make_pred(record, dtree['children'][val])
    else:
        return dtree['pred']

def conf_matrix(test, n_class, dtree):
    pred = []
    labels = [x[0] for x in test]
    for record in test:
        pred.append(make_pred(record, dtree))
    conf_m = []
    for i in range(n_class):
        conf_m.append([0]*n_class)
    for r in range(len(pred)):
        i = labels[r]
        j = pred[r]
        conf_m[i-1][j-1] += 1
    return conf_m

if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    train = readfile(train_file)
    test = readfile(test_file)
    labeltrain = [x[0] for x in train]
    labeltest = [x[0] for x in test]
    n_class = max(labeltrain+labeltest)
    features = list(range(1,len(train[0])))
    max_dep = max(1, round(len(features)*0.8))
    p = 1.0
    dtree = build_tree(train, features, p, 0, max_dep)
    conf_m = conf_matrix(test, n_class, dtree)
    for k in range(len(conf_m)):
        strlst = [str(x) for x in conf_m[k]]
        output = " ".join(strlst)
        print(output)
