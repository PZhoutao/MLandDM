# Implementaion of random forest
# python 3.5

from DecisionTree import *
import random

def build_rf(data, ntree, p):
    rf = []
    n_record = len(data)
    features = list(range(1,len(data[0])))
    max_dep = max(1, round(len(features)*0.8))
    for i in range(ntree):
        sample = [random.randint(0, n_record-1) for i in range(n_record)]
        bootstrap = [data[i] for i in sample]
        dtree = build_tree(bootstrap, features, p, 0, max_dep)
        rf.append(dtree)
    return rf

def make_pred_rf(record, rf):
    pred_rf = {}
    for dtr in rf:
        pred = make_pred(record, dtr)
        if pred in pred_rf:
            pred_rf[pred] += 1
        else:
            pred_rf[pred] = 1
    major_class = -1
    major_class_size = -1
    for k,v in pred_rf.items():
        if v > major_class_size:
            major_class = k
            major_class_size = v
    return major_class

def conf_matrix_rf(data, n_class, rf):
    pred = []
    labels = [x[0] for x in test]
    for record in test:
        pred.append(make_pred_rf(record, rf))
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
    p = 0.4
    rf = build_rf(train, 100, p)
    conf_m = conf_matrix_rf(test, n_class, rf)
    for k in range(len(conf_m)):
        strlst = [str(x) for x in conf_m[k]]
        output = " ".join(strlst)
        print(output)

