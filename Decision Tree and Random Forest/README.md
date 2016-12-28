## Decision Tree 
This decision tree model can be used in classification of dataset with
categorical attributes. The input dataset should be in the following format:

(label) (attribute1):(value1) (attribute2):(value2) ...

Several train and test datasets are provided here. To run this program, type
in command line:

python DecisionTree.py trainingFile testFile

Some details:
* for each split, each categorical value of the splitting feature will have a branch
* gini-impurity is used to select splitting feature. Similar to GainRatio in C4.5, 
the reduction in impurity is normalized to consider different numbers of levels 
in categorical features
* default maximum depth is 5
* the output is the confusion matrix of the classifier on testing data, where the 
i-th row, j-th number is the number of data points in test data where the actual 
label is i and predicted label is j

## Random Forest
To run this program, type
in command line: 

python RandomForest.py trainingFile testFile

Some details:
* maximum depth for decision tree = 0.8*n_feature
* p = 0.4, which is the proportion of features randomly sampled as candidates 
at each split
* ntree = 100
* output is the same with decision tree
