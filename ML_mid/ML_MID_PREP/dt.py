import numpy as np
import pandas as pd

def entropy(y):
    unique,counts =np.unique(y,return_counts=True)
    prob = counts/sum(counts)
    return -np.sum(prob*np.log2(prob+-1e9))

def information_gain(x,y,threshold):
    n=len(y)
    xleft_ind =x <=threshold
    xright_ind= x>threshold
    if (sum(xleft_ind)==0 or sum(xright_ind)==0):
        return 0
    leftentropy=entropy(y[xleft_ind])
    rightentropy= entropy(y[xright_ind])

    nl,nr = sum(xleft_ind),sum(xright_ind)

    weighted_entropy =(nl/n)*leftentropy+(nr/n)*rightentropy

    return entropy(y)-weighted_entropy

class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,value=None):
        self.feature =feature
        self.threshold =threshold
        self.left =left
        self.right =right
        self.value =value

class DT:
    def __init__(self,max_depth=None):
        self.max_depth =max_depth
        self.root =None
    
    def fit(self,x,y):
        self.root=self.grow_tree(x,y)
    
    def most_common_label(self,y):
        unique,counts =np.unique(y,return_counts= True)
        return unique[np.argmax(counts)]
    
    def grow_tree(self,x,y,depth=0):

        n_samples,n_features =x.shape
        n_labels= len(np.unique(y))

        if (depth>= self.max_depth or n_labels ==1 or n_samples==0):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)
        
        best_gain=-1
        split_feature,split_thresh =None,None
        for feature_id in range(n_features):
            x_column= x.iloc[:,feature_id]
            thresholds =np.unique(x_column)

            for threshold in thresholds:
                gain =information_gain(x_column,y,threshold)
                if gain>best_gain:
                    best_gain=gain
                    split_thresh=threshold
                    split_feature=feature_id
        
        left_indices = x[:,split_feature]<=split_thresh
        right_indices = x[:,split_feature]>split_thresh

        left= self.grow_tree(x[left_indices],y[left_indices],depth+1)
        right =self.grow_tree(x[right_indices],y[right_indices],depth+1)

        return Node(split_feature,split_thresh,left,right)
    
    def traverse_tree(self,x,node):
        if node.value is not None:
            return node.value
        if x[node.feature]<= node.threshold:
            return self.traverse_tree(x,node.left)
        return self.traverse_tree(x,node.right)
    
    def predict(self,x):
        return np.array([self.traverse_tree(x,self.root)])
    


