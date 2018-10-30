"""
use different algorithm to solve the digit recognizer
"""
import os
import numpy as np
import pandas as pd
import  time
import  matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
size_img = 28

threshold_color = 100/255
file = open('train.csv')
data_train = pd.read_csv(file)

y_train = np.array(data_train.iloc[:,0])
x_train = np.array(data_train.iloc[:,1:])


file = open('test.csv')
data_test = pd.read_csv(file)
x_test = np.array(data_test)

n_features_train = x_train.shape[1]
n_samples_train = x_train.shape[0]
n_features_test = x_test.shape[1]
n_samples_test = x_test.shape[0]
print(n_features_train,n_samples_train,n_features_test,n_samples_test)
print(x_train.shape, y_train.shape, x_test.shape)

"""
show the image 
"""
def show_img(x):
    plt.figure(figsize= (8,7))
    if x.shape[0] > 100:
        print(x.shape[0])
        n_imgs = 16
        n_samples = x.shape[0]
        x = x.reshape(n_samples,size_img,size_img)
        for i in range(n_imgs):
            plt.subplot(4,4,i+1)
            plt.imshow(x[i])
        plt.show()
    else:
        plt.imshow(x)
        plt.show()

#show_img(x_train)
# show_img(x_test)

#Normalized
def int2float_grey(x):
    x = x/255
    return x
# resize and Cropping
def find_left_edge(x):
    edge_left = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for j in range(size_img):
            for i in range(size_img):
                if x[k,size_img*i +j] >= threshold_color:
                    edge_left.append(j)
                    break
            if len(edge_left) >k :
                break
    return edge_left

def find_right_edge(x):
    edge_right = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for j in range(size_img):
            for i in range(size_img):
                if x[k,size_img*i +(size_img -j-1)] >= threshold_color:
                    edge_right.append(size_img - 1- j)
                    break
            if len(edge_right) >k :
                break
    return edge_right


def find_top_edge(x):
    edge_top = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for j in range(size_img):
            for i in range(size_img):
                if x[k,size_img*i +j] >= threshold_color:
                    edge_top.append(i)
                    break
            if len(edge_top) >k :
                break
    return edge_top

def find_bottom_edge(x):
    edge_bottom = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for j in range(size_img):
            for i in range(size_img):
                if x[k,size_img*(size_img-1 -i) +j] >= threshold_color:
                    edge_bottom.append(size_img-1 -i)
                    break
            if len(edge_bottom) >k :
                break
    return edge_bottom

from skimage import transform
def stretch_image(x):
    edge_left = find_left_edge(x)
    edge_right = find_right_edge(x)
    edge_top  = find_top_edge(x)
    edge_bottom = find_bottom_edge(x)
    n_samples = x.shape[0]
    x = x.reshape(n_samples, size_img,size_img)

    for i in range(n_samples):
        x[i] = transform.resize(x[i][edge_top[i]:edge_bottom[i]+1,edge_left[i]:edge_right[i]+1],(size_img,size_img))
    x = x.reshape(n_samples,size_img **2)
    show_img(x)

# feature selection
def get_threshold(x_train,x_test):
    selector = VarianceThreshold(threshold= 0).fit(x_train)
    x_train = selector.transform(x_train)
    x_test = selector.transform(x_test)
    print("x_train.shape:",x_train.shape)
    print("x_test.shape:", x_test.shape)
    return x_train, x_test


from sklearn.decomposition import  PCA
def get_PCA(x_train,x_test):
    pca = PCA(n_components=0.95)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    return x_train,x_test


x_train = int2float_grey(x_train)
x_test = int2float_grey(x_test)
# stretch_image(x_train)
# stretch_image(x_test)
#x_train,x_test = get_threshold(x_train,x_test)
x_train,x_test = get_PCA(x_train,x_test)  #use pca


def general_function(mod_name, model_name):
    y_pred = model_train_predict(mod_name, model_name)
    out_predict(y_pred,model_name)


from sklearn.model_selection import cross_val_score
def model_train_predict(mod_name, model_name):

    start_time = time.time()
    import_mod = __import__(mod_name, fromlist= str(True))
    if hasattr(import_mod, model_name):
        f = getattr(import_mod, model_name)
    else:
        print('404')
        return []

    clf = f()
    clf.fit(x_train,y_train)
    y_predict = clf.predict(x_train)
    print("mod_name.model_name  is : ",mod_name,model_name)
    end_time = time.time()
    print("train's time is :", end_time - start_time)
    get_acc(y_predict,y_train)
    scores = cross_val_score(clf,x_train,y_train,cv=5)
    print("Accuracy :%0.2f (+- %0.2f)"%(scores.mean(),scores.std()*2))

    y_predict = clf.predict(x_test)
    return y_predict

# get accuracy
def get_acc(y_pred,y_train):
    right_num = (y_train==y_pred).sum()
    print("acc ;" ,right_num/n_samples_train)

#output the csv
def out_predict(y_pred,model_name):
    print(y_pred)
    data_pred = {'ImageId':range(1,n_samples_test +1),'Label': y_pred}
    data_pred = pd.DataFrame(data_pred)
    data_pred.to_csv("Pre_%s.csv"%model_name,index=False)


from  sklearn.naive_bayes import  GaussianNB, MultinomialNB,BernoulliNB
mod_name = "sklearn.naive_bayes"
model_name = "GaussianNB"
general_function(mod_name,model_name)

#not use pca ，beacuse the inpout must be non-negative
# model_name = "MultinomialNB" 
# general_function(mod_name,model_name)

model_name = "BernoulliNB"
general_function(mod_name,model_name)

from sklearn.svm import SVC
mod_name = "sklearn.svm"
model_name = "SVC"

general_function(mod_name,model_name)

from sklearn.neighbors import  KNeighborsClassifier

mod_name = "sklearn.neighbors"
model_name = "KNeighborsClassifier"
general_function(mod_name,model_name)

from sklearn.cluster import  KMeans
mod_name = "sklearn.cluster"
model_name = "KMeans"
general_function(mod_name,model_name)

from sklearn.tree import  DecisionTreeClassifier
mod_name = "sklearn.tree"
model_name = "DecisionTreeClassifier"
general_function(mod_name,model_name)

from sklearn.ensemble import  RandomForestClassifier
mod_name = "sklearn.ensemble"
model_name = "RandomForestClassifier"
general_function(mod_name,model_name)
