# -*- coding: utf-8 -*-
"""
author:LU XUE-JIN
"""
import random
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.layers import Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from keras import backend as K
import numpy
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn import preprocessing
from sklearn import svm, metrics
#numpy.random.seed(10)
import sys
import all_22277_list
SAME_SYMPTOMS = 1 #進階Label,兩個病人在該時間點都染病或未染病,
DIFF_SYMPTOMS = 0 #進階Label,兩個病人在該時間點一個有病一個沒病
POSITIVE = 1 #基礎label,表示該病人是有染病
NEGATIVE = 0 #基礎label,表示該病人是未染病
is_suv = 2 #總共有陰性、陽性兩種基礎label
all_df = pd.read_csv("rma_H1N1_leave_"+sys.argv[1]+"_out.csv",sep='\t',encoding='utf-8')#訓練資料集
standar_df = pd.read_csv("H1N1_health_"+sys.argv[2]+".txt",sep='\t',encoding='utf-8')#健康樣本資料集
test_df = pd.read_csv("rma_H1N1_"+sys.argv[1]+".txt",sep='\t',encoding='utf-8')#測試資料資料集
cols = all_22277_list.cols#資料集的features
#將沒使用到的features剔除
all_df=all_df[cols]
standar_df = standar_df[cols]
test_df=test_df[cols]

#train data 和 validation data 8:2分配
msk = numpy.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
val_df = all_df[~msk]

#data normalization
def PreprocessData(raw_df):
   
    df=raw_df.drop(['ID'], axis=1)#移除name欄位
    ndarray = df.values#dataframe轉換為array
    Features = ndarray[:,1:] 
    Label = ndarray[:,0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)    
    
    return scaledFeatures,Label

#依序將train validation test 健康樣本的features,label提出
train_Features,train_Label=PreprocessData(train_df)
val_Features,val_Label=PreprocessData(val_df)
test_Features,test_Label=PreprocessData(test_df)
standar_Features,standar_Label=PreprocessData(standar_df)
#歐式距離
def euclid_dis(vects):
    x,y = vects
    sum_square = K.sum(K.square(x-y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
#loss
def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

#create training data pair
def create_pairs(x, train_symptoms):#x:features train_symptoms:基礎label
    n=min([len(train_symptoms[NEGATIVE]),len(train_symptoms[POSITIVE])])#陽性語陰性label取最小讓正負樣本數相同
    random_pairs = []
    random_labels = []
    negative_negative_pairs = [] #兩個陰性樣本的配對
    negative_positive_pairs = [] #一陰性一陽性樣本的配對
    positive_positive_pairs = [] #一陽性一陰性樣本的配對
    positive_negative_pairs = [] #兩個陽性樣本的配對
    negative_negative_pairs = create_negative_negative_pairs(x, train_symptoms)
    negative_positive_pairs = create_negative_positive_pairs(x, train_symptoms)
    positive_positive_pairs = create_positive_positive_pairs(x, train_symptoms)
    positive_negative_pairs = create_positive_negative_pairs(x, train_symptoms)
    for i in range(n*n//4):#可以透過減少迴圈次數決定配對資料量的大小,將剛剛照順序製造出的pair順序打亂,再分別給予
        random_pairs += [negative_negative_pairs[i],negative_positive_pairs[i]]
        random_labels += [SAME_SYMPTOMS,DIFF_SYMPTOMS]
        random_pairs += [positive_positive_pairs[i],positive_negative_pairs[i]]
        random_labels += [SAME_SYMPTOMS,DIFF_SYMPTOMS]
    return numpy.array(random_pairs), numpy.array(random_labels)

def create_negative_negative_pairs(x, train_symptoms):#製作兩個基本標籤都是陰性的pair
    negative_negative_pairs = []
    random_negative_negative_pairs = []
    n=min([len(train_symptoms[NEGATIVE]),len(train_symptoms[POSITIVE])])#避免因性與陽性資料量不一致
    for i in range(len(train_symptoms[NEGATIVE])):#用每兩個資料就湊成一對來擴增訓練資料,製造出所有可能的配對方法的pair
        for j in range(len(train_symptoms[NEGATIVE])):
            z1, z2 = train_symptoms[NEGATIVE][i], train_symptoms[NEGATIVE][j]
            negative_negative_pairs += [[x[z1], x[z2]]]

    negative_negative_pairs_index = random.sample(range(0,len(negative_negative_pairs)), n*n//4)#將剛剛製作出來的pair順序打散
    for i in range(len(negative_negative_pairs_index)):#製作隨機順序的pair
        random_negative_negative_pairs += [negative_negative_pairs[negative_negative_pairs_index[i]]]
    return random_negative_negative_pairs

def create_negative_positive_pairs(x, train_symptoms):#製作兩個基本標籤是一個陰性一個陽性的pair,製作方式與上面方法相同
    negative_positive_pairs = []
    random_negative_positive_pairs = []
    k = 0
    n=min([len(train_symptoms[NEGATIVE]),len(train_symptoms[POSITIVE])])
    for i in range(len(train_symptoms[NEGATIVE])):
        for j in range(len(train_symptoms[NEGATIVE])):
            if(k < len(train_symptoms[POSITIVE])):    
                z1, z2 = train_symptoms[NEGATIVE][i], train_symptoms[POSITIVE][k]
                negative_positive_pairs += [[x[z1], x[z2]]]
                k += 1
            else:
                k = 0
                z1, z2 = train_symptoms[NEGATIVE][i], train_symptoms[POSITIVE][k]
                negative_positive_pairs += [[x[z1], x[z2]]]
    negative_positive_pairs_index = random.sample(range(0,len(negative_positive_pairs)), n*n//4)
    for i in range(len(negative_positive_pairs_index)):
        random_negative_positive_pairs += [negative_positive_pairs[negative_positive_pairs_index[i]]]
    return random_negative_positive_pairs

def create_positive_positive_pairs(x, train_symptoms):#製作兩個基本標籤都是陽性的pair,製作方式與上面方法相同
    positive_positive_pairs = []
    random_positive_positive_pairs = []
    n=min([len(train_symptoms[NEGATIVE]),len(train_symptoms[POSITIVE])])
    for i in range(len(train_symptoms[POSITIVE])):
        for j in range(len(train_symptoms[POSITIVE])):
            z1, z2 = train_symptoms[POSITIVE][i], train_symptoms[POSITIVE][j]
            positive_positive_pairs += [[x[z1], x[z2]]]
    positive_positive_pairs_index = random.sample(range(0,len(positive_positive_pairs)), n*n//4)
    for i in range(len(positive_positive_pairs_index)):
        random_positive_positive_pairs += [positive_positive_pairs[positive_positive_pairs_index[i]]]
    return random_positive_positive_pairs

def create_positive_negative_pairs(x, train_symptoms):#製作兩個基本標籤是一個陰性一個陰性的pair,製作方式與上面方法相同
    positive_negative_pairs = []
    random_positive_negative_pairs = []
    k = 0
    n=min([len(train_symptoms[NEGATIVE]),len(train_symptoms[POSITIVE])])
    for i in range(len(train_symptoms[POSITIVE])):
        for j in range(len(train_symptoms[POSITIVE])):
            if(k < len(train_symptoms[NEGATIVE])):    
                z1, z2 = train_symptoms[POSITIVE][i], train_symptoms[NEGATIVE][k]
                positive_negative_pairs += [[x[z1], x[z2]]]
                k += 1
            else:
                k = 0
                z1, z2 = train_symptoms[POSITIVE][i], train_symptoms[NEGATIVE][k]
                positive_negative_pairs += [[x[z1], x[z2]]]
                k = 0
    positive_negative_pairs_index = random.sample(range(0,len(positive_negative_pairs)), n*n//4)
    for i in range(len(positive_negative_pairs_index)):
        random_positive_negative_pairs += [positive_negative_pairs[positive_negative_pairs_index[i]]]
    return random_positive_negative_pairs

def create_test_pairs(x, train_symptoms, y, test_symptoms):#x:訓練資料的features train_symptoms:訓練資料的基礎label y:validation data features ,test_symptoms:validation data basic label
    
    positive_random_index = []#利用index避免抓到重複的資料
    negative_random_index = []
    test_pairs = []
    test_labels = []
    #製作出一邊為看過的training data,另一邊為8:2分剩下的validation data
    if(len(test_symptoms[NEGATIVE])>0):
        positive_random_index = random.sample(range(0,len(train_symptoms[POSITIVE])), len(test_symptoms[NEGATIVE]))
        negative_random_index = random.sample(range(0,len(train_symptoms[NEGATIVE])), len(test_symptoms[NEGATIVE]))
        for i in range(len(test_symptoms[NEGATIVE])):
                z1, z2 = test_symptoms[NEGATIVE][i], train_symptoms[NEGATIVE][negative_random_index[i]]
                test_pairs += [[y[z1], x[z2]]]
                z1, z2 = test_symptoms[NEGATIVE][i], train_symptoms[POSITIVE][positive_random_index[i]]
                test_pairs += [[y[z1], x[z2]]]
                test_labels += [SAME_SYMPTOMS,DIFF_SYMPTOMS]
    if(len(test_symptoms[NEGATIVE])>0):
        positive_random_index = random.sample(range(0,len(train_symptoms[POSITIVE])), len(test_symptoms[NEGATIVE]))
        negative_random_index = random.sample(range(0,len(train_symptoms[NEGATIVE])), len(test_symptoms[NEGATIVE]))
        for i in range(len(test_symptoms[NEGATIVE])):
                z1, z2 = train_symptoms[NEGATIVE][negative_random_index[i]], test_symptoms[NEGATIVE][i]
                test_pairs += [[x[z1], y[z2]]]
                z1, z2 = train_symptoms[POSITIVE][positive_random_index[i]], test_symptoms[NEGATIVE][i]
                test_pairs += [[x[z1], y[z2]]]
                test_labels += [SAME_SYMPTOMS,DIFF_SYMPTOMS]
    if(len(test_symptoms[POSITIVE])>0):
        positive_random_index = random.sample(range(0,len(train_symptoms[POSITIVE])), len(test_symptoms[POSITIVE]))
        negative_random_index = random.sample(range(0,len(train_symptoms[NEGATIVE])), len(test_symptoms[POSITIVE]))
        for i in range(len(test_symptoms[POSITIVE])):
                z1, z2 = test_symptoms[POSITIVE][i], train_symptoms[POSITIVE][positive_random_index[i]]
                test_pairs += [[y[z1], x[z2]]]
                z1, z2 = test_symptoms[POSITIVE][i], train_symptoms[NEGATIVE][negative_random_index[i]]
                test_pairs += [[y[z1], x[z2]]]
                test_labels += [SAME_SYMPTOMS,DIFF_SYMPTOMS]
    if(len(test_symptoms[POSITIVE])>0):
        positive_random_index = random.sample(range(0,len(train_symptoms[POSITIVE])), len(test_symptoms[POSITIVE]))
        negative_random_index = random.sample(range(0,len(train_symptoms[NEGATIVE])), len(test_symptoms[POSITIVE]))
        for i in range(len(test_symptoms[POSITIVE])):
                z1, z2 = train_symptoms[POSITIVE][positive_random_index[i]], test_symptoms[POSITIVE][i]
                test_pairs += [[x[z1], y[z2]]]
                z1, z2 = train_symptoms[NEGATIVE][negative_random_index[i]], test_symptoms[POSITIVE][i]
                test_pairs += [[x[z1], y[z2]]]
                test_labels += [SAME_SYMPTOMS,DIFF_SYMPTOMS]
    return numpy.array(test_pairs), numpy.array(test_labels)

def create_standar_test_pairs(x, standar_symptoms, y, test_symptoms): #x:放健康樣本的features,standar_symptoms:放健康樣本的basic label,y:test data features,test_symptoms:test data basic label
    #製作一邊為test data 另一邊為健康樣本的pair
    test_pairs = []
    test_labels = []
    if(len(test_symptoms[NEGATIVE])>0):
        for i in range(len(standar_symptoms[NEGATIVE])):
            for j in range(len(test_symptoms[NEGATIVE])):
                z1, z2 = standar_symptoms[NEGATIVE][i], test_symptoms[NEGATIVE][j]
                test_pairs += [[x[z1], y[z2]]]
                test_labels += [SAME_SYMPTOMS]
    if(len(test_symptoms[POSITIVE])>0):
        for i in range(len(standar_symptoms[NEGATIVE])):
            for j in range(len(test_symptoms[POSITIVE])):
                z1, z2 = standar_symptoms[NEGATIVE][i], test_symptoms[POSITIVE][j]
                test_pairs += [[x[z1], y[z2]]]
                test_labels += [DIFF_SYMPTOMS]
         
    return numpy.array(test_pairs), numpy.array(test_labels)

def create_base_net(input_shape):#製作孿生網路
    input = Input(shape = input_shape)
    x = Dense(50, activation='relu')(input)
    x = Dense(50, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(input, x)
    return model

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return numpy.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

input_shape = train_Features.shape[1:]

standar_symptoms = [numpy.where(standar_Label == i)[0] for i in range(is_suv)]#提取健康樣本的label
train_symptoms = [numpy.where(train_Label == i)[0] for i in range(is_suv)]#提取訓練樣本的label
train_pairs, train_y = create_pairs(train_Features, train_symptoms)#製作好的pair和pair的label
val_symptoms = [numpy.where(val_Label == i)[0] for i in range(is_suv)]
val_pairs, val_y = create_test_pairs(train_Features, train_symptoms, val_Features, val_symptoms)

test_symptoms = [numpy.where(test_Label == i)[0] for i in range(is_suv)]
test_pairs, test_y = create_standar_test_pairs(standar_Features, standar_symptoms, test_Features, test_symptoms)
# network definition
base_network = create_base_net(input_shape)
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclid_dis,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])



model = Model([input_a, input_b], distance)

import keras
rms = RMSprop()
opt = keras.optimizers.Adam(lr=0.00001)

model.compile(loss=contrastive_loss, optimizer=opt, metrics=[accuracy])
train_history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_y,
          batch_size=200,
          epochs=25,
          validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_y)
          ,verbose=0)

# compute final accuracy on training and test sets
y_pred_train = model.predict([train_pairs[:, 0], train_pairs[:, 1]])
train_acc = compute_accuracy(train_y, y_pred_train)
y_pred_val = model.predict([val_pairs[:, 0], val_pairs[:, 1]])
val_acc = compute_accuracy(val_y, y_pred_val)
test_pred = []
y_pred_test = model.predict([test_pairs[:, 0], test_pairs[:, 1]])
test_acc = compute_accuracy(test_y, y_pred_test)
for i in y_pred_test:
    if(i > 0.5):
        #print(i)
        test_pred.append(0)
    else:
        #print(i)
        test_pred.append(1)
        
test_TP=0
test_TN=0
test_FP=0
test_FN=0

for j in range(len(test_pred)):
    if(test_pred[j]==SAME_SYMPTOMS and test_pred[j]==test_y[j]):
        test_TP=test_TP+1
    else:
        test_TP=test_TP+0
    
    if(test_pred[j]==DIFF_SYMPTOMS and test_pred[j]==test_y[j]):
        test_TN=test_TN+1
    else:
        test_TN=test_TN+0
    
    if(test_y[j]==DIFF_SYMPTOMS and test_pred[j]==SAME_SYMPTOMS):
        test_FP=test_FP+1
    else:
        test_FP=test_FP+0
    
    if(test_y[j]==SAME_SYMPTOMS and test_pred[j]==DIFF_SYMPTOMS):
        test_FN=test_FN+1
    else:
        test_FN=test_FN+0
print("test_TP:",test_TP)
print("test_TN:",test_TN)
print("test_FP:",test_FP)
print("test_FN:",test_FN)

print('* Accuracy on training set: %0.2f%%' % (100 * train_acc))
print('* Accuracy on validation set: %0.2f%%' % (100 * val_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * test_acc))

with open("H1N1_no_"+sys.argv[1]+"_result_"+sys.argv[3]+".txt", 'w', newline='') as f:
    writer = f.write("TP:%d\n" % test_TP)
    writer = f.write("TN:%d\n" % test_TN)
    writer = f.write("FP:%d\n" % test_FP)
    writer = f.write("FN:%d\n" % test_FN)
    writer = f.write('* Accuracy on training set: %0.2f%%' % (100 * train_acc))
    writer = f.write('* Accuracy on validation set: %0.2f%%' % (100 * val_acc))
    writer = f.write('* Accuracy on test set: %0.2f%%' % (100 * test_acc))
    writer = f.write('\n')
    for i in test_y:
        writer = f.write("test_y:%d\n" % i)
    for j in y_pred_test:
        writer = f.write("y_pred_test:%f\n" % j)
    
import matplotlib.pyplot as plt
def show_train_history(train_history,train,test):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[test])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
#print training result figure
#show_train_history(train_history,'accuracy','val_accuracy')
#show_train_history(train_history,'loss','val_loss') 

