import argparse
import random,os,sys
import numpy as np
import csv
from scipy import stats
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import pandas as pd
import keras.backend as K
from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Input,InputLayer,Multiply,ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense,Activation,Dropout,Flatten,Concatenate
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras import optimizers,utils
from keras.constraints import max_norm
from keras import regularizers
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping,History
#from keras.utils import multi_gpu_model,plot_model
from tensorflow.keras.optimizers import Adam, SGD
from keras.models import model_from_json
import tensorflow as tf
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error, r2_score

from scipy.stats import pearsonr
from funs_sundar import *

from sklearn.model_selection import RepeatedKFold
import scipy.sparse as sp
import argparse
import pandas as pd
import matplotlib.pyplot as plt

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input,InputLayer,Multiply,ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D,Conv1D,MaxPooling1D
from keras.layers import Dense,Activation,Dropout,Flatten,Concatenate
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras.layers import Dropout,GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
#from layers.graph import GraphLayer,GraphConv

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, quniform
import tempfile


def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)
def data():
  DPATH = '/content/DeepCDR-master/data'
  Genomic_mutation_file = '%s/CCLE/final_sel_feat_mut.xlsx'%DPATH
  CNV_file = '%s/CCLE/final_sel_feat_cnv.xlsx'%DPATH
  Cancer_response_exp_file = '%s/CCLE/DR_100-screened_drugs_imputed-new.xlsx'%DPATH
  Gene_expression_file = '%s/CCLE/final_sel_feat_mrna.xlsx'%DPATH
  Methylation_file = '%s/CCLE/final_sel_feat_meth.xlsx'%DPATH
  random.seed(0)
  cnv_feature,mutation_feature, gexpr_feature,methylation_feature, Y = MetadataGenerate(CNV_file,Genomic_mutation_file,Gene_expression_file,Methylation_file,False)
  X1=np.matrix(cnv_feature)
  X2=np.matrix(mutation_feature)
  X3=np.matrix(gexpr_feature)
  X4=np.matrix(methylation_feature)
  Y=np.matrix(Y)
  cv = RepeatedKFold(n_splits=2, n_repeats=1, random_state=1)
  for train_ix,test_ix in cv.split(X1):
    print(train_ix)
    X1_train,X2_train,X3_train,X4_train, X1_test,X2_test,X3_test,X4_test= X1[train_ix],X2[train_ix],X3[train_ix],X4[train_ix],X1[test_ix],X2[test_ix],X3[test_ix],X4[test_ix]
    Y_train,Y_test = Y[train_ix], Y[test_ix]
  X_train=[X1_train,X2_train,X3_train,X4_train]
  X_test=[X1_test,X2_test,X3_test,X4_test]
  return X_train,Y_train,X_test,Y_test  
     
   
  



def create_model(X_train,Y_train,X_test,Y_test,use_gexp,use_methy,regr):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    print("AM BACK")
    layer_1_size = {{quniform(12, 256, 4)}}
    l1_dropout = {{uniform(0.001, 0.7)}}
    params = {
        'l1_size': layer_1_size,
        'l1_dropout': l1_dropout
    }
    print(X_train[0].shape[-1])
    y_dim=Y_train.shape[-1]
    print("??????????????????????")
    cnv_input = Input(shape=(X_train[0].shape[-1],))
    mut_input = Input(shape=(X_train[1].shape[-1],))
    gexpr_input = Input(shape=(X_train[2].shape[-1],))
    methy_input = Input(shape=(X_train[3].shape[-1],))

    x_cnv = Dense(16)(cnv_input)
    x_cnv = Activation('tanh')(x_cnv)
    x_cnv = BatchNormalization()(x_cnv)
    x_cnv = Dropout(0.1)(x_cnv)
    x_cnv = Dense((y_dim),activation='relu')(x_cnv)

    x_mut = Dense(16)(mut_input)
    x_mut = Activation('tanh')(x_mut)
    x_mut = BatchNormalization()(x_mut)
    x_mut = Dropout(0.1)(x_mut)
    x_mut = Dense((y_dim),activation='relu')(x_mut)
        #gexp feature
    x_gexpr = Dense(64)(gexpr_input)
    x_gexpr = Activation('tanh')(x_gexpr)
    x_gexpr = BatchNormalization()(x_gexpr)
    x_gexpr = Dropout(0.1)(x_gexpr)
    x_gexpr = Dense((y_dim),activation='relu')(x_gexpr)
        #methylation feature
    x_methy = Dense(16)(methy_input)
    x_methy = Activation('tanh')(x_methy)
    x_methy = BatchNormalization()(x_methy)
    x_methy = Dropout(0.1)(x_methy)
    x_methy = Dense((y_dim),activation='relu')(x_methy)
    
    x = Concatenate()([x_cnv,x_mut,x_gexpr,x_methy])
    x = Dense(300,activation = 'relu')(x)
    x = Dropout({{uniform(0, 1)}})(x)
    x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
    x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)
    x = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
    x = MaxPooling2D(pool_size=(1,2))(x)
    x = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
    x = MaxPooling2D(pool_size=(1,3))(x)
    x = Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
    x = MaxPooling2D(pool_size=(1,3))(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    x = Dropout({{uniform(0, 1)}})(x)
    if regr:
      output = Dense((y_dim),name='output')(x)
    else:
      output = Dense(1,activation = 'sigmoid',name='output')(x)
    model  = Model(inputs=[cnv_input,mut_input,gexpr_input,methy_input],outputs=output)  
      
    model.compile(loss='mean_squared_error', metrics=['mse'],
                  optimizer={{choice(['adam', 'rmsprop', 'sgd'])}})
    x_train=[X_train[0],X_train[1],X_train[2],X_train[3]]
    
    result = model.fit(x=x_train,y=Y_train,
              batch_size={{choice([64, 128])}},
              epochs=150,
              verbose=2,
              validation_split=0.1)
    #get the highest validation accuracy of the training epochs
    x_test=[X_test[0],X_test[1],X_test[2],X_test[3]]
    print("RRRRRRRRRRRRRRRRRRR")
    y_pred = model.predict([X_test[0],X_test[1],X_test[2],X_test[3]], verbose=0)
    #tr_acc = compute_accuracy(Y_test, y_pred)
    mse=model.evaluate([X_test[0],X_test[1],X_test[2],X_test[3]],Y_test, verbose=0)
    #print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
   
    
    print("KKKKKKKKKKKKKKKKKKKKKKKKKK")
    print(y_pred)
    DF1 = pd.DataFrame(y_pred)
    DF1.to_csv("pred_sund.csv")
    print("KKKKKKKKKKKKKKKKKKKKKKKKKK")
    print(Y_test)
    DF2 = pd.DataFrame(Y_test)
    DF2.to_csv("real_sund.csv")
    
    out = {
        
        'loss': mse[0],
        'status': STATUS_OK,
        'model_params': params,
    }
    #print(overall_pcc)
    print("KKKKKKKKKKKKKKKKKKKKKKKKKK")
    temp_name = tempfile.gettempdir()+'/'+next(tempfile._get_candidate_names()) + '.h5'
    model.save(temp_name)
    with open(temp_name, 'rb') as infile:
        model_bytes = infile.read()
    out['model_serial'] = model_bytes
    print("i will be here %s"%temp_name)
    return out

if __name__ == "__main__":
    random.seed(0)
    
    
    
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          functions = [compute_accuracy],
                                          algo=tpe.suggest,
                                          max_evals=4,
                                          trials=Trials(),
                                         keep_temp=True)  # this last bit is important
    print("am here")
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    

    
