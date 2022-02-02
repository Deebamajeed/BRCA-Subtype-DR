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
from layers.graph import GraphLayer,GraphConv

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
import tempfile

DPATH = '/content/DeepCDR-master/data'
Drug_info_file = '%s/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'%DPATH
Cancer_response_exp_file = '%s/CCLE/DR_100-screened_drugs_imputed-new.xlsx'%DPATH
TCGA_label_set = ["ALL","BLCA","BRCA","CESC","DLBC","LIHC","LUAD",
                  "ESCA","GBM","HNSC","KIRC","LAML","LCML","LGG",
                  "LUSC","MESO","MM","NB","OV","PAAD","SCLC","SKCM",
                  "STAD","THCA",'COAD/READ']
use_mut=True
use_gexp=True
use_methy=True
regr=True
####################################Constants Settings###########################
def MetadataGenerate(CNV_file,Genomic_mutation_file,Gene_expression_file,Methylation_file,filtered):
    #drug_id --> pubchem_id
    
    #load demap cell lines genomic mutation features
    cnv_feature = pd.read_excel(CNV_file,header=0,index_col=[0])
    #load demap cell lines genomic mutation features
    mutation_feature = pd.read_excel(Genomic_mutation_file,header=0,index_col=[0])
    #cell_line_id_set = list(mutation_feature.index)
 
    #load gene expression faetures
    gexpr_feature = pd.read_excel(Gene_expression_file,header=0,index_col=[0])
    
    #only keep overlapped cell lines
    mutation_feature = mutation_feature.loc[list(gexpr_feature.index)]
    
    #load methylation 
    methylation_feature = pd.read_excel(Methylation_file,header=0,index_col=[0])
    assert methylation_feature.shape[0]==gexpr_feature.shape[0]==mutation_feature.shape[0]        
    experiment_data = pd.read_excel(Cancer_response_exp_file,header=0,index_col=[0])
    return cnv_feature,mutation_feature, gexpr_feature,methylation_feature,experiment_data