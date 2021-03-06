import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import math
import sklearn.preprocessing as sk
import seaborn as sns
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
#import utils
#from myutils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
#from mymetrics import AverageNonzeroTripletsMetric
#from utils import RandomNegativeTripletSelector
from ncafs import NCAFSC
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import random
from random import randint
from sklearn.model_selection import StratifiedKFold,RepeatedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp

import numpy as np
import warnings
import math
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
warnings.filterwarnings("ignore")
#save_results_to = '/home/winston/MOLI/res/all_6index/nofs/'
save_results_to = '/content/results'
#torch.manual_seed(42)
random.seed(42)

max_iter = 1



TCGAEx = pd.read_csv("/content/gdrive/MyDrive/moanna_training_data.tsv",sep='\t')
#TCGAEx_rna = pd.read_csv("/content/training_mrna.csv", 
#                    index_col=0, header=None)

#TCGAEx_meth = pd.read_csv("brca_meth_subtype.csv",                index_col=0, header=None)
TCGAEx=TCGAEx.values
#TCGAEx_cnv1=TCGAEx_cnv[:,0:277]
#TCGAEx_cnv2=TCGAEx_cnv[:,277:606]
#TCGAEx1=np.hstack((TCGAEx_cnv1,TCGAEx_cnv2))
TCGAEx_cnv=TCGAEx[:,1:15593]
TCGAEx_rna=TCGAEx[:,15593:46778]
#TCGAEx_rna=TCGAEx_rna.values
#TCGAEx_rna1=TCGAEx_rna[:,0:277]
#TCGAEx_rna2=TCGAEx_rna[:,277:606]
#TCGAEx2=np.hstack((TCGAEx_rna1,TCGAEx_rna2))

#TCGAEx_meth=TCGAEx_meth.values
#TCGAEx_meth1=TCGAEx_meth[:,0:277]
#TCGAEx_meth2=TCGAEx_meth[:,277:606]
#TCGAEx3=np.hstack((TCGAEx_meth1,TCGAEx_meth2))
#print(TCGAEx)
label_df=pd.read_csv("/content/gdrive/MyDrive/moanna_training_label.csv", header=None)
#label=TCGAEx1[0,:]
data1=TCGAEx_cnv[:,:]
data2=TCGAEx_rna[:,:]
#data3=TCGAEx3[1:,:]
#print(label)

#label[np.where(label=='lminalA')]=0
#label[np.where(label=='lminalB')]=1
#label[np.where(label=='TNBC')]=2
#label[np.where(label=='ERBB2')]=3
#label[np.where(label=='normal')]=4

#data1=data1.T
#data2=data2.T
#data3=data3.T
print(data1.shape)
print(data2.shape)
#print(data3.shape)
print(label_df.shape)
Y=[]
Y=label_df[0]

min_max_scaler = preprocessing.MinMaxScaler()

x1 = min_max_scaler.fit_transform(data1)
x2 = min_max_scaler.fit_transform(data2)
#TCGAE3 = min_max_scaler.fit_transform(data3)
TCGAE1 =NCFASC(1000).fit_transform(x1, Y)
TCGAE2 =NCAFSC(1000).fit_transform(x2, Y)
#TCGAE3 = SelectKBest(chi2, k=5000).fit_transform(x3, Y)
#print(Y)
'''
ls_mb_size = [13, 36, 64]
ls_h_dim = [1024, 256, 128, 512, 64, 16]
#ls_h_dim = [32, 16, 8, 4]
ls_marg = [0.5, 1, 1.5, 2, 2.5, 3]
ls_lr = [0.5, 0.1, 0.05, 0.01, 0.001, 0.005, 0.0005, 0.0001,0.00005, 0.00001]
ls_epoch = [20, 50, 90, 100]
ls_rate = [0.3, 0.4, 0.5]
ls_wd = [0.1, 0.001, 0.0001]
ls_lam = [0.1, 0.5, 0.01, 0.05, 0.001, 0.005]
'''
ls_mb_size = [64]
ls_h_dim = [512]

ls_lr = [0.01]
ls_epoch = [100]
ls_rate = [0.5]
ls_wd = [0.001]


skf = StratifiedKFold(n_splits=5, random_state=None,shuffle=True)

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
for iters in range(max_iter):
    #print('iters:',iters)
    k = 0
    mbs = random.choice(ls_mb_size)
    hdm = random.choice(ls_h_dim)
    #mrg = random.choice(ls_marg)
    lre = random.choice(ls_lr)
    lrCL = random.choice(ls_lr)
    epch = random.choice(ls_epoch)
    rate = random.choice(ls_rate)
    wd = random.choice(ls_wd)   
    #lam = random.choice(ls_lam)

    costtr_all=[]
    costts_all=[]
    auctr_all=[]
    aucts_all=[]
    acctr_all=[]
    accts_all=[]
    sentr_all=[]
    sents_all=[]
    spetr_all=[]
    spets_all=[]
   
    mean_tpr = 0.0              # ?????????????????????ROC???????????????

    mean_fpr = np.linspace(0, 1, 100)

    #cnt = 0
    
    for repeat in range(10):
        for train_index, test_index in skf.split(TCGAE1, Y.astype('int')):
            
            #if(k%5==0):
            k = k + 1
            X_trainE1 = TCGAE1[train_index,:]
            X_trainE2 = TCGAE2[train_index,:]
            #X_trainE3 = TCGAE3[train_index,:]
            X_testE1 =  TCGAE1[test_index,:]
            X_testE2 =  TCGAE2[test_index,:]
            #X_testE3 =  TCGAE3[test_index,:]
            y_trainE = Y[train_index]
            y_testE = Y[test_index]
       
            
            TX_testE1 = torch.FloatTensor(X_testE1)
            TX_testE2 = torch.FloatTensor(X_testE2)
            #TX_testE3 = torch.FloatTensor(X_testE3)
            ty_testE = torch.FloatTensor(y_testE.to_numpy())
            
            #Train
            class_sample_count = np.array([len(np.where(y_trainE==t)[0]) for t in np.unique(y_trainE)])
            #print(class_sample_count)
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in y_trainE])

            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

            mb_size = mbs

            trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_trainE1),torch.FloatTensor(X_trainE2),
                                          torch.FloatTensor(y_trainE.to_numpy()))

            trainLoader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size=mb_size, shuffle=False, num_workers=0, sampler = sampler)

            n_sampE1, IE1_dim = X_trainE1.shape
            n_sampE2, IE2_dim = X_trainE2.shape
            #n_sampE3, IE3_dim = X_trainE3.shape

            h_dim1 = hdm
            h_dim2 = hdm
            #h_dim3 = hdm
            Z_in = h_dim1+h_dim2
            #marg = mrg
            lrE = lre
            epoch = epch

            costtr = []
            auctr = []
            costts = []
            aucts = []
            acctr=[]
            accts=[]
            acc0tr=[]
            acc0ts=[]
            sentr=[]
            sents=[]
            spetr=[]
            spets=[]
            class AEE(nn.Module):
                def __init__(self):
                    super(AEE, self).__init__()
                    self.EnE = torch.nn.Sequential(
                        nn.Linear(IE1_dim, h_dim1),
                        nn.BatchNorm1d(h_dim1),
                        nn.ReLU(),
                        nn.Dropout(rate))
                def forward(self, x):
                    output = self.EnE(x)
                    return output

            class AEM(nn.Module):
                def __init__(self):
                    super(AEM, self).__init__()
                    self.EnM = torch.nn.Sequential(
                        nn.Linear(IE2_dim, h_dim2),
                        nn.BatchNorm1d(h_dim2),
                        nn.ReLU(),
                        nn.Dropout(rate))
                def forward(self, x):
                    output = self.EnM(x)
                    return output    

            class OnlineTriplet(nn.Module):
                def __init__(self, marg, triplet_selector):
                    super(OnlineTriplet, self).__init__()
                    self.marg = marg
                    self.triplet_selector = triplet_selector
                def forward(self, embeddings, target):
                    triplets = self.triplet_selector.get_triplets(embeddings, target)
                    return triplets

            class OnlineTestTriplet(nn.Module):
                def __init__(self, marg, triplet_selector):
                    super(OnlineTestTriplet, self).__init__()
                    self.marg = marg
                    self.triplet_selector = triplet_selector
                def forward(self, embeddings, target):
                    triplets = self.triplet_selector.get_triplets(embeddings, target)
                    return triplets    

            class Classifier(nn.Module):
                def __init__(self):
                    super(Classifier, self).__init__()
                    self.FC = torch.nn.Sequential(
                        nn.Linear(Z_in, 5),
                        nn.Dropout(rate))
                def forward(self, x):
                    return F.softmax(self.FC(x))
                    #return self.FC(x)

            torch.cuda.manual_seed_all(42)

            AutoencoderE1 = AEE()
            AutoencoderE2 = AEM()
           

            solverE1 = optim.Adagrad(AutoencoderE1.parameters(), lr=lrE)
            solverE2 = optim.Adagrad(AutoencoderE2.parameters(), lr=lrE)
            #solverE3 = optim.Adagrad(AutoencoderE3.parameters(), lr=lrE)
            Clas = Classifier()
            SolverClass = optim.Adagrad(Clas.parameters(), lr=lrCL, weight_decay = wd)
            C_loss = torch.nn.CrossEntropyLoss()
            print('epoch_all',epoch)
            for it in range(epoch):
                #print('epoch:',it)
                epoch_cost4 = []
                epoch_cost3 = []
                p_real=[]
                p_pred=[]
                n_real=[]
                n_pred=[]
                p_realt=[]
                p_predt=[]
                n_realt=[]
                n_predt=[]
             
                correct = torch.zeros(1).squeeze()
                total = torch.zeros(1).squeeze()
                correctt = torch.zeros(1).squeeze()
                totalt = torch.zeros(1).squeeze()
                num_minibatches = int(n_sampE1 / mb_size) 

                for i, (dataE1,dataE2, target) in enumerate(trainLoader):
                    
                    flag = 0
                    AutoencoderE1.train()
                    AutoencoderE2.train()
                    #AutoencoderE3.train()

                    Clas.train()

                    #if torch.mean(target)!=0. and torch.mean(target)!=1.: 
                    ZEX1 = AutoencoderE1(dataE1)
                    ZEX2 = AutoencoderE2(dataE2)
                   # ZEX3 = AutoencoderE3(dataE3)
                    ZT = torch.cat((ZEX1, ZEX2), 1)
                    ZT = F.normalize(ZT, p=2, dim=0)
                    Pred = Clas(ZT)

                    #Triplets = TripSel2(ZEX, target)
                    #loss = lam * trip_criterion(ZEX[Triplets[:,0],:],ZEX[Triplets[:,1],:],ZEX[Triplets[:,2],:]) + C_loss(Pred,target.view(-1,1))     
                    loss=C_loss(Pred,target.view(-1,1).long().squeeze())
                    y_true = target.view(-1,1)
                    y_pred = Pred

                    prediction = torch.argmax(y_pred, 1)
                    correct += (prediction == target.long().squeeze()).sum().float()
                    total += len(target)
                   

                            
                    solverE1.zero_grad()
                    solverE2.zero_grad()
                    #solverE3.zero_grad()
                    SolverClass.zero_grad()

                    loss.backward()

                    solverE1.step()
                    solverE2.step()
                    #solverE3.step()
                    SolverClass.step()

                    epoch_cost4.append(loss)
                    
                flag = 1

                if flag == 1:
                    costtr.append(torch.mean(torch.FloatTensor(epoch_cost4)))
                   
                    acc=(correct/total).detach().numpy()
                    acctr.append(acc)
                if(it==epoch-1):
                    print('acctrain:',acc)
                    
                    
                with torch.no_grad():

                    AutoencoderE1.eval()
                    AutoencoderE2.eval()
                    #AutoencoderE3.eval()
                    Clas.eval()

                    #ZET = AutoencoderE(TX_testE)
                    ZET1 = AutoencoderE1(TX_testE1)
                    ZET2 = AutoencoderE2(TX_testE2)
                    #ZET3 = AutoencoderE3(TX_testE3)

                    ZTT = torch.cat((ZET1, ZET2), 1)
                    ZTT = F.normalize(ZTT, p=2, dim=0)
                    PredT = Clas(ZTT)

                    #TripletsT = TripSel2(ZET, ty_testE)
                    #lossT = lam * trip_criterion(ZET[TripletsT[:,0],:], ZET[TripletsT[:,1],:], ZET[TripletsT[:,2],:]) + C_loss(PredT,ty_testE.view(-1,1))
                    lossT=C_loss(PredT,ty_testE.view(-1,1).long().squeeze())
                    y_truet = ty_testE.view(-1,1)
                    y_predt = PredT

                    predictiont = torch.argmax(y_predt, 1)
                    correctt += (predictiont == y_truet.long().squeeze()).sum().float()
                    totalt += len(y_truet)
                   
                    costts.append(lossT)
                    acc=(correctt/totalt).detach().numpy()
                    accts.append(acc)
                
                
                if(it==epoch-1):
                   
                    print('acctest:',acc)
                   
                    

            acctr_all.append(acctr)
            accts_all.append(accts)
            costtr_all.append(costtr)
            costts_all.append(costts)

            if k%50==0:
                
                costtr_all=np.array(costtr_all)
                costts_all=np.array(costts_all)
                costtr5=sum(costtr_all)/k
                costts5=sum(costts_all)/k
                print('cost200:',costts5[-1])
                print('costmin:',min(costts5))
                acctr_all=np.array(acctr_all)
                accts_all=np.array(accts_all)
                acctr5=sum(acctr_all)/k
                accts5=sum(accts_all)/k
                print('acc200:',accts5[-1])
                print('accmax:',max(accts5))
                print('cost:',costts5)
                print('acc:',accts5)
                plt.plot(np.squeeze(costtr5), '-r',np.squeeze(costts5), '-b')
                plt.ylabel('Total cost')
                plt.xlabel('epoch')

                title = 'all_ Cost(5class5000)-skf10   mb_size = {},  h_dim = {} , lrE = {}, epoch = {}, rate = {}, wd = {}, lrCL = {}'.\
                              format( mbs, hdm,  lre, epch, rate, wd, lrCL)

                #title = 'Cost  iter = {}, fold = {}, mb_size = {},  h_dim = {}, marg = {}, lrE = {}, epoch = {}, rate = {}, wd = {}, lrCL = {}, lam = {}'.\
                #              format(iters, k, mbs, hdm, mrg, lre, epch, rate, wd, lrCL, lam)

                plt.suptitle(title)
                plt.savefig(save_results_to + title + '.png', dpi = 150)
                plt.close()
               

                plt.plot(np.squeeze(acctr5), '-r',np.squeeze(accts5), '-b')
                plt.ylabel('Accuracy')
                plt.xlabel('epoch')

                title = 'all_ Accuracy(5class5000)-skf10   mb_size = {},  h_dim = {},  lrE = {}, epoch = {}, rate = {}, wd = {}, lrCL = {}, '.\
                              format( mbs, hdm, lre, epch, rate, wd, lrCL)        

                plt.suptitle(title)
                plt.savefig(save_results_to + title + '.png', dpi = 150)
                plt.close()
               
