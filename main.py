import time
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from modules.utils import MarketDataset, ScoreTracker, train_fn, eval_fn, seed_everything
from modules.model import MarketPredictorResnet


DATA_PATH = './data'
CACHE_PATH = './'

SEED = 42
BATCH_SIZE = 8192
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOP = 6
NFOLDS = 5
TRAIN = False
VALID_RATIO = 0.1

FC_INPUT = 132
H_LEN = 256
DROPOUT_RATE = 0.2
DEPTH = 10



feat_cols = [f'feature_{i}' for i in range(130)]

train = pd.read_csv(f'{DATA_PATH}/train.csv')
train.loc[train.date < 85 ].reset_index(drop=True)

target_cols = ['action','action_1','action_2','action_3','action_4']

train['action'] = (train['resp'] > 0).astype('int')
train['action_1'] = (train['resp_1'] > 0).astype('int')
train['action_2'] = (train['resp_2'] > 0).astype('int')
train['action_3'] = (train['resp_3'] > 0).astype('int')
train['action_4'] = (train['resp_4'] > 0).astype('int')

if TRAIN:
    f_mean = train.mean().values
    np.save(f'{CACHE_PATH}/f_mean_online_copy.npy',f_mean)

    train.fillna(train.mean(), inplace=True)
else:
    f_mean = np.load(f'{CACHE_PATH}/f_mean_online_copy.npy')
    f_mean = pd.Series(f_mean[-130:],feat_cols)
    
    train.fillna(f_mean, inplace=True)

train['cross_41_42_43'] = train['feature_41']+train['feature_42']+train['feature_43']
train['cross_1_2'] = train['feature_1']/(train['feature_2']+1e-5)
ext_feat_cols = feat_cols.copy()
ext_feat_cols.extend(['cross_41_42_43','cross_1_2'])

valid_len = int(len(train)*VALID_RATIO)
train_len = len(train) - valid_len

fullset = MarketDataset(train, ext_feat_cols, target_cols)

if TRAIN:
    start_time = time.time()
    for _fold in range(NFOLDS):
        print(f'Fold{_fold}:')
        seed_everything(seed = SEED+_fold)
        [trainset, validset] = random_split(fullset,[train_len, valid_len], generator=torch.Generator().manual_seed(SEED+_fold))
        train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False)
        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
        model = MarketPredictorResnet(FC_INPUT,H_LEN,DROPOUT_RATE,DEPTH).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.BCEWithLogitsLoss()
        
        weight_path = f"{CACHE_PATH}/online_model{_fold}.pth"
        st = ScoreTracker(patience=EARLY_STOP, mode='max')
        scheduler = None
        
        for epoch in range(EPOCHS):
            train_loss = train_fn(model, optimizer, scheduler, loss_fn, train_loader, device)
            #valid_pred = inference_fn(model, roc_auc_score, valid_loader, device)
            valid_auc = eval_fn(model, roc_auc_score, valid_loader, device)
            valid_logloss = eval_fn(model, log_loss, valid_loader, device)
            
            print(f"FOLD{_fold} EPOCH:{epoch:3} train_loss={train_loss:.5f} "
                      f"valid_logloss={valid_logloss:.5f} valid_auc={valid_auc:.5f} "
                      f"time: {(time.time() - start_time) // 60:d}min {np.mod(time.time() - start_time, 60):d} sec")
            st(valid_auc, model, model_path=weight_path)
            if st.early_stop:
                print("Early stopping")
                break
else:
    seed_everything(SEED+NFOLDS)
    [trainset, validset] = random_split(fullset,[train_len, valid_len], generator=torch.Generator().manual_seed(SEED+NFOLDS))
    valid_loader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False)
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")
    
    class FoldedModel(nn.Module):
        def __init__(self, models):
            super(FoldedModel,self).__init__()
            self.models = models
            self.nfolds = len(models)
        
        def forward(self, x):
            out = self.models[0](x)
            for model in self.models[1:]:
                out += model(x)
            
            out /= self.nfolds
            
            return out
    
    models = nn.ModuleList([MarketPredictorResnet(FC_INPUT, H_LEN, DROPOUT_RATE, DEPTH) for _fold in range(NFOLDS)])
    for _fold in range(NFOLDS):
        models[_fold].to(device)
        models[_fold].load_state_dict(torch.load(f"{CACHE_PATH}/online_model{_fold}.pth", device))
        
        
    models = FoldedModel(models).to(device)
    models.eval()
    
    valid_auc = eval_fn(models, roc_auc_score, valid_loader, device)
    valid_logloss = eval_fn(models, log_loss, valid_loader, device)
    
    print("MODEL EVALUATION "
          f"valid_logloss={valid_logloss:.5f} valid_auc={valid_auc:.5f} ")