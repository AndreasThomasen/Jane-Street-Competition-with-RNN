import random
import os
import pickle
import numpy as np
import torch

def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dic,f)
    
def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    
    return message_dict

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def fillna_npwhere_njit(array, values):
    if np.isnan(array).any():
        array = np.where(np.isnan(array), values, array)
    return array

def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0
    
    for data in dataloader:
        optimizer.zero_grad()
        features = data['features'].to(device)
        targets = data['targets'].to(device)
        outputs = model(features)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        final_loss += loss.item()
    
    final_loss /= len(dataloader)
    
    return final_loss
    
def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        features = data['features'].to(device)

        with torch.no_grad():
            outputs = model(features)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds
    
def eval_fn(model, score_fn, dataloader, device):
    model.eval()
    ev = 0
    
    for data in dataloader:
        features = data['features'].to(device)
        targets = data['targets'].to(device)
        
        
        
        with torch.no_grad():
            outputs = model(features)
            score = score_fn(targets.cpu(),outputs.sigmoid().cpu())
        
        ev += score
        
    ev /= len(dataloader)
    
    return ev

class ScoreTracker:
    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
        
    def __call__(self, epoch_score, model, model_path):
        
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0
        
    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            torch.save(model.state_dict(),model_path)
            
        self.val_score = epoch_score

class MarketDataset:
    def __init__(self, df, feature_names, target_names):
        self.features = df[feature_names].values
        self.targets = df[target_names].values
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'targets': torch.tensor(self.targets[idx], dtype=torch.float)
            }
        
        