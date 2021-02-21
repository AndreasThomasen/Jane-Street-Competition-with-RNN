import torch.nn as nn

class MarketPredictor(nn.Module):
    def __init__(self,fc_input,h_dims,dropout_rate,e_size):
        super(MarketPredictor, self).__init__()
        
        self.deep = nn.Sequential(
            nn.Linear(fc_input,h_dims[0]),
            nn.BatchNorm1d(h_dims[0]),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_dims[0],h_dims[1]),
            nn.BatchNorm1d(h_dims[1]),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_dims[1],h_dims[2]),
            nn.BatchNorm1d(h_dims[2]),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_dims[2],h_dims[3]),
            nn.BatchNorm1d(h_dims[3]),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_dims[3],e_size),
            nn.BatchNorm1d(e_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
            )
        self.reduce = nn.utils.weight_norm(nn.Linear(e_size,5))
        
    def forward(self,xf):
        f_out = self.deep(xf)
        ef_out = self.reduce(f_out)
        
        return ef_out
    
class MarketPredictorResnet(nn.Module):
    def __init__(self,fc_input,h_len,dropout_rate,depth):
        super(MarketPredictorResnet,self).__init__()
        
        self.depth = depth
        
        self.initialblock = nn.Sequential(
            nn.Linear(fc_input,h_len),
            nn.BatchNorm1d(h_len),
            nn.Dropout(dropout_rate))
        
        
        self.resblocks = nn.ModuleList([nn.Sequential(
            nn.BatchNorm1d(h_len),
            nn.ReLU(),
            nn.Linear(h_len,h_len),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(h_len),
            nn.ReLU(),
            nn.Linear(h_len,h_len),
            nn.Dropout(dropout_rate)) for i in range(depth)])
        
        self.reduce = nn.utils.weight_norm(nn.Linear(h_len,5))
        
    def forward(self,x):
        x = self.initialblock(x)
        for i in range(self.depth):
            
            x = x + self.resblocks[i](x)
        
        x = self.reduce(x)
        return x