import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class cModel(nn.Module):
    '''
    def __init__(self, test_input, n_classes):
        super(cModel, self).__init__()
        self.feed = nn.Sequential(
            nn.Conv2d(1, 4, (5, 1), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, (5, 1), stride=1, padding=1),
            nn.Conv2d(8, 16, (5, 1), stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, (5, 1), stride=2, padding=0),
            nn.ReLU(),
        )
        #import ipdb; ipdb.set_trace()
        #x = self.feed(test_input)
        #dims = [int(z) for z in x.shape]
        self.mlp = nn.Sequential(
            nn.Flatten(start_dim=1),
            #nn.Linear(in_features=np.prod(dims[1:]), out_features=512),
            #nn.Linear(in_features=7936, out_features=512),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=n_classes),
            nn.ReLU()
        )
    '''

    def __init__(self, test_input, n_classes):
        super(cModel, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(1, 4, (5, 1), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, (5, 1), stride=1, padding=1),
            nn.Conv2d(8, 16, (5, 1), stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, (5, 1), stride=2, padding=0),
            nn.ReLU()
        )

        self.block_2 = nn.Sequential(
            #nn.Linear(in_features=7936, out_features=512),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=n_classes),
            nn.ReLU()
        )

    def forward(self, x):
        #out = self.block_1(x)
        #out = out.reshape(out.size(0), -1)
        #out = self.block_2(out)

        out = x.reshape(x.size(0), -1)
        out = self.block_2(out)
        return out

def KL(alpha, K, beta):

    S_alpha = torch.sum(alpha, axis=1, keepdim=True)
    S_beta = torch.sum(beta, axis=1, keepdim=True)

    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), axis=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), axis=1, keepdim=True) - torch.lgamma(S_beta)
    
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    
    kl = torch.sum((alpha - beta)*(dg1-dg0), axis=1, keepdim=True) + lnB + lnB_uni

    return kl

def simplex_loss(p, alpha, global_step, annealing_step, K, beta): 

    S = torch.sum(alpha, axis=1, keepdim=True) 
    E = alpha - 1
    m = alpha / S
    
    A = torch.sum((p-m)**2, axis=1, keepdim=True) 
    B = torch.sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdim=True) 
    
    annealing_coef = torch.min(torch.tensor(1.0), global_step/annealing_step)

    # remove non-misleading evidence and match to uniform distribution    
    alp = E*(1-p) + 1 
    C = annealing_coef * KL(alp, K, beta)

    return torch.mean(A + B + C)
