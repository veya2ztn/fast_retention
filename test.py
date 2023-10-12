from einops import rearrange
import torch
import torch.nn as nn
from torch_discounted_cumsum import  discounted_cumsum_left

class Discounted_Cumsum(nn.Module):
    """
    Assume input it (B, H, S, D) or (B, H, S, D1, D2)
                 or (B, D, H, S) or (B, D1, D2, H, S)
    ---> firstly, convert to
        - input (B*D, S)
        - gamma (B*D)
    ---> then, compute discounted cumsum by
        discounted_cumsum_left(input, gamma)
    ---> finally, convert back to original shape
    """
    def __init__(self, dim_head = -2, dim_leng = -1):
        super().__init__()
        self.dim_head  = dim_head
        self.dim_leng  = dim_leng
        
    def forward(self, tensor, gamma):
        _shape = tensor.shape
        assert _shape[self.dim_head] == gamma.shape[-1]
        ## then permute the target dim into 
        if self.dim_head == -2 and self.dim_leng == -1: #(B, D, H, S) or (B, D1, D2, H, S)
            tensor = tensor.view(-1, _shape[-1]) # (B*D*H, S)
        elif self.dim_head == 1 and self.dim_leng == 2:
            if   len(_shape) == 4:tensor = rearrange(tensor, 'B H S D -> (B D H) S')
            elif len(_shape) == 5:tensor = rearrange(tensor, 'B H S D1 D2 -> (B D1 D2 H) S')
            else:raise NotImplementedError
        else:
            raise NotImplementedError
        gamma  = gamma.repeat(len(tensor)//len(gamma)) #(H,) -> (B*D*H,) ## same as gamma.unsqueeze(0).unsqueeze(0).repeat(B,D,1).view(-1)
        tensor = discounted_cumsum_left(tensor, gamma)
        if   len(_shape) == 4:
            B,H,S,D = _shape
            tensor = rearrange(tensor, '(B D H) S -> B H S D', B=B,  D=D)
        elif len(_shape) == 5:
            B,H,S,D1,D2 = _shape
            tensor = rearrange(tensor, '(B D1 D2 H) S -> B H S D1 D2',  B=B, D1=D1, D2=D2)
        else:
            tensor = tensor.view(*_shape)
        return tensor

class ParallelRetention_fast(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma_cusum_1 = Discounted_Cumsum(1,2)
        self.gamma_cusum_2 = Discounted_Cumsum(1,2)
    
    def forward(self, q, k, v, omask):
        gamma = omask[0,:,1,0]
        L     = omask.sum(dim=-1).sqrt()[...,None]
        qL    = q/L
        Tbf   = self.gamma_cusum_1(k,gamma)
        P     = torch.einsum('BHia, BHia->BHi',qL, Tbf)
        P     = P[...,None].detach().abs().clamp(min=1)
        D     = torch.einsum('BHia,BHic->BHiac',k, v)
        D     = self.gamma_cusum_2(D,gamma)
        O     = torch.einsum('BHia,BHiac->BHic',qL,D)/P
        return O

class ParallelRetention_reduce(nn.Module):
    
    def forward(self, q, k, v, omask):
        q_bar_coef = omask[...,:,0]/omask.sum(dim=-1).sqrt()
        k_bar_coef = 1/(omask[...,:,0])#<----this will overflow~~~~!!!!
        q_bar = q_bar_coef[...,None]*q
        k_bar = k_bar_coef[...,None]*k
        T = torch.cumsum(k_bar,dim=-2)
        P = torch.einsum('BHia,BHia->BHi', T,q_bar)
        P = P[...,None].detach().abs().clamp(min=1)
        D = torch.einsum('BHia,BHic->BHiac',k_bar, v)
        D = torch.cumsum(D,dim=-3)
        O = torch.einsum('BHia,BHiac->BHic',q_bar,D)/P
        return O

class ParallelRetention_origin(nn.Module):
    
    
    def forward(self, q, k, v, omask):
        mask = omask / omask.sum(dim=-1, keepdim=True).sqrt()
        mask = torch.nan_to_num(mask, nan=0.0)
        decay_mask = mask
        retention = q @ k.transpose(-1, -2)  # --> (B,H,S,S)
        retention = retention * decay_mask   # --> (B,H,S,S)
        retention = retention / retention.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1) # --> (B,H,S,S)
        output = retention @ v  # [b, h, t, v_dim / h] ## # --> (B,H,S,D)
        return output
    

def get_omask(slen, num_heads):
    decay = torch.log(1 - 2**(-5 - torch.arange(num_heads, dtype=torch.float)))
    index = torch.arange(slen).float()
    mask = torch.tril(torch.ones(slen, slen))
    mask = torch.masked_fill(
        index[:, None] - index[None, :], ~mask.bool(), torch.inf)
    mask = torch.exp(mask * decay[:, None, None])

    mask = torch.nan_to_num(mask)
    omask = mask.unsqueeze(0)  # [1, h, t, t]
    # mask = omask / omask.sum(dim=-1, keepdim=True).sqrt()
    # mask = torch.nan_to_num(mask, nan=0.0)
    return omask

if __name__ == "__main__":
    ## benchmark!!
    from tqdm.auto import tqdm
    import time
    import numpy as np
    import pandas as pd
    layer1= ParallelRetention_fast()
    layer2= ParallelRetention_reduce()
    layer3= ParallelRetention_origin()
    records = []
    H = 16
    for B in [2]:
        for S in [30, 300, 3000]:
            for D in [4, 8 , 16, 32]:
                q     =  torch.randn(2,H,S,D).cuda()
                k     =  torch.randn(2,H,S,D).cuda()
                v     =  torch.randn(2,H,S,D).cuda()
                omask = get_omask(S,H).cuda()
                O1 = layer1(q,k,v,omask)
                O2 = layer2(q,k,v,omask)
                O3 = layer3(q,k,v,omask)
                e1 = torch.dist(O1,O3).item()
                e2 = torch.dist(O2,O3).item()
                record = [B,S,D,e1,e2]
                print(record)
                for model in [layer1, layer2, layer3]:
                    costs = []
                    for _ in tqdm(range(100)):
                        now = time.time()
                        O = model(q,k,v,omask)
                        #O.mean().backward()
                        cost = time.time()-now
                        costs.append(cost)
                    record.append(np.mean(cost))
                records.append(record)
    dataframe = pd.DataFrame(records, columns=['B','S','D','e1','e2','fast','reduce','origin'])
    print(dataframe)
    dataframe.to_csv('benchmark.csv',index=False)