from einops import rearrange
import torch
import torch.nn as nn
import numpy as py

try:
    from torch_discounted_cumsum import  discounted_cumsum_left, discounted_cumsum3_left
    from torch_discounted_cumsum.discounted_cumsum import qkvg_retention,weighted_cumsum_batch
except:
    print('discounted_cumsum not found, pass')
    
class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

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
        return self.forward2(tensor, gamma)

    def forward1(self, tensor, gamma):
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

    def forward2(self, tensor, gamma):
        _shape = tensor.shape
        assert _shape[self.dim_head] == gamma.shape[-1]
        ## then permute the target dim into
        # (B, D, H, S) or (B, D1, D2, H, S)
        if self.dim_head == -2 and self.dim_leng == -1:
            tensor = tensor.view(-1, _shape[-1])  # (B*D*H, S)
        elif self.dim_head == 1 and self.dim_leng == 2:
            if len(_shape) == 4:
                tensor = tensor.permute(0,3,1,2).flatten(0,1)
                #tensor = rearrange(tensor, 'B H S D -> (B D) H S')
            elif len(_shape) == 5:
                tensor = rearrange(tensor, 'B H S D1 D2 -> (B D1 D2) H S')
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        # (H,) -> (B*D*H,) ## same as gamma.unsqueeze(0).unsqueeze(0).repeat(B,D,1).view(-1)
        gamma = gamma#.repeat(len(tensor)//len(gamma))
        tensor = discounted_cumsum3_left(tensor, gamma)
        if len(_shape) == 4:
            B, H, S, D = _shape
            tensor = tensor.reshape(B,D,H,S).permute(0,2,3,1)
            #tensor = rearrange(tensor, '(B D) H S -> B H S D', B=B)
        elif len(_shape) == 5:
            B, H, S, D1, D2 = _shape
            tensor = rearrange(
                tensor, '(B D1 D2) H S -> B H S D1 D2',  B=B, D1=D1)
        else:
            raise
            tensor = tensor.view(*_shape)
        return tensor

class ParallelRetention_fast(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma_cusum_1 = Discounted_Cumsum(1,2)
        self.gamma_cusum_2 = Discounted_Cumsum(1,2)
    
    def forward(self, q, k, v, omask=None, gamma=None, L=None):
        if gamma is None:gamma = omask[0,:,1,0].float()
        if L is None:L     = omask.sum(dim=-1,keepdim=True)
        qL    = q/L
        Tbf   = self.gamma_cusum_1(k,gamma)
        P     = torch.einsum('BHia, BHia->BHi',qL, Tbf)
        P     = P[...,None].detach().abs().clamp(min=1)
        D     = torch.einsum('BHia,BHic->BHiac',k, v)
        D     = self.gamma_cusum_2(D,gamma)
        O     = torch.einsum('BHia,BHiac->BHic',qL,D)/P
        return O

class ParallelRetention_fast2(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma_cusum_1 = Discounted_Cumsum(1,2)
        self.gamma_cusum_2 = Discounted_Cumsum(1,2)
    
    def forward(self, q, k, v, omask=None, gamma=None, L=None):
        if gamma is None:gamma = omask[0,:,1,0].float()
        if L is None:L     = omask.sum(dim=-1,keepdim=True)
        qL    = q/L
        Tbf   = self.gamma_cusum_1(k,gamma)
        P     = torch.einsum('BHia, BHia->BHi',qL, Tbf)
        P     = P[...,None].detach().abs().clamp(min=1)
        #qL    = qL
        O     = qkvg_retention(qL,k,v,omask[0])/P
        return O

class ParallelRetention_fast3(nn.Module):
    def __init__(self):
        super().__init__()
        
    
    def forward(self, q, k, v, omask=None, gamma=None, L=None):
        if gamma is None:gamma = omask[0,:,1,0].float()
        if L is None:L     = omask.sum(dim=-1,keepdim=True)
        B,H,S,D = k.shape
        qL    = q/L
        Tbf   = weighted_cumsum_batch(k.permute(0,3,1,2).flatten(0,1), omask[0]).reshape(B,D,H,S).permute(0,2,3,1)
        P     = torch.einsum('BHia, BHia->BHi',qL, Tbf)
        P     = P[...,None].detach().abs().clamp(min=1)
        O     = qkvg_retention(qL,k,v,omask[0])/P
        return O
    
def rmsenorm(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class ParallelRetention_reduce(nn.Module):
    
    def forward(self, q, k, v, omask,  mask_normer, *args, normlized=True,**kargs):
        Coef_Q,Coef_K,scale = omask
        B,H,S1,D1 = q.shape
        B,H,S2,D1 = k.shape
        B,H,S2,D2 = v.shape
        q_bar = q*Coef_Q.view(1,H,S1,1)/mask_normer.view(1,H,S1,1)
        k_bar = k*Coef_K.view(1,H,S2,1)
        if normlized:
            T = torch.cumsum(k_bar,dim=-2)
            P = torch.einsum('BHia,BHia->BHi', T,q_bar)
            P = P[...,None].detach().abs().clamp(min=1)
            q_bar = q_bar/P
        D = torch.einsum('BHia,BHic->BHiac',k_bar, v)
        D = torch.cumsum(D,dim=-3)
        O = torch.einsum('BHia,BHiac->BHic',q_bar,D)
        O = rmsenorm(O)
        return O

class ParallelRetention_origin(nn.Module):
    
    
    def forward(self, q, k, v, omask, mask_normer, *args, normlized=True,**kargs):
        B,H,S1,D1 = q.shape
        B,H,S2,D1 = k.shape
        B,H,S2,D2 = v.shape
        H,S1 = mask_normer.shape
        mask = omask / mask_normer.view(1,H,S1,1 )
        mask = torch.nan_to_num(mask, nan=0.0)
        decay_mask = mask
        retention = q @ k.transpose(-1, -2)  # --> (B,H,S,S)
        retention = retention * decay_mask   # --> (B,H,S,S)
        #retention = retention/2
        if normlized:retention = retention / retention.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1) # --> (B,H,S,S)
        output = retention @ v  # [b, h, t, v_dim / h] ## # --> (B,H,S,D)
        output = rmsenorm(output)
        return output
import opt_einsum as oe    
from opt_einsum import contract
class ParallelRetention_plain(nn.Module):
    def __init__(self):
        super().__init__()
        self.best_contract_path = {}

    def forward(self, q, k, v, omask,*args, normlized=False,**kargs):
        assert not normlized
        mask = omask / omask.sum(dim=-1, keepdim=True).sqrt()
        mask = torch.nan_to_num(mask, nan=0.0)[0]
        B,H,S1,D1 = q.shape
        B,H,S2,D1 = k.shape
        B,H,S2,D2 = v.shape
        name = f"{B}.{H}.{S1}.{S2}.{D1}.{D2}"

        if name not in self.best_contract_path:
            self.best_contract_path[name] = oe.contract_path('BHia,Hij,BHja,BHjc->BHic',q, mask, k, v, optimize='random-greedy-128')[0]
        output = contract('BHia,Hij,BHja,BHjc->BHic',q, mask, k, v,optimize=self.best_contract_path[name])
        #output = output/np.sqrt(D1*S2)
        output = rmsenorm(output)
        return output

def get_omask(slen, num_heads):
    decay = torch.log(1 - 2**(-5 - torch.arange(num_heads, dtype=torch.float)))
    index = torch.arange(slen).float()
    mask  = torch.tril(torch.ones(slen, slen))
    mask  = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), torch.inf)
    mask  = torch.exp(mask * decay[:, None, None])
    mask  = torch.nan_to_num(mask)
    omask = mask.unsqueeze(0)  # [1, h, t, t]
    L     = mask.sum(dim=-1).sqrt()
    # mask = omask / omask.sum(dim=-1, keepdim=True).sqrt()
    # mask = torch.nan_to_num(mask, nan=0.0)
    return omask,L
def get_omask_reduce(slen, num_heads):
    decay = torch.log(1 - 2**(-5 - torch.arange(num_heads, dtype=torch.float)))
    index = torch.arange(slen).float()
    mask  = torch.tril(torch.ones(slen, slen))
    mask  = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), torch.inf)
    mask  = torch.exp(mask * decay[:, None, None])
    mask  = torch.nan_to_num(mask)
    omask = mask.unsqueeze(0)  # [1, h, t, t]
    L     = mask.sum(dim=-1).sqrt()
    # mask = omask / omask.sum(dim=-1, keepdim=True).sqrt()
    # mask = torch.nan_to_num(mask, nan=0.0)
    index = torch.arange(slen).float()
    index = index - slen//2
    Coef_Q= torch.exp( index[None,:] * decay[ :,None])
    Coef_K= torch.exp(-index[None,:] * decay[ :,None])
    scale = torch.exp(       slen//2 * decay[ :,None])
    return (Coef_Q,Coef_K,scale),L
if __name__ == "__main__":
    ## benchmark!!
    from tqdm.auto import tqdm
    import time
    import numpy as np
    import pandas as pd
    layers = [
        #['layer_fast',  ParallelRetention_fast()],
        # ['layer_fast2', ParallelRetention_fast2()],
        # ['layer_fast3', ParallelRetention_fast3()],
        ['layer_reduce', ParallelRetention_reduce()],
        ['layer_origin', ParallelRetention_origin()],
        #['layer_plain', ParallelRetention_plain()]
    ]
    
    records = []
    configs = []
    
    for B in [1]:
        for S in [1000]:
            for D1 in [8, 16, 32]:
                for D2 in [8, 16, 32]:
                    for H in [16]:
                        configs.append([B,H,S,D1,D2])
    records = []
    dataframe = pd.DataFrame()
    dtype = torch.bfloat16
    #dtype = torch.float32
    for i,(B,H,S,D1,D2) in tqdm(enumerate(configs), desc="Outer Loop", position=0):
        omask_origin,mask_normer_origin =  get_omask(S,H)
        omask_reduce,mask_normer_reduce =  get_omask_reduce(S,H)
        q     =  (torch.randn(B,H,S,D1)/np.sqrt(S*D1)).cuda().to(dtype)#.bfloat16() #and bfloat16 may get wrong result if the q@k@v too large. Normlize it thus q@k@v be a normal distribution
        k     =  (torch.randn(B,H,S,D1)/np.sqrt(S*D1)).cuda().to(dtype)#.bfloat16() #and bfloat16 may get wrong result if the q@k@v too large. Normlize it thus q@k@v be a normal distribution
        v     =  (torch.randn(B,H,S,D2)/np.sqrt(S*D2)).cuda().to(dtype)#.bfloat16() #and bfloat16 may get wrong result if the q@k@v too large. Normlize it thus q@k@v be a normal distribution
        omask_dict =      {'layer_origin':omask_origin.cuda().to(dtype), 
                           'layer_reduce':tuple([t.cuda().to(dtype) for t in omask_reduce])
                           }
        mask_normer_dict={'layer_origin':mask_normer_origin.cuda().to(dtype), 
                          'layer_reduce':mask_normer_reduce.cuda().to(dtype)}
        gamma = None
        O_results= {name:layer(q,k,v,omask_dict[name], 
                                     mask_normer_dict[name], 
                                     gamma, 
                                     mask_normer_dict[name],
                                     normlized=False) for name, layer in layers}
        
        O_should = O_results['layer_origin']
        error_record = {}
        for name in O_results.keys():
            if name in ['layer_origin']:continue
            O = O_results[name]
            e = torch.dist(O,O_should).item()
            error_record[name] = e
        
        time_cost_record = {}
        for name, model in layers:
            costs = []
            for i in tqdm(range(100), desc="Inner Loop", position=1, leave=False):
                now = time.time()
                O = model(q,k,v,omask_dict[name], 
                                     mask_normer_dict[name], 
                                     gamma, 
                                     mask_normer_dict[name])
                #O.mean().backward()
                cost = time.time()-now
                if i>1:costs.append(cost)
            cost = np.mean(cost)
            
            time_cost_record[name] = cost
        line_record = {}
        for name in time_cost_record.keys():
            line_record[name+'_time'] = time_cost_record[name]
        for name in error_record.keys():
            line_record[name+'_error'] = error_record[name]
        now_line  = pd.DataFrame([[B,H,S,D1,D2]],columns=['B','H','S','D1','D2'],index=[i])
        info_line = pd.DataFrame(line_record,index=[i])
        now_line = pd.concat([now_line,info_line],axis=1)
        dataframe= pd.concat([dataframe,now_line])
    print(dataframe)
    # dataframe = pd.DataFrame(records, columns=['B','H','S','D1','D2','e1','e2',#'e3','e4',
    #                                            'fast',#'fast2','fast3',
    #                                            'reduce','origin'])
    # dataframe['speed_up_fast'] = np.round(dataframe['origin']/dataframe['fast'],3)
    # dataframe['speed_up_reduce'] = np.round(dataframe['origin']/dataframe['reduce'],3)
    # print(dataframe)
    # dataframe.to_csv('benchmark_more.csv',index=False)

    ####### percision case
    # for B in [1]:
    #     for S in [10, 50, 100,150, 200, 500, 1000, 2000]:
    #         for D1 in [ 128]:
    #             #for D2 in [64]:
    #             for H in [16]:
    #                 configs.append([B,H,S,D1,D1])
    # records = []
    # for B,H,S,D1,D2 in tqdm(configs):
        
    #     q     =  torch.randn(B,H,S,D1).cuda()#  float16 and bfloat16 always get wrong result ###.half()
    #     k     =  torch.randn(B,H,S,D1).cuda()#  float16 and bfloat16 always get wrong result ###.half()
    #     v     =  torch.randn(B,H,S,D2).cuda()#  float16 and bfloat16 always get wrong result ###.half()
    #     omask =         get_omask(S,H).cuda()#  float16 and bfloat16 always get wrong result ###.half()
    #     omask = omask[omask!=0]
    #     records.append([S,H,
    #                     omask.min().item(), omask.max().item(), omask.half().min().item(), 
    #                     omask.half().max().item(), omask.bfloat16().min().item(), omask.bfloat16().max().item()])

    # dataframe = pd.DataFrame(records, columns=['S','H','fp32_min','fp32_max','fp16_min','fp16_max','bf16_min','bf16_max'])
    # print(dataframe)
    # dataframe.to_csv('gamma_vs_percision.csv',index=False)