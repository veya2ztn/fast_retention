# Faster_Retention for large sequence
The implementation is on par with the official implementation at [`torchscale`](https://github.com/microsoft/torchscale) repo for paper (https://arxiv.org/pdf/2307.08621.pdf) and [huggingface transformer compatible implementation of Retention Networks](https://github.com/syncdoth/RetNet)

In this repo, you can find:
1. nX speed up implementation of retention operation of Retention Networks. 
    **Notice**: This implement only suitable for **large** sequence length and small head dimension. Assume the dimension of query $q$ is $(B,H,S,D)$, you can enjoy considerable speed up for $S>>D^2$. Run `python test.py` for benchmark.
    ```
    python test.py #(Platform: 3090) (e1 = fast-origin) (e2 = reduce - origin)
    
        B     S   D        e1        e2      fast    reduce    origin  speed_up
    0   1    30   8  0.000032  0.000033  0.000435  0.000212  0.000118     0.272
    1   1    30  16  0.000068  0.000073  0.000530  0.000247  0.000141     0.266
    2   1    30  32  0.000178  0.000189  0.000472  0.000269  0.000123     0.261
    3   1    30  64  0.000570  0.000490  0.000694  0.000221  0.000122     0.175
    4   1   300   8  0.000147  0.000203  0.000469  0.000223  0.000118     0.251
    5   1   300  16  0.000300  0.000440  0.000488  0.000210  0.000118     0.242
    6   1   300  32  0.000681  0.001043  0.001454  0.000327  0.000240     0.165
    7   1   300  64  0.001849  0.002668  0.005716  0.000982  0.000259     0.045
    8   1  3000   8  0.000969       NaN  0.001877  0.002474  0.016054     8.552
    9   1  3000  16  0.001753       NaN  0.005466  0.002907  0.016108     2.947
    10  1  3000  32  0.003381       NaN  0.020679  0.004071  0.016576     0.802
    11  1  3000  64  0.007355       NaN  0.081177  0.010387  0.017319     0.213
    12  1  5000   8  0.001459       NaN  0.004210  0.005091  0.044165    10.490
    13  1  5000  16  0.002664       NaN  0.012461  0.005538  0.044373     3.561
    14  1  5000  32  0.005147       NaN  0.044402  0.007455  0.045701     1.029
    15  1  5000  64  0.010328       NaN  0.178734  0.018047  0.047592     0.266
    ```
2. Any chunksize Recurrent:
   - If chunksize==wholelength, it become parallel mode.
   - If chunksize==1, it become recurrent mode.
     
   We reformulation the retention, change the operation order and achieve an identity implement that can correct preduce the kv cache and the gk cache of retention.
   ```
    import torch
    import torch.nn as nn
    import numpy as np
    from self_retention import SelfRetentionV2,RetNetRelPosV2, RMSNorm
    from configuration_retnet import RetNetConfig
    S = 30
    B = 2
    H = 8
    qk_dim = 32
    v_dim  = 64
    q = torch.randn(B,H,S,qk_dim).cuda()
    k = torch.randn(B,H,S,qk_dim).cuda()
    v = torch.randn(B,H,S, v_dim).cuda()
    
    config = RetNetConfig(decoder_layers=1,
                          decoder_embed_dim=256,
                          decoder_value_embed_dim=256,
                          decoder_retention_heads=8,
                          decoder_ffn_embed_dim=128)
    retnet_rel_pos = RetNetRelPosV2(config).cuda()
    model          = SelfRetentionV2(config)
    group_norm     = RMSNorm(H,0,False)
    
    model.group_norm = nn.Identity() ## remove the group norm which we add by ourselves
    use_gk = True
    mode   = 'qk_first'
    print("     ================= random chunksize recurrent test ====================")
    partition = np.sort(np.random.choice(np.arange(2,S-2),(5,),replace=False)).tolist() + [S]
    print(f"     partition: {partition}")
    past_kv = None
    full_rnn_state = []
    last = 0
    for i in partition:
        qm = q[:,:,last:i]
        km = k[:,:,last:i]
        vm = v[:,:,last:i]
        (cos, sin), (chunk_gamma, unnormlized_decay_mask, mask_normlizer) = retnet_rel_pos(
            i, recurrent_chunk_size=qm.shape[-2], forward_impl='chunkwise_recurrent')
        one_step_output, _, past_kv = model(qm, km, vm,
                                            (chunk_gamma, unnormlized_decay_mask,mask_normlizer),
                                            past_key_value= past_kv,
                                            normlize_for_stable=use_gk, mode=mode)
        full_rnn_state.append(one_step_output)
        last = i
    full_rnn_state = torch.cat(full_rnn_state, dim=1)

   ```

---------------------------
We use [`torch-discounted-cumsum`](https://github.com/toshas/torch-discounted-cumsum) to accelerate computation.

```
#pip install torch-discounted-cumsum --no-build-isolation
pip install git+https://github.com/veya2ztn/torch-discounted-cumsum.git --no-build-isolation
```

## Motivation
The "Parallel" formation of retention is simple 
```
def forward(self, q, k, v, decay_mask):
    """
        q -> (B, H, S, D)
        k -> (B, H, S, D)
        v -> (B, H, S, D)
        decay_mask-> (1, H, S, S)
    """
    retention = q @ k.transpose(-1, -2)  # --> (B,H,S,S)
    retention = retention * decay_mask   # --> (B,H,S,S)
    retention = retention / retention.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1) # --> (B,H,S,S)
    output = retention @ v  # [b, h, t, v_dim / h] ## # --> (B,H,S,D)
    return output
```

However, it is $\mathcal{O}(S^2)$ implement as the appear of intermeidate shape $(S,S)$. It is wellknow retention is a 'Linear-like' Transfomer that it is possible we contract the $S$ dimension first and reduce the complexity into $\mathcal{O}(D^2)$.

Lets drive it step by step.

Firstly, 
$$R_{i}^j = q_i^{\alpha}k_{\alpha}^j$$


with decay mask 
$$R_{i}^j = C_i^j q_i^{\alpha}k_{\alpha}^j$$

where
$$C_i^j = \lambda^{i-j} \delta_{i>j}$$

Thus
$$R_i^j = \lambda^{i} q_i^{\alpha}k_{\alpha}^j \lambda^{-j} = Q_i^{\alpha} K_{\alpha}^j \delta_{i>j}$$

One thing that paper don't say is it will do normlaization.
$$C_i^j = \frac{\lambda^{i-j}}{\sqrt{\sum_i^t (\lambda^{i-t})^2}}=\frac{\lambda^{i-j}}{L_i}=\frac{\lambda^i}{L_i}\lambda^{-j} \delta_{i>j}$$
Notice all the coef is fixed, thus we can precompute at early begining.

Now, we have reduced $q$ and $k$
$$Q_i^{\alpha} =  \frac{\lambda^i}{L_i}  q_i^{\alpha}$$
$$K_{\alpha}^j =  \lambda^{-j} k_{\alpha}^j $$
Second thing that paper don't say here: another normlaization.
$$R_{i}^j \rightarrow \frac{R_i^j }{|\sum_t^j R_i^t|} = \frac{R_i^j }{|P_i|}$$
The $P_i$ can dirctly compute
$$P_i=\sum_t R_i^t = \sum_t (Q_i^{\alpha}K_{\alpha}^t \delta_{i>t})=\sum_t^i (Q_i^{\alpha}K_{\alpha}^t)$$
$$=Q_i^{\alpha}\sum_t^i (K_{\alpha}^t)=Q_i^{\alpha}T_{\alpha}^i$$
$$=\frac{q_i^{\alpha}}{L_i}\sum_t^i (\lambda^{i-j}k_{\alpha}^t)=\frac{q_i^{\alpha}}{L_i}T_{\alpha}^i$$


second line is the formula for reduced $q$ and $k$

third line is the formula for `discounted-cumsum`

Next
$$O_i^\gamma=R_i^j v_{j}^\gamma = \frac{R_i^j v_{j}^\gamma}{|P_i|}= \frac{o_i^\gamma}{|P_i|}$$

Still can compute
$$o_i^{\gamma} = R_i^j v_{j}^\gamma = Q_i^{\alpha}\delta_{i>j}K_{\alpha}^j  v_{j}^\gamma$$

$$= Q_i^{\alpha}(\delta_{i>j} K_{\alpha}^j  v_{j}^\gamma) = Q_i^{\alpha} (D_i)_{\alpha}^\gamma $$

$$=\sum_{\alpha} \frac{q_i^{\alpha}}{L_i} \sum_j^i(\lambda^{i-j} k_{\alpha}^j  v_{j}^\gamma)$$
$$=\sum_{\alpha}\frac{q_i^{\alpha}}{L_i} (D_i)_{\alpha}^\gamma$$

$$=\sum_{\alpha}\sum_j \frac{q_i^{\alpha}}{L_i} C_{ij} k_{\alpha}^j v_{j} ^\gamma = \sum_{\alpha}\frac{q_i^{\alpha}}{L_i} (D_i)_{\alpha}^\gamma$$


second line is the formula for reduced $q$ and $k$

third line is the formula for `discounted-cumsum`

(3) ~~[TODO]: dirctly build the operation $\sum_{\alpha}\sum_j \frac{q_i^{\alpha}}{L_i} C_{ij} k_{\alpha}^j v_{j} ^\gamma$~~


Now, the max intermediate is $D$ = (B, H, S, D, D).

----
## TODO list
- ~~broadcast operation~~
- ~~bfloat16 cuda~~ (Lazy implement) (wondering pow percision for bfloat16)
---------------
# Recurrent Formulation
## Parallel mode

Firstly, consider the parallel retention. (We use bold $\bf{a}$ represent Einstein summation)

$$
R_{i}^j = q_i^{\bf{a}}k_{\bf{a}}^j
$$


with the decay $C_i^j = \lambda^{i-j} \delta_{j\leq i}$

```math
R_{i}^j = C_i^j q_i^{\bf{a}}k_{\bf{a}}^j
```

Notice, in the real code, the decay mask will be normalized along $j$ dimension.

```math
\mathcal{C}_i^j = \frac{\lambda^{i-j}}{L_i}\delta_{j\leq i}=\frac{\lambda^{i-j}}{\sqrt{\sum_i^t (\lambda^{i-t})^2}}\delta_{j\leq i}
```

There is one more normalization during parallel. 

```math
\mathcal{R}_{i}^j = \frac{R_i^j }{|\sum_t^j R_i^t|} = \frac{R_i^j }{|P_i|}
```

The final step is $kvq$ 

```math
O_i^c=\mathcal{R}_i^{\bf{j}} v_{\bf{j}}^c = \frac{R_i^{\bf{j}} v_{\bf{j}}^c}{|P_i|} 
= \frac{\mathcal{C}_i^{\bf{j}}q_i^{\bf{a}}k_{\bf{a}}^{\bf{j}}v_{\bf{j}}^c}{|1_{\bf{j}}\mathcal{C}_i^{\bf{j}}q_i^{\bf{a}}k_{\bf{a}}^{\bf{j}}|}
= \frac{\frac{C_i^{\bf{j}}}{Li}q_i^{\bf{a}}k_{\bf{a}}^{\bf{j}}v_{\bf{j}}^c}{|1_{\bf{j}}\frac{C_i^{\bf{j}}}{Li}q_i^{\bf{a}}k_{\bf{a}}^{\bf{j}}|}
```

## Recurrent mode 


```math
O_i^c= \frac{\frac{q_i^{\bf{a}}}{L_i}{C}_i^{\bf{j}}k_{\bf{a}}^{\bf{j}}v_{\bf{j}}^c}{|\frac{q_i^{\bf{a}}}{L_i}1_{\bf{j}}{C}_i^{\bf{j}}k_{\bf{a}}^{\bf{j}}|} 
 = \frac{\bar{q}_i^{\bf{a}}H_{c,{\bf{a}}}^i}{|\bar{q}_i^{\bf{a}}T_{{\bf{a}}}^i|}
```
where 

```math
H_{c,a}^i = C_i^{\bf{j}}k_{a}^{\bf{j}}v_{\bf{j}}^c = \lambda^{i-{\bf{j}}} \delta_{{\bf{j}}\leq i} k_{a}^{\bf{j}}v_{\bf{j}}^c
```
and 

```math
T_{a}^i = 1_{\bf{j}}{C}_i^{\bf{j}}k_{a}^{\bf{j}}=\lambda^{i-{\bf{j}}} 1_{\bf{j}}\delta_{{\bf{j}}\leq i} k_{a}^{\bf{j}}
``` 

For example, let omit $a$ and $c$ and write down the first row along $i$

```math
\begin{align}
H^0 &= k^0 v_0 && &&\\
H^1 &= \lambda k^0v_0 &+& k^1 v_1 \\
H^2 &= \lambda^2 k^0v_0 &+& \lambda k^1 v_1 &+& k^2 v_2 \\
\cdots
\end{align}
```
It has obviously recurrent formulation:

```math
H^{n+1}_{c,a} = \lambda H^{n}_{c,a} + k_a^{n+1}v_{n+1}^c
```
Same

```math
T_{a}^{n+1}=\lambda T_a^{n} + k_a^{n+1}
```

## Chunkwise

Now consider the chunk-wise recurrent that forward with multi step.

```math
\begin{align}
H^0 &= k^0 v_0 \\
H^1 &= \lambda k^0v_0 &+& k^1 v_1 \\
H^2 &= \lambda^2 k^0v_0 &+& \lambda k^1 v_1 &+& k^2 v_2 \\
H^3 &= \lambda^3 k^0v_0 &+& \lambda k^2 v_1 &+& \lambda k^1 v_2 &+& k^3 v_3 \\
H^4 &= \lambda^4 k^0v_0 &+& \lambda k^3 v_1 &+& \lambda k^2 v_2 &+& \lambda k^3 v_3&+& k^4 v_4 \\
\cdots
\end{align}
```


```math
H^{n:n+m} = [\lambda,\lambda^2,\cdots,\lambda^m]\otimes H^n+ \sum_{row}{\bf{C}}{\bf{k}}^{n:n+m}{\bf{v}}_{n:n+m}
```

where the $\bf{C}$ is the decay mask like

```math
\begin{align}
 1        & &           & &           & &         & &     \\
\lambda   & &     1     & &           & &         & &     \\
\lambda^2 & & \lambda   & &     1     & &         & &     \\
\lambda^3 & & \lambda^2 & & \lambda   & &    1    & &     \\
\lambda^4 & & \lambda^3 & & \lambda^2 & & \lambda & &  1  \\
\end{align}
```
which is the first $m$ row of origin decay mask.

Thus, the recurrent - chunk_size formulation is 

```python
current_kv = torch.einsum('Hi,BHiac->BHiac', gamma, past_kv) + 	 
		  	 torch.einsum('Hij,BHja,BHjc-> BHiac', mask, k_more, v_more)
current_gk = torch.einsum('Hi,BHia->BHia', gamma, past_gk) + 
			 torch.einsum('Hij,BHja-> BHia', mask, k_more)
```
Notice, at current (2023.10.24), the `group_norm` is use `RMSNorm` which normalize the $D2$ dimension and ensure each vector $|x \rangle$ (D2,) in (B, H, S) satisfy:
```math
\langle x|x \rangle=D2
```
That is 
```math
X_i^{c}\rightarrow \frac{X_i^{c}}{\sqrt{X_i^\alpha X_i^\alpha}}
```
Given any **gauge** only effect on the $i$ dimension $Y_i^c =  T_iX_i^c$.

The result will hold invariant.
```math
\frac{Y_i^{c}}{\sqrt{Y_i^\alpha Y_i^\alpha}}=\frac{T_iX_i^c}{\sqrt{T_iX_i^\alpha T_iX_i^\alpha}}=\frac{X_i^{c}}{\sqrt{X_i^\alpha X_i^\alpha}}
```
This mean we can totally remove the $P_i$ or `current_gk` if we finally apply the group_norm normalization. 

For numerical reason, we can rescale the `retention` use $P_i$ to avoid `Nan` or `Inf`. 

The cost to obtain  `current_gk`  and `current_kv` almost same, thus, disable computing the `retention` vis set `normlize_for_stable=False` can indeed accelerate the inference 2 times fast.
