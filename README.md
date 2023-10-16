# Faster_Retention
nX speed up implementation of retention operation of Retention Networks. (https://arxiv.org/pdf/2307.08621.pdf)
The implementation is on par with the official implementation at [`torchscale`](https://github.com/microsoft/torchscale) repo and [huggingface transformer compatible implementation of Retention Networks](https://github.com/syncdoth/RetNet)

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
## 

