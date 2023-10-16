# Faster_Retention
nX speed up implementation of retention operation of Retention Networks. (https://arxiv.org/pdf/2307.08621.pdf)
The implementation is on par with the official implementation at torchscale repo.

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
```
python test.py

    B     S   D        e1        e2      fast    reduce    origin
0   2    30   4  0.000015  0.000015  0.000620  0.000306  0.000208
1   2    30   8  0.000032  0.000033  0.000611  0.000329  0.000190
2   2    30  16  0.000067  0.000079  0.000655  0.000319  0.000192
3   2    30  32  0.000168  0.000182  0.000776  0.000324  0.000195
4   2   300   4  0.000081  0.000116  0.000726  0.000404  0.000397
5   2   300   8  0.000153  0.000212  0.000768  0.000440  0.000397
6   2   300  16  0.000305  0.000428  0.001063  0.000504  0.000400
7   2   300  32  0.000707  0.001000  0.002192  0.000608  0.000399
8   2  3000   4  0.000515       NaN  0.001859  0.002302  0.016224
9   2  3000   8  0.000936       NaN  0.002565  0.002805  0.016248
10  2  3000  16  0.001750       NaN  0.006313  0.003227  0.016289
11  2  3000  32  0.003378       NaN  0.021750  0.004380  0.016682
```
