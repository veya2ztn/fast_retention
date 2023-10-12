
2X speed up implementation of retention operation of Retention Networks. (https://arxiv.org/pdf/2307.08621.pdf)
The implementation is on par with the official implementation at torchscale repo.

We use [`torch-discounted-cumsum`](https://github.com/toshas/torch-discounted-cumsum) to accelerate computation.

```
pip install torch-discounted-cumsum --no-build-isolation
```

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

However, it is $\mathcal{O}(S^2)$ implement as the appear of intermeidate shape $(S,S)$. It is wellknow retention is a 'Linear' like Transfomer, thus it is possible we contract the $S$ dimension first and reduce the complexity into $\mathcal{O}(D^2)$.

Lets drive it step by step.

Firstly, 
$$
R_{i}^j = q_i^{\alpha}k_{\alpha}^j
$$

with decay mask

$$
R_{i}^j = C_i^j q_i^{\alpha}k_{\alpha}^j
$$
where
$$
C_i^j = \lambda^{i-j} \delta_{i>j}
$$
Thus
$$
R_{i}^j = \lambda^{i} q_i^{\alpha}k_{\alpha}^j \lambda^{-j} =  \bar{q}_i^{\alpha}\bar{k}_{\alpha}^j \delta_{i>j}
$$

One thing that paper don't say is it will do normlaization.
$$
C_i^j = \frac{\lambda^{i-j}}{\sqrt{\sum_i^t (\lambda^{i-t})^2}}=\frac{\lambda^{i-j}}{L_i}=\frac{\lambda^i}{L_i}\lambda^{-j} \delta_{i>j}
$$

Notice all the coef is fixed, thus we can precompute at early begining.

Now, we have reduced $q$ and $k$
$$
\bar{q}_i^{\alpha} =  \frac{\lambda^i}{L_i}  q_i^{\alpha}
$$

$$
\bar{k}_{\alpha}^j =  \lambda^{-j} k_{\alpha}^j 
$$

Second thing that paper don't say here: another normlaization.
$$
R_{i}^j \rightarrow \frac{R_i^j }{|\sum_t^j R_i^t|} = \frac{R_i^j }{|P_i|}
$$

The $P_i$ can dircly compute
$$
\begin{align}
P_i=\sum_t R_i^t = \sum_t (\bar{q}_i^{\alpha}\bar{k}_{\alpha}^t \delta_{i>t})
=\sum_t^i (\bar{q}_i^{\alpha}\bar{k}_{\alpha}^t)
&=\bar{q}_i^{\alpha}\sum_t^i (\bar{k}_{\alpha}^t)=\bar{q}_i^{\alpha}T_{\alpha}^i\\
&=\frac{q_i^{\alpha}}{L_i}\sum_t^i (\lambda^{i-j}k_{\alpha}^t)=\frac{q_i^{\alpha}}{L_i}\mathbf{T}_{\alpha}^i\\

\end{align}
$$
(1) is the formula for reduced $q$ and $k$

(2) is the formula for `discounted-cumsum`

Next
$$
O_i^\gamma=R_i^j v_{j}^\gamma = \frac{R_i^j v_{j}^\gamma}{|P_i|}= \frac{o_i^\gamma}{|P_i|}
$$

Still can compute

$$
\begin{align}
o_i^{\gamma} = R_i^j v_{j}^\gamma = \bar{q}_i^{\alpha}\delta_{i>j}\bar{k}_{\alpha}^j  v_{j}^\gamma
&= \bar{q}_i^{\alpha}(\delta_{i>j} \bar{k}_{\alpha}^j  v_{j}^\gamma) = \bar{q}_i^{\alpha} (D_i)_{\alpha}^\gamma\\
&=\sum_{\alpha} \frac{q_i^{\alpha}}{L_i} \sum_j^i(\lambda^{i-j} k_{\alpha}^j  v_{j}^\gamma) = \sum_{\alpha}\frac{q_i^{\alpha}}{L_i} (\mathcal{D}_i)_{\alpha}^\gamma\\
&=\sum_{\alpha}\sum_j \frac{q_i^{\alpha}}{L_i} C_{ij} k_{\alpha}^j v_{j} ^\gamma = \sum_{\alpha}\frac{q_i^{\alpha}}{L_i} (\mathcal{D}_i)_{\alpha}^\gamma\\
\end{align}
$$

(1) is the formula for reduced $q$ and $k$

(2) is the formula for `discounted-cumsum`

Now, the max intermediate is $D$ = (B, H, S, D, D).