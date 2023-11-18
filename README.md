# TSMixer in PyTorch

Reimplementation of TSMixer in PyTorch.

* Original paper: [https://arxiv.org/pdf/2303.06053.pdf](https://arxiv.org/abs/2303.06053)

## Implementation notes from the paper

### Training parameters

> For multivariate long-term forecasting datasets, we follow the settings in recent research (Liu et al., 2022b; Zhou et al., 2022a; Nie et al., 2023). We set the input length L = 512 as suggested in Nie et al. (2023) and evaluate the results for prediction lengths of T = {96, 192, 336, 720}. We use the Adam optimization algorithm (Kingma & Ba, 2015) to minimize the mean square error (MSE) training objective, and consider MSE and mean absolute error (MAE) as the evaluation metrics. We apply reversible instance normalization (Kim et al., 2022) to ensure a fair comparison with the state-of-the-art PatchTST (Nie et al., 2023).

> For the M5 dataset, we mostly follow the data processing from Alexandrov et al. (2020). We consider the prediction length of T = 28 (same as the competition), and set the input length to L = 35. We optimize log-likelihood of negative binomial distribution as suggested by Salinas et al. (2020). We follow the competition’s protocol (Makridakis et al., 2022) to aggregate the predictions at different levels and evaluate them using the weighted root mean squared scaled error (WRMSSE). More details about the experimental setup and hyperparameter tuning can be found in Appendices C and E.

### Reversible Instance Normalization for Time Series Forecasting

Reversible instance normalization https://openreview.net/pdf?id=cGDAkQo1C0p

> First, we normalize the input data x(i) using its instance-specific mean and stan- dard deviation, which is widely accepted as instance normalization (Ulyanov et al., 2016). The mean and standard deviation are computed for every instance x(i) ∈ RTx of the input data (Fig. 2(a-3)) as

```
Mean[xi_kt] = mean_{j=1}^Tx ( xi_kj )
Var[xi_kt] = var_{j=1}^Tx ( xi_kj )
```
Where `i` = sample in the batch, `K` = num variables (features), `Tx` = num time steps in input, `Ty` = num time steps in output (prediction).

> Then, we apply the normalization to the **input data** (sent to model) as

```
xhati_kt = gamma_k * (xi_kt - Mean[xi_kt]) / sqrt(Var[xi_kt] + epsilon) + beta_k
```

where gamma_k and beta_k are learnable parameters for each variable k (**recall: K = num features**).

After final layer of model, we get output `yhati_kt`, apply the reverse transformation to the **output data** (sent to loss function) as

```
yi_kt = (yhati_kt - beta_k) * sqrt(Var[xi_kt] + epsilon) / gamma_k + Mean[xi_kt]
```

where `yhati_kt` is the output of the model for variable `k` at time `t` for sample `i`, and `yi_kt` is sent to the loss function.