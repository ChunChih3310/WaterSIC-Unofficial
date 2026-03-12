# Algorithm Notes

## Plain WaterSIC

From Section 3 of the paper:

- Objective: minimize transformed error `||Y - W_hat L||_F^2` with `Y = W L`.
- Cholesky factor: `Sigma_X = L L^T`.
- Unequal spacing: `alpha_i = c / L_ii`.
- ZSIC recursion runs from the last column to the first:
  - `z_i = round(Y_i / c)`
  - `gamma_i = <Y_i, c z_i> / ||c z_i||^2`
  - update `Y <- Y - gamma_i c z_i L_i,:`
- Preliminary reconstruction is `W_0 = Z diag(alpha)`.
- Final per-column LMMSE factor starts from `Gamma = diag(gamma)`.

## Full WaterSIC Terms

Activation drift correction:

- replace the plain objective with `min ||Y_hat - W_hat L_hat||_F^2`
- `Sigma_Xhat = E[X_hat X_hat^T]`
- `Sigma_X,Xhat = E[X X_hat^T]`
- `Y_hat = W Sigma_X,Xhat (L_hat^T)^(-1)`

Residual stream correction for `o_proj` and `down_proj`:

- `Sigma_Delta,Xhat = E[(R - R_hat) X_hat^T]`
- corrected transformed target uses:
  - `Y_hat = (W Sigma_X,Xhat + Sigma_Delta,Xhat) (L_hat^T)^(-1)`

Attention-weighted calibration for QKV:

- token importance score:
  - `p_j = (1 / (N H (T - j))) sum_h sum_{i=j}^{T-1} alpha_{h,i,j}`
- weighted moments replace uniform averages only for `q_proj`, `k_proj`, `v_proj`

Adaptive mixing from equation (20):

- `Sigma_X(final) = (1 - epsilon_aw) Sigma_X^(w) + epsilon_aw Sigma_X`
- analogous mixing applies to `Sigma_Xhat` and `Sigma_X,Xhat`, with `epsilon_qr` replacing `X_hat` by `X`

## What The Current Code Does

- The transformed-space ZSIC implementation now follows the paper’s `Y, L, alpha_i, gamma_i` formulation.
- The layer quantizer forms `Sigma_Xhat`, `Sigma_X,Xhat`, and `Sigma_Delta,Xhat`, then solves the transformed objective by triangular solve instead of explicit inverse.
- The diagonal rescaler step is implemented in the transformed objective using the update equations from Section 4.
- Adaptive mixing search is not fully optimized yet; fixed `epsilon_qr` and `epsilon_aw` values are supported in configs.
