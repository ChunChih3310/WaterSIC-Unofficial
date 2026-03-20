# Layer1 o_proj Residual Debug Report

- Timestamp: `2026-03-13T05:57:00Z`
- Git commit: `2cd4111a008137dc73453c348ccc4a760a3e1188`
- Model: `meta-llama/Llama-3.2-1B`
- Target module: `model.layers.1.self_attn.o_proj`
- Baseline small-eval PPL: `8.9880`
- Target rate at o_proj: `3.5564`
- Reference stats requested: `True`
- Reference stats used: `True`
- Rescalers enabled: `False`

## Config Audit

- Calibration: `train` / `6` sequences / len `2048` / batch `1`
- Probe eval: `test` / `8` sequences / len `2048` / batch `1`
- Residual scales: `[0.0, 0.25, 0.5, 0.75, 1.0]`
- Legacy ridge for audit: `1e-06`

## Timing Audit

- Stage timing: `same-layer, post-QKV, pre-o_proj`
- Delta definition: `Delta = R - R_hat = reference_layer_input - quantized_layer_input`
- Residual target kind check: `True`
- qkv weight MSE before o_proj: `{'q_proj': 0.05565570563235518, 'k_proj': 0.07743969640594256, 'v_proj': 0.06151273806661382}`
- o_proj weight MSE before quantization: `0.000000e+00`
- Manual Sigma_Delta match error: `0.000000e+00`
- Wrong-sign mismatch error: `3.840059e-03`
- Manual Sigma_Xhat match error: `0.000000e+00`
- Layer-input delta ||R-Rhat||_F (first batch): `4.105417e+00`
- o_proj-input delta ||X-Xhat||_F (first batch): `7.903088e+00`

## Sweep

| Residual Scale | Small-Eval PPL | Rel Weight MSE | ||WΣ|| | ||ΣΔ|| | ||sum|| | ||ΣΔ||/||WΣ|| | H cond | max |Y| | legacy max |Y| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 9.6219 | 1.542571e-01 | 5.334849e-01 | 1.920029e-03 | 5.334849e-01 | 0.000000e+00 | 5.432919e+05 | 1.904271e-01 | 8.594701e+01 |
| 0.25 | 9.6313 | 1.575171e-01 | 5.334849e-01 | 1.920029e-03 | 5.334989e-01 | 8.997581e-04 | 5.432919e+05 | 1.904301e-01 | 8.594701e+01 |
| 0.50 | 9.6356 | 1.677981e-01 | 5.334849e-01 | 1.920029e-03 | 5.335133e-01 | 1.799516e-03 | 5.432919e+05 | 1.904330e-01 | 8.594701e+01 |
| 0.75 | 9.6459 | 1.860864e-01 | 5.334849e-01 | 1.920029e-03 | 5.335282e-01 | 2.699274e-03 | 5.432919e+05 | 1.904360e-01 | 8.594701e+01 |
| 1.00 | 9.6433 | 2.124589e-01 | 5.334849e-01 | 1.920029e-03 | 5.335435e-01 | 3.599032e-03 | 5.432919e+05 | 1.904390e-01 | 8.594701e+01 |
