# WaterSIC Run Report

- Timestamp: `2026-03-13T04:09:42Z`
- Git commit: `be111fcc0b57299450d969eaab8a98e2a1ce5b03`
- Environment: `watersic`
- Device: `cuda`
- Model: `meta-llama/Llama-3.2-1B`
- Model revision: `main`
- Tokenizer: `meta-llama/Llama-3.2-1B`
- Sequence length: `2048`
- Calibration sequences: `6`
- Target global bitwidth: `3.0000`
- Achieved global bitwidth: `2.9912`
- Raw average bitwidth: `9.3971`
- Entropy average bitwidth: `2.9779`
- Huffman average bitwidth: `3.0340`
- Side-information overhead: `0.0133`
- Perplexity: `700443.0656432873`

## Layer Summary

| Layer | Kind | Target | Achieved | Entropy | Huffman | Side Info | Weighted Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model.layers.0.self_attn.q_proj | q_proj | 3.0000 | 2.8762 | 2.8601 | 2.9176 | 0.0161 | 2.754837e-07 |
| model.layers.0.self_attn.k_proj | k_proj | 3.0077 | 2.6968 | 2.6557 | 2.7205 | 0.0410 | 6.403970e-07 |
| model.layers.0.self_attn.v_proj | v_proj | 3.0127 | 2.6360 | 2.5949 | 2.6567 | 0.0410 | 2.304000e-08 |
| model.layers.0.self_attn.o_proj | o_proj | 3.0187 | 2.8806 | 2.8645 | 2.8927 | 0.0161 | 2.313133e-10 |
| model.layers.0.mlp.gate_proj | gate_proj | 3.0283 | 2.9346 | 2.9247 | 2.9759 | 0.0099 | 1.169041e-07 |
| model.layers.0.mlp.up_proj | up_proj | 3.0639 | 2.9772 | 2.9673 | 3.0232 | 0.0099 | 8.699814e-08 |
| model.layers.0.mlp.down_proj | down_proj | 3.1173 | 3.0271 | 3.0169 | 3.0845 | 0.0103 | 3.128655e-10 |
| model.layers.1.self_attn.q_proj | q_proj | 3.2617 | 3.1064 | 3.0902 | 3.1522 | 0.0162 | 4.293394e-07 |
| model.layers.1.self_attn.k_proj | k_proj | 3.3652 | 2.9084 | 2.8674 | 2.9060 | 0.0411 | 1.308113e-06 |
| model.layers.1.self_attn.v_proj | v_proj | 3.4565 | 3.0570 | 3.0159 | 3.0688 | 0.0411 | 5.877239e-08 |
| model.layers.1.self_attn.o_proj | o_proj | 3.5564 | 3.4066 | 3.3904 | 3.4429 | 0.0161 | 7.153965e-02 |
