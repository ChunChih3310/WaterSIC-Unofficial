# WaterSIC Run Report

- Timestamp: `2026-03-12T21:36:08Z`
- Git commit: `00d7223ac98d5d6e268a3f2ce7d0598f4aa9de7d`
- Environment: `watersic`
- Device: `cuda`
- Model: `meta-llama/Llama-3.2-1B`
- Model revision: `main`
- Tokenizer: `meta-llama/Llama-3.2-1B`
- Sequence length: `2048`
- Calibration sequences: `2`
- Target global bitwidth: `3.0000`
- Achieved global bitwidth: `3.0105`
- Raw average bitwidth: `10.0000`
- Entropy average bitwidth: `2.9986`
- Huffman average bitwidth: `3.0433`
- Side-information overhead: `0.0119`
- Perplexity: `18227.769720855307`

## Layer Summary

| Layer | Kind | Target | Achieved | Entropy | Huffman | Side Info | Weighted Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model.layers.0.self_attn.q_proj | q_proj | 3.0000 | 2.9463 | 2.9302 | 2.9763 | 0.0161 | 4.778460e+06 |
| model.layers.0.self_attn.k_proj | k_proj | 3.0040 | 2.6648 | 2.6238 | 2.6607 | 0.0410 | 4.243886e+06 |
| model.layers.0.self_attn.v_proj | v_proj | 3.0104 | 2.8375 | 2.7965 | 2.8383 | 0.0410 | 1.012416e+03 |
| model.layers.0.self_attn.o_proj | o_proj | 3.0137 | 2.9349 | 2.9188 | 2.9338 | 0.0161 | 7.754834e+02 |
| model.layers.0.mlp.gate_proj | gate_proj | 3.0203 | 2.9948 | 2.9849 | 3.0249 | 0.0099 | 3.167971e+01 |
| model.layers.0.mlp.up_proj | up_proj | 3.0330 | 2.9814 | 2.9715 | 3.0125 | 0.0099 | 9.343440e+00 |
| model.layers.0.mlp.down_proj | down_proj | 3.0845 | 3.1226 | 3.1123 | 3.1734 | 0.0103 | 9.675413e+02 |
