# WaterSIC Run Report

- Timestamp: `2026-03-12T20:57:52Z`
- Git commit: `unknown`
- Environment: `watersic`
- Device: `cuda`
- Model: `meta-llama/Llama-3.2-1B`
- Model revision: `main`
- Tokenizer: `meta-llama/Llama-3.2-1B`
- Sequence length: `2048`
- Calibration sequences: `2`
- Target global bitwidth: `3.0000`
- Achieved global bitwidth: `3.0104`
- Raw average bitwidth: `10.0000`
- Entropy average bitwidth: `2.9985`
- Huffman average bitwidth: `3.0431`
- Side-information overhead: `0.0119`
- Perplexity: `15835.218600489361`

## Layer Summary

| Layer | Kind | Target | Achieved | Entropy | Huffman | Side Info | Weighted Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model.layers.0.self_attn.q_proj | q_proj | 3.0000 | 2.9543 | 2.9381 | 2.9828 | 0.0161 | 1.260407e+07 |
| model.layers.0.self_attn.k_proj | k_proj | 3.0034 | 2.6366 | 2.5956 | 2.6245 | 0.0410 | 1.097894e+07 |
| model.layers.0.self_attn.v_proj | v_proj | 3.0103 | 2.8294 | 2.7884 | 2.8357 | 0.0410 | 2.213290e+03 |
| model.layers.0.self_attn.o_proj | o_proj | 3.0138 | 2.9349 | 2.9188 | 2.9338 | 0.0161 | 7.754834e+02 |
| model.layers.0.mlp.gate_proj | gate_proj | 3.0204 | 2.9948 | 2.9849 | 3.0249 | 0.0099 | 3.167971e+01 |
| model.layers.0.mlp.up_proj | up_proj | 3.0331 | 2.9814 | 2.9715 | 3.0125 | 0.0099 | 9.343440e+00 |
| model.layers.0.mlp.down_proj | down_proj | 3.0848 | 3.1226 | 3.1123 | 3.1734 | 0.0103 | 9.675413e+02 |
