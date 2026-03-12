# WaterSIC Run Report

- Timestamp: `2026-03-12T21:10:27Z`
- Git commit: `ec44fef656b9c4d3312dbf440385794d15cfb05c`
- Environment: `watersic`
- Device: `cuda`
- Model: `meta-llama/Llama-3.2-1B`
- Model revision: `main`
- Tokenizer: `meta-llama/Llama-3.2-1B`
- Sequence length: `2048`
- Calibration sequences: `2`
- Target global bitwidth: `3.0000`
- Achieved global bitwidth: `3.0102`
- Raw average bitwidth: `10.0000`
- Entropy average bitwidth: `2.9983`
- Huffman average bitwidth: `3.0427`
- Side-information overhead: `0.0119`
- Perplexity: `24887.471472403067`

## Layer Summary

| Layer | Kind | Target | Achieved | Entropy | Huffman | Side Info | Weighted Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model.layers.0.self_attn.q_proj | q_proj | 3.0000 | 2.9512 | 2.9351 | 2.9779 | 0.0161 | 1.505816e+07 |
| model.layers.0.self_attn.k_proj | k_proj | 3.0036 | 2.6372 | 2.5961 | 2.6232 | 0.0410 | 1.316466e+07 |
| model.layers.0.self_attn.v_proj | v_proj | 3.0105 | 2.8305 | 2.7895 | 2.8345 | 0.0410 | 2.724183e+03 |
| model.layers.0.self_attn.o_proj | o_proj | 3.0140 | 2.9349 | 2.9188 | 2.9338 | 0.0161 | 7.754834e+02 |
| model.layers.0.mlp.gate_proj | gate_proj | 3.0206 | 2.9948 | 2.9849 | 3.0249 | 0.0099 | 3.167971e+01 |
| model.layers.0.mlp.up_proj | up_proj | 3.0335 | 2.9814 | 2.9715 | 3.0125 | 0.0099 | 9.343440e+00 |
| model.layers.0.mlp.down_proj | down_proj | 3.0855 | 3.1226 | 3.1123 | 3.1734 | 0.0103 | 9.675413e+02 |
