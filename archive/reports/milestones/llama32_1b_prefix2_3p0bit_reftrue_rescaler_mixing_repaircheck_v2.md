# WaterSIC Run Report

- Timestamp: `2026-03-14T17:02:30Z`
- Git commit: `efda59d419fc818c6ad4becb63d0d3d195813335`
- Environment: `watersic`
- Device: `cuda`
- Model: `meta-llama/Llama-3.2-1B`
- Model revision: `main`
- Tokenizer: `meta-llama/Llama-3.2-1B`
- Sequence length: `2048`
- Calibration sequences: `8`
- Target global bitwidth: `3.0000`
- Achieved global bitwidth: `2.9862`
- Raw average bitwidth: `10.0431`
- Entropy average bitwidth: `2.9743`
- Huffman average bitwidth: `3.0262`
- Side-information overhead: `0.0119`
- Perplexity: `9.560034225816583`

## Layer Summary

| Layer | Kind | Target | Achieved | Entropy | Huffman | Side Info | Weighted Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model.layers.0.self_attn.q_proj | q_proj | 3.0000 | 2.8868 | 2.8707 | 2.9284 | 0.0161 | 3.240816e-07 |
| model.layers.0.self_attn.k_proj | k_proj | 3.0040 | 2.7057 | 2.6646 | 2.7310 | 0.0410 | 7.469823e-07 |
| model.layers.0.self_attn.v_proj | v_proj | 3.0067 | 2.6269 | 2.5858 | 2.6493 | 0.0410 | 2.751028e-08 |
| model.layers.0.self_attn.o_proj | o_proj | 3.0102 | 2.8778 | 2.8617 | 2.8914 | 0.0161 | 2.478157e-10 |
| model.layers.0.mlp.gate_proj | gate_proj | 3.0152 | 2.9250 | 2.9151 | 2.9679 | 0.0099 | 1.288176e-07 |
| model.layers.0.mlp.up_proj | up_proj | 3.0312 | 2.9444 | 2.9345 | 2.9914 | 0.0099 | 9.904060e-08 |
| model.layers.0.mlp.down_proj | down_proj | 3.0500 | 2.9586 | 2.9484 | 2.9923 | 0.0103 | 4.153092e-10 |
| model.layers.1.self_attn.q_proj | q_proj | 3.0752 | 2.9263 | 2.9101 | 2.9537 | 0.0162 | 6.031503e-07 |
| model.layers.1.self_attn.k_proj | k_proj | 3.0862 | 2.6425 | 2.6015 | 2.6570 | 0.0411 | 2.079251e-06 |
| model.layers.1.self_attn.v_proj | v_proj | 3.0946 | 2.6996 | 2.6586 | 2.7001 | 0.0411 | 1.065424e-07 |
| model.layers.1.self_attn.o_proj | o_proj | 3.1022 | 2.9757 | 2.9596 | 3.0045 | 0.0161 | 1.057592e-09 |
| model.layers.1.mlp.gate_proj | gate_proj | 3.1127 | 3.0236 | 3.0137 | 3.0745 | 0.0099 | 1.763116e-07 |
| model.layers.1.mlp.up_proj | up_proj | 3.1572 | 3.0686 | 3.0587 | 3.1147 | 0.0099 | 1.190624e-07 |
| model.layers.1.mlp.down_proj | down_proj | 3.2459 | 3.1459 | 3.1357 | 3.1834 | 0.0103 | 5.746874e-10 |
