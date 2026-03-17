# WaterSIC Run Report

- Timestamp: `2026-03-16T18:58:57Z`
- Git commit: `df5e4c7f380b31ac7389e26f6036cb73ee7cfecb`
- Environment: `watersic`
- Device: `cuda`
- Model: `meta-llama/Llama-3.2-1B`
- Model revision: `main`
- Tokenizer: `meta-llama/Llama-3.2-1B`
- Sequence length: `2048`
- Calibration sequences: `64`
- Target global bitwidth: `3.0000`
- Achieved global bitwidth: `2.9984`
- Raw average bitwidth: `8.4159`
- Entropy average bitwidth: `2.9865`
- Huffman average bitwidth: `3.0376`
- Huffman shortest symbol length: `1`
- Huffman longest symbol length: `25`
- Side-information overhead: `0.0119`
- Perplexity: `11.187436789939435`

## Layer Summary

| Layer | Kind | Target | Achieved | Entropy | Huffman | Huff Min | Huff Max | Side Info | Weighted Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model.layers.0.self_attn.q_proj | q_proj | 3.0000 | 2.8882 | 2.8721 | 2.9314 | 2 | 22 | 0.0161 | 4.319680e-07 |
| model.layers.0.self_attn.k_proj | k_proj | 3.0005 | 2.7042 | 2.6632 | 2.7295 | 1 | 20 | 0.0410 | 9.919665e-07 |
| model.layers.0.self_attn.v_proj | v_proj | 3.0008 | 2.6204 | 2.5793 | 2.6471 | 2 | 18 | 0.0410 | 3.632250e-08 |
| model.layers.0.self_attn.o_proj | o_proj | 3.0012 | 2.8637 | 2.8475 | 2.8828 | 1 | 22 | 0.0161 | 3.128323e-10 |
| model.layers.0.mlp.gate_proj | gate_proj | 3.0018 | 2.9109 | 2.9009 | 2.9572 | 2 | 24 | 0.0099 | 1.641274e-07 |
| model.layers.0.mlp.up_proj | up_proj | 3.0034 | 2.9166 | 2.9067 | 2.9648 | 2 | 25 | 0.0099 | 1.285709e-07 |
| model.layers.0.mlp.down_proj | down_proj | 3.0050 | 2.9134 | 2.9032 | 2.9521 | 2 | 23 | 0.0103 | 7.939579e-10 |
| model.layers.1.self_attn.q_proj | q_proj | 3.0067 | 2.8583 | 2.8421 | 2.8829 | 2 | 22 | 0.0162 | 7.769446e-07 |
| model.layers.1.self_attn.k_proj | k_proj | 3.0074 | 2.5637 | 2.5227 | 2.5902 | 2 | 19 | 0.0411 | 2.723281e-06 |
| model.layers.1.self_attn.v_proj | v_proj | 3.0079 | 2.6130 | 2.5720 | 2.6231 | 2 | 19 | 0.0411 | 1.402200e-07 |
| model.layers.1.self_attn.o_proj | o_proj | 3.0083 | 2.8814 | 2.8652 | 2.9038 | 2 | 22 | 0.0161 | 1.604475e-09 |
| model.layers.1.mlp.gate_proj | gate_proj | 3.0089 | 2.9204 | 2.9105 | 2.9662 | 2 | 24 | 0.0099 | 2.387666e-07 |
| model.layers.1.mlp.up_proj | up_proj | 3.0106 | 2.9218 | 2.9118 | 2.9700 | 2 | 24 | 0.0099 | 1.713213e-07 |
| model.layers.1.mlp.down_proj | down_proj | 3.0123 | 2.9019 | 2.8917 | 2.9400 | 2 | 23 | 0.0103 | 1.455786e-09 |
| model.layers.2.self_attn.q_proj | q_proj | 3.0145 | 2.9233 | 2.9072 | 2.9539 | 2 | 22 | 0.0161 | 1.039799e-06 |
| model.layers.2.self_attn.k_proj | k_proj | 3.0149 | 2.6366 | 2.5955 | 2.6427 | 2 | 19 | 0.0410 | 3.361334e-06 |
| model.layers.2.self_attn.v_proj | v_proj | 3.0154 | 2.6479 | 2.6069 | 2.6495 | 2 | 20 | 0.0410 | 2.783727e-07 |
| model.layers.2.self_attn.o_proj | o_proj | 3.0159 | 2.8818 | 2.8656 | 2.9125 | 2 | 22 | 0.0161 | 2.121517e-09 |
| model.layers.2.mlp.gate_proj | gate_proj | 3.0165 | 2.9327 | 2.9228 | 2.9770 | 2 | 23 | 0.0099 | 3.490398e-07 |
| model.layers.2.mlp.up_proj | up_proj | 3.0182 | 2.9283 | 2.9184 | 2.9777 | 2 | 24 | 0.0099 | 2.160974e-07 |
| model.layers.2.mlp.down_proj | down_proj | 3.0201 | 2.9232 | 2.9129 | 2.9619 | 2 | 23 | 0.0103 | 2.664949e-09 |
| model.layers.3.self_attn.q_proj | q_proj | 3.0222 | 2.8397 | 2.8235 | 2.8639 | 2 | 22 | 0.0161 | 1.221865e-06 |
| model.layers.3.self_attn.k_proj | k_proj | 3.0231 | 2.7183 | 2.6773 | 2.7169 | 2 | 19 | 0.0410 | 2.882135e-06 |
| model.layers.3.self_attn.v_proj | v_proj | 3.0236 | 2.7144 | 2.6733 | 2.7244 | 2 | 19 | 0.0410 | 3.679830e-07 |
| model.layers.3.self_attn.o_proj | o_proj | 3.0240 | 2.8890 | 2.8729 | 2.9233 | 2 | 21 | 0.0161 | 4.885156e-09 |
| model.layers.3.mlp.gate_proj | gate_proj | 3.0247 | 2.9256 | 2.9157 | 2.9661 | 2 | 22 | 0.0099 | 4.919126e-07 |
| model.layers.3.mlp.up_proj | up_proj | 3.0269 | 2.9387 | 2.9288 | 2.9886 | 2 | 24 | 0.0099 | 2.388134e-07 |
| model.layers.3.mlp.down_proj | down_proj | 3.0289 | 2.9352 | 2.9249 | 2.9740 | 2 | 23 | 0.0103 | 3.683024e-09 |
| model.layers.4.self_attn.q_proj | q_proj | 3.0310 | 2.8978 | 2.8816 | 2.9230 | 2 | 22 | 0.0161 | 1.067965e-06 |
| model.layers.4.self_attn.k_proj | k_proj | 3.0318 | 2.6756 | 2.6345 | 2.6757 | 2 | 19 | 0.0410 | 3.082573e-06 |
| model.layers.4.self_attn.v_proj | v_proj | 3.0323 | 2.7305 | 2.6895 | 2.7413 | 2 | 20 | 0.0410 | 3.408980e-07 |
| model.layers.4.self_attn.o_proj | o_proj | 3.0327 | 2.8993 | 2.8832 | 2.9261 | 2 | 22 | 0.0161 | 6.260064e-09 |
| model.layers.4.mlp.gate_proj | gate_proj | 3.0335 | 2.9493 | 2.9394 | 2.9907 | 2 | 23 | 0.0099 | 4.971951e-07 |
| model.layers.4.mlp.up_proj | up_proj | 3.0355 | 2.9359 | 2.9260 | 2.9846 | 2 | 24 | 0.0099 | 2.283114e-07 |
| model.layers.4.mlp.down_proj | down_proj | 3.0380 | 2.9474 | 2.9371 | 2.9861 | 2 | 23 | 0.0103 | 3.969219e-09 |
| model.layers.5.self_attn.q_proj | q_proj | 3.0402 | 2.9404 | 2.9243 | 2.9716 | 2 | 23 | 0.0161 | 1.109149e-06 |
| model.layers.5.self_attn.k_proj | k_proj | 3.0409 | 2.6070 | 2.5660 | 2.6195 | 2 | 19 | 0.0410 | 4.144979e-06 |
| model.layers.5.self_attn.v_proj | v_proj | 3.0415 | 2.6598 | 2.6188 | 2.6609 | 2 | 20 | 0.0410 | 3.178220e-07 |
| model.layers.5.self_attn.o_proj | o_proj | 3.0422 | 2.8979 | 2.8818 | 2.9309 | 2 | 22 | 0.0161 | 6.733499e-09 |
| model.layers.5.mlp.gate_proj | gate_proj | 3.0431 | 2.9597 | 2.9498 | 3.0006 | 2 | 24 | 0.0099 | 4.446204e-07 |
| model.layers.5.mlp.up_proj | up_proj | 3.0452 | 2.9569 | 2.9470 | 3.0075 | 2 | 24 | 0.0099 | 2.359622e-07 |
| model.layers.5.mlp.down_proj | down_proj | 3.0476 | 2.9622 | 2.9520 | 3.0006 | 2 | 24 | 0.0103 | 4.326890e-09 |
| model.layers.6.self_attn.q_proj | q_proj | 3.0500 | 2.9374 | 2.9212 | 2.9692 | 2 | 22 | 0.0161 | 1.019082e-06 |
| model.layers.6.self_attn.k_proj | k_proj | 3.0508 | 2.6102 | 2.5692 | 2.6195 | 2 | 19 | 0.0410 | 4.154518e-06 |
| model.layers.6.self_attn.v_proj | v_proj | 3.0515 | 2.6775 | 2.6365 | 2.6827 | 2 | 20 | 0.0410 | 4.184409e-07 |
| model.layers.6.self_attn.o_proj | o_proj | 3.0522 | 2.9103 | 2.8941 | 2.9430 | 2 | 22 | 0.0161 | 9.972693e-09 |
| model.layers.6.mlp.gate_proj | gate_proj | 3.0532 | 2.9463 | 2.9364 | 2.9856 | 2 | 24 | 0.0099 | 4.448496e-07 |
| model.layers.6.mlp.up_proj | up_proj | 3.0563 | 2.9726 | 2.9627 | 3.0247 | 2 | 23 | 0.0099 | 2.348520e-07 |
| model.layers.6.mlp.down_proj | down_proj | 3.0587 | 2.9720 | 2.9618 | 3.0114 | 2 | 24 | 0.0103 | 4.360706e-09 |
| model.layers.7.self_attn.q_proj | q_proj | 3.0614 | 2.9588 | 2.9427 | 2.9898 | 2 | 22 | 0.0161 | 1.135021e-06 |
| model.layers.7.self_attn.k_proj | k_proj | 3.0622 | 2.7296 | 2.6886 | 2.7295 | 2 | 19 | 0.0410 | 3.736920e-06 |
| model.layers.7.self_attn.v_proj | v_proj | 3.0628 | 2.6870 | 2.6460 | 2.6932 | 2 | 22 | 0.0410 | 4.759842e-07 |
| model.layers.7.self_attn.o_proj | o_proj | 3.0636 | 2.9274 | 2.9113 | 2.9596 | 2 | 22 | 0.0161 | 8.643549e-09 |
| model.layers.7.mlp.gate_proj | gate_proj | 3.0646 | 2.9723 | 2.9624 | 3.0113 | 2 | 24 | 0.0099 | 3.854026e-07 |
| model.layers.7.mlp.up_proj | up_proj | 3.0676 | 2.9721 | 2.9622 | 3.0223 | 2 | 22 | 0.0099 | 2.431214e-07 |
| model.layers.7.mlp.down_proj | down_proj | 3.0708 | 2.9837 | 2.9735 | 3.0221 | 2 | 24 | 0.0103 | 4.707600e-09 |
| model.layers.8.self_attn.q_proj | q_proj | 3.0738 | 2.8897 | 2.8735 | 2.9193 | 2 | 22 | 0.0161 | 1.272034e-06 |
| model.layers.8.self_attn.k_proj | k_proj | 3.0754 | 2.7878 | 2.7468 | 2.7852 | 2 | 19 | 0.0410 | 3.738440e-06 |
| model.layers.8.self_attn.v_proj | v_proj | 3.0760 | 2.7301 | 2.6891 | 2.7422 | 2 | 20 | 0.0410 | 4.281687e-07 |
| model.layers.8.self_attn.o_proj | o_proj | 3.0768 | 2.9402 | 2.9241 | 2.9731 | 2 | 22 | 0.0161 | 1.193430e-08 |
| model.layers.8.mlp.gate_proj | gate_proj | 3.0780 | 2.9602 | 2.9503 | 2.9980 | 2 | 24 | 0.0099 | 4.305460e-07 |
| model.layers.8.mlp.up_proj | up_proj | 3.0823 | 2.9947 | 2.9848 | 3.0444 | 2 | 23 | 0.0099 | 2.646573e-07 |
| model.layers.8.mlp.down_proj | down_proj | 3.0856 | 2.9963 | 2.9861 | 3.0355 | 2 | 24 | 0.0103 | 6.135445e-09 |
| model.layers.9.self_attn.q_proj | q_proj | 3.0891 | 2.9577 | 2.9416 | 2.9904 | 2 | 22 | 0.0161 | 1.440376e-06 |
| model.layers.9.self_attn.k_proj | k_proj | 3.0904 | 2.7264 | 2.6854 | 2.7266 | 2 | 19 | 0.0410 | 3.707495e-06 |
| model.layers.9.self_attn.v_proj | v_proj | 3.0913 | 2.7839 | 2.7429 | 2.7969 | 2 | 18 | 0.0410 | 4.621423e-07 |
| model.layers.9.self_attn.o_proj | o_proj | 3.0921 | 2.9516 | 2.9355 | 2.9816 | 2 | 23 | 0.0161 | 1.481566e-08 |
| model.layers.9.mlp.gate_proj | gate_proj | 3.0935 | 3.0120 | 3.0021 | 3.0520 | 2 | 24 | 0.0099 | 4.371699e-07 |
| model.layers.9.mlp.up_proj | up_proj | 3.0970 | 2.9963 | 2.9864 | 3.0450 | 2 | 23 | 0.0099 | 2.815661e-07 |
| model.layers.9.mlp.down_proj | down_proj | 3.1014 | 3.0123 | 3.0021 | 3.0530 | 2 | 24 | 0.0103 | 7.298284e-09 |
| model.layers.10.self_attn.q_proj | q_proj | 3.1055 | 2.9342 | 2.9181 | 2.9634 | 2 | 23 | 0.0161 | 1.430418e-06 |
| model.layers.10.self_attn.k_proj | k_proj | 3.1075 | 2.6789 | 2.6379 | 2.6800 | 2 | 19 | 0.0410 | 4.154626e-06 |
| model.layers.10.self_attn.v_proj | v_proj | 3.1087 | 2.7496 | 2.7086 | 2.7520 | 2 | 19 | 0.0410 | 4.492101e-07 |
| model.layers.10.self_attn.o_proj | o_proj | 3.1098 | 2.9631 | 2.9470 | 2.9947 | 2 | 22 | 0.0161 | 1.052326e-08 |
| model.layers.10.mlp.gate_proj | gate_proj | 3.1115 | 3.0609 | 3.0510 | 3.1061 | 2 | 25 | 0.0099 | 4.816626e-07 |
| model.layers.10.mlp.up_proj | up_proj | 3.1140 | 3.0124 | 3.0025 | 3.0651 | 2 | 23 | 0.0099 | 3.330609e-07 |
| model.layers.10.mlp.down_proj | down_proj | 3.1193 | 3.0304 | 3.0201 | 3.0724 | 2 | 25 | 0.0103 | 9.074627e-09 |
| model.layers.11.self_attn.q_proj | q_proj | 3.1242 | 2.9787 | 2.9626 | 3.0123 | 2 | 23 | 0.0161 | 1.391127e-06 |
| model.layers.11.self_attn.k_proj | k_proj | 3.1263 | 2.7321 | 2.6911 | 2.7293 | 2 | 19 | 0.0410 | 4.155718e-06 |
| model.layers.11.self_attn.v_proj | v_proj | 3.1277 | 2.7608 | 2.7198 | 2.7632 | 2 | 19 | 0.0410 | 4.696447e-07 |
| model.layers.11.self_attn.o_proj | o_proj | 3.1290 | 2.9802 | 2.9640 | 3.0133 | 2 | 22 | 0.0161 | 7.564493e-09 |
| model.layers.11.mlp.gate_proj | gate_proj | 3.1311 | 3.0440 | 3.0341 | 3.0887 | 2 | 24 | 0.0099 | 5.442510e-07 |
| model.layers.11.mlp.up_proj | up_proj | 3.1364 | 3.0488 | 3.0389 | 3.0994 | 2 | 22 | 0.0099 | 3.574474e-07 |
| model.layers.11.mlp.down_proj | down_proj | 3.1420 | 3.0528 | 3.0426 | 3.0976 | 2 | 24 | 0.0103 | 1.006008e-08 |
| model.layers.12.self_attn.q_proj | q_proj | 3.1482 | 3.0103 | 2.9942 | 3.0486 | 2 | 22 | 0.0161 | 1.133556e-06 |
| model.layers.12.self_attn.k_proj | k_proj | 3.1506 | 2.7388 | 2.6978 | 2.7434 | 2 | 20 | 0.0410 | 3.283541e-06 |
| model.layers.12.self_attn.v_proj | v_proj | 3.1524 | 2.8064 | 2.7654 | 2.8070 | 2 | 19 | 0.0410 | 4.356336e-07 |
| model.layers.12.self_attn.o_proj | o_proj | 3.1539 | 3.0087 | 2.9926 | 3.0458 | 2 | 22 | 0.0161 | 7.743120e-09 |
| model.layers.12.mlp.gate_proj | gate_proj | 3.1565 | 3.0731 | 3.0632 | 3.1191 | 2 | 23 | 0.0099 | 5.194373e-07 |
| model.layers.12.mlp.up_proj | up_proj | 3.1630 | 3.0721 | 3.0622 | 3.1182 | 2 | 21 | 0.0099 | 3.699589e-07 |
| model.layers.12.mlp.down_proj | down_proj | 3.1707 | 3.0825 | 3.0722 | 3.1266 | 2 | 24 | 0.0103 | 1.196422e-08 |
| model.layers.13.self_attn.q_proj | q_proj | 3.1788 | 3.0542 | 3.0381 | 3.0968 | 2 | 23 | 0.0161 | 1.258372e-06 |
| model.layers.13.self_attn.k_proj | k_proj | 3.1817 | 2.7574 | 2.7164 | 2.7541 | 2 | 20 | 0.0410 | 3.407768e-06 |
| model.layers.13.self_attn.v_proj | v_proj | 3.1842 | 2.8279 | 2.7869 | 2.8304 | 2 | 19 | 0.0410 | 6.970832e-07 |
| model.layers.13.self_attn.o_proj | o_proj | 3.1864 | 3.0452 | 3.0291 | 3.0841 | 2 | 22 | 0.0161 | 9.428409e-09 |
| model.layers.13.mlp.gate_proj | gate_proj | 3.1898 | 3.1117 | 3.1018 | 3.1504 | 2 | 23 | 0.0099 | 5.208803e-07 |
| model.layers.13.mlp.up_proj | up_proj | 3.1982 | 3.1058 | 3.0959 | 3.1460 | 2 | 22 | 0.0099 | 4.141571e-07 |
| model.layers.13.mlp.down_proj | down_proj | 3.2094 | 3.1212 | 3.1110 | 3.1577 | 2 | 23 | 0.0103 | 1.597459e-08 |
| model.layers.14.self_attn.q_proj | q_proj | 3.2216 | 3.0915 | 3.0753 | 3.1379 | 2 | 22 | 0.0161 | 9.774840e-07 |
| model.layers.14.self_attn.k_proj | k_proj | 3.2263 | 2.8561 | 2.8151 | 2.8584 | 2 | 20 | 0.0410 | 1.878367e-06 |
| model.layers.14.self_attn.v_proj | v_proj | 3.2296 | 2.8347 | 2.7937 | 2.8272 | 2 | 20 | 0.0410 | 1.022013e-06 |
| model.layers.14.self_attn.o_proj | o_proj | 3.2332 | 3.0869 | 3.0708 | 3.1223 | 2 | 21 | 0.0161 | 1.845814e-08 |
| model.layers.14.mlp.gate_proj | gate_proj | 3.2387 | 3.1380 | 3.1281 | 3.1737 | 2 | 24 | 0.0099 | 6.020632e-07 |
| model.layers.14.mlp.up_proj | up_proj | 3.2566 | 3.1745 | 3.1646 | 3.2051 | 2 | 22 | 0.0099 | 4.257617e-07 |
| model.layers.14.mlp.down_proj | down_proj | 3.2744 | 3.1830 | 3.1728 | 3.2112 | 2 | 23 | 0.0103 | 1.860104e-08 |
| model.layers.15.self_attn.q_proj | q_proj | 3.2996 | 3.1400 | 3.1239 | 3.1842 | 2 | 22 | 0.0161 | 8.882757e-07 |
| model.layers.15.self_attn.k_proj | k_proj | 3.3114 | 2.9227 | 2.8817 | 2.9170 | 2 | 20 | 0.0410 | 1.970817e-06 |
| model.layers.15.self_attn.v_proj | v_proj | 3.3187 | 3.0031 | 2.9620 | 3.0046 | 2 | 20 | 0.0410 | 8.757371e-07 |
| model.layers.15.self_attn.o_proj | o_proj | 3.3248 | 3.1726 | 3.1565 | 3.2173 | 2 | 22 | 0.0161 | 3.190353e-08 |
| model.layers.15.mlp.gate_proj | gate_proj | 3.3375 | 3.2486 | 3.2387 | 3.2745 | 2 | 24 | 0.0099 | 5.408669e-07 |
| model.layers.15.mlp.up_proj | up_proj | 3.3819 | 3.2963 | 3.2864 | 3.3197 | 2 | 23 | 0.0099 | 4.056213e-07 |
| model.layers.15.mlp.down_proj | down_proj | 3.4676 | 3.3737 | 3.3635 | 3.3964 | 2 | 25 | 0.0103 | 2.791509e-08 |
