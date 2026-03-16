# WaterSIC Run Report

- Timestamp: `2026-03-13T06:33:43Z`
- Git commit: `34b86fc9bfd78fc9a75752a9039cd829ec220986`
- Environment: `watersic`
- Device: `cuda`
- Model: `meta-llama/Llama-3.2-1B`
- Model revision: `main`
- Tokenizer: `meta-llama/Llama-3.2-1B`
- Sequence length: `2048`
- Calibration sequences: `8`
- Target global bitwidth: `3.0000`
- Achieved global bitwidth: `2.9984`
- Raw average bitwidth: `8.4774`
- Entropy average bitwidth: `2.9865`
- Huffman average bitwidth: `3.0368`
- Huffman shortest symbol length: `n/a`
- Huffman longest symbol length: `n/a`
- Side-information overhead: `0.0119`
- Perplexity: `16.868369287133703`

## Layer Summary

| Layer | Kind | Target | Achieved | Entropy | Huffman | Huff Min | Huff Max | Side Info | Weighted Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model.layers.0.self_attn.q_proj | q_proj | 3.0000 | 2.8879 | 2.8717 | 2.9296 | n/a | n/a | 0.0161 | 3.011346e-07 |
| model.layers.0.self_attn.k_proj | k_proj | 3.0005 | 2.7010 | 2.6600 | 2.7259 | n/a | n/a | 0.0410 | 7.000186e-07 |
| model.layers.0.self_attn.v_proj | v_proj | 3.0008 | 2.6206 | 2.5795 | 2.6434 | n/a | n/a | 0.0410 | 2.583929e-08 |
| model.layers.0.self_attn.o_proj | o_proj | 3.0012 | 2.8675 | 2.8514 | 2.8805 | n/a | n/a | 0.0161 | 2.514840e-10 |
| model.layers.0.mlp.gate_proj | gate_proj | 3.0018 | 2.9115 | 2.9016 | 2.9541 | n/a | n/a | 0.0099 | 1.312714e-07 |
| model.layers.0.mlp.up_proj | up_proj | 3.0034 | 2.9167 | 2.9068 | 2.9622 | n/a | n/a | 0.0099 | 1.029593e-07 |
| model.layers.0.mlp.down_proj | down_proj | 3.0050 | 2.9143 | 2.9040 | 2.9444 | n/a | n/a | 0.0103 | 4.443879e-10 |
| model.layers.1.self_attn.q_proj | q_proj | 3.0066 | 2.8577 | 2.8415 | 2.8823 | n/a | n/a | 0.0162 | 6.576188e-07 |
| model.layers.1.self_attn.k_proj | k_proj | 3.0073 | 2.5635 | 2.5224 | 2.5926 | n/a | n/a | 0.0411 | 2.315885e-06 |
| model.layers.1.self_attn.v_proj | v_proj | 3.0078 | 2.6122 | 2.5712 | 2.6221 | n/a | n/a | 0.0411 | 1.192544e-07 |
| model.layers.1.self_attn.o_proj | o_proj | 3.0083 | 2.8821 | 2.8660 | 2.9036 | n/a | n/a | 0.0161 | 1.207776e-09 |
| model.layers.1.mlp.gate_proj | gate_proj | 3.0089 | 2.9200 | 2.9101 | 2.9628 | n/a | n/a | 0.0099 | 2.038901e-07 |
| model.layers.1.mlp.up_proj | up_proj | 3.0106 | 2.9220 | 2.9121 | 2.9686 | n/a | n/a | 0.0099 | 1.462905e-07 |
| model.layers.1.mlp.down_proj | down_proj | 3.0123 | 2.9121 | 2.9018 | 2.9497 | n/a | n/a | 0.0103 | 8.101866e-10 |
| model.layers.2.self_attn.q_proj | q_proj | 3.0143 | 2.9219 | 2.9058 | 2.9522 | n/a | n/a | 0.0161 | 1.109000e-06 |
| model.layers.2.self_attn.k_proj | k_proj | 3.0147 | 2.6352 | 2.5942 | 2.6425 | n/a | n/a | 0.0410 | 3.596489e-06 |
| model.layers.2.self_attn.v_proj | v_proj | 3.0152 | 2.6499 | 2.6089 | 2.6535 | n/a | n/a | 0.0410 | 2.998593e-07 |
| model.layers.2.self_attn.o_proj | o_proj | 3.0156 | 2.8926 | 2.8764 | 2.9256 | n/a | n/a | 0.0161 | 1.843663e-09 |
| model.layers.2.mlp.gate_proj | gate_proj | 3.0163 | 2.9315 | 2.9216 | 2.9738 | n/a | n/a | 0.0099 | 3.086611e-07 |
| model.layers.2.mlp.up_proj | up_proj | 3.0180 | 2.9280 | 2.9181 | 2.9758 | n/a | n/a | 0.0099 | 1.914512e-07 |
| model.layers.2.mlp.down_proj | down_proj | 3.0198 | 2.9306 | 2.9204 | 2.9689 | n/a | n/a | 0.0103 | 1.920987e-09 |
| model.layers.3.self_attn.q_proj | q_proj | 3.0217 | 2.8392 | 2.8231 | 2.8620 | n/a | n/a | 0.0161 | 1.108217e-06 |
| model.layers.3.self_attn.k_proj | k_proj | 3.0227 | 2.7182 | 2.6772 | 2.7161 | n/a | n/a | 0.0410 | 2.620905e-06 |
| model.layers.3.self_attn.v_proj | v_proj | 3.0231 | 2.7129 | 2.6719 | 2.7218 | n/a | n/a | 0.0410 | 3.351893e-07 |
| model.layers.3.self_attn.o_proj | o_proj | 3.0235 | 2.8939 | 2.8778 | 2.9289 | n/a | n/a | 0.0161 | 4.012327e-09 |
| model.layers.3.mlp.gate_proj | gate_proj | 3.0242 | 2.9250 | 2.9151 | 2.9648 | n/a | n/a | 0.0099 | 4.403488e-07 |
| model.layers.3.mlp.up_proj | up_proj | 3.0264 | 2.9378 | 2.9279 | 2.9865 | n/a | n/a | 0.0099 | 2.144949e-07 |
| model.layers.3.mlp.down_proj | down_proj | 3.0284 | 2.9389 | 2.9287 | 2.9773 | n/a | n/a | 0.0103 | 2.796691e-09 |
| model.layers.4.self_attn.q_proj | q_proj | 3.0305 | 2.8949 | 2.8788 | 2.9193 | n/a | n/a | 0.0161 | 9.185922e-07 |
| model.layers.4.self_attn.k_proj | k_proj | 3.0312 | 2.6762 | 2.6352 | 2.6759 | n/a | n/a | 0.0410 | 2.651332e-06 |
| model.layers.4.self_attn.v_proj | v_proj | 3.0318 | 2.7291 | 2.6881 | 2.7390 | n/a | n/a | 0.0410 | 2.919724e-07 |
| model.layers.4.self_attn.o_proj | o_proj | 3.0322 | 2.8999 | 2.8838 | 2.9252 | n/a | n/a | 0.0161 | 5.389872e-09 |
| model.layers.4.mlp.gate_proj | gate_proj | 3.0330 | 2.9494 | 2.9395 | 2.9898 | n/a | n/a | 0.0099 | 4.421291e-07 |
| model.layers.4.mlp.up_proj | up_proj | 3.0350 | 2.9356 | 2.9257 | 2.9831 | n/a | n/a | 0.0099 | 2.037647e-07 |
| model.layers.4.mlp.down_proj | down_proj | 3.0374 | 2.9474 | 2.9372 | 2.9865 | n/a | n/a | 0.0103 | 3.065866e-09 |
| model.layers.5.self_attn.q_proj | q_proj | 3.0396 | 2.9410 | 2.9249 | 2.9714 | n/a | n/a | 0.0161 | 9.389036e-07 |
| model.layers.5.self_attn.k_proj | k_proj | 3.0403 | 2.6073 | 2.5663 | 2.6205 | n/a | n/a | 0.0410 | 3.545121e-06 |
| model.layers.5.self_attn.v_proj | v_proj | 3.0410 | 2.6599 | 2.6189 | 2.6632 | n/a | n/a | 0.0410 | 2.690647e-07 |
| model.layers.5.self_attn.o_proj | o_proj | 3.0416 | 2.8976 | 2.8814 | 2.9298 | n/a | n/a | 0.0161 | 5.934213e-09 |
| model.layers.5.mlp.gate_proj | gate_proj | 3.0425 | 2.9594 | 2.9495 | 2.9992 | n/a | n/a | 0.0099 | 3.915548e-07 |
| model.layers.5.mlp.up_proj | up_proj | 3.0446 | 2.9560 | 2.9461 | 3.0038 | n/a | n/a | 0.0099 | 2.086343e-07 |
| model.layers.5.mlp.down_proj | down_proj | 3.0470 | 2.9603 | 2.9501 | 2.9987 | n/a | n/a | 0.0103 | 3.392134e-09 |
| model.layers.6.self_attn.q_proj | q_proj | 3.0494 | 2.9367 | 2.9206 | 2.9662 | n/a | n/a | 0.0161 | 8.659262e-07 |
| model.layers.6.self_attn.k_proj | k_proj | 3.0502 | 2.6106 | 2.5696 | 2.6214 | n/a | n/a | 0.0410 | 3.558998e-06 |
| model.layers.6.self_attn.v_proj | v_proj | 3.0510 | 2.6791 | 2.6381 | 2.6864 | n/a | n/a | 0.0410 | 3.547852e-07 |
| model.layers.6.self_attn.o_proj | o_proj | 3.0516 | 2.9102 | 2.8941 | 2.9421 | n/a | n/a | 0.0161 | 8.579566e-09 |
| model.layers.6.mlp.gate_proj | gate_proj | 3.0526 | 2.9463 | 2.9364 | 2.9850 | n/a | n/a | 0.0099 | 3.908357e-07 |
| model.layers.6.mlp.up_proj | up_proj | 3.0557 | 2.9715 | 2.9616 | 3.0202 | n/a | n/a | 0.0099 | 2.072188e-07 |
| model.layers.6.mlp.down_proj | down_proj | 3.0582 | 2.9708 | 2.9606 | 3.0095 | n/a | n/a | 0.0103 | 3.527942e-09 |
| model.layers.7.self_attn.q_proj | q_proj | 3.0609 | 2.9549 | 2.9387 | 2.9842 | n/a | n/a | 0.0161 | 9.819149e-07 |
| model.layers.7.self_attn.k_proj | k_proj | 3.0617 | 2.7324 | 2.6914 | 2.7312 | n/a | n/a | 0.0410 | 3.227210e-06 |
| model.layers.7.self_attn.v_proj | v_proj | 3.0623 | 2.6856 | 2.6446 | 2.6928 | n/a | n/a | 0.0410 | 4.112304e-07 |
| model.layers.7.self_attn.o_proj | o_proj | 3.0630 | 2.9233 | 2.9072 | 2.9519 | n/a | n/a | 0.0161 | 7.277311e-09 |
| model.layers.7.mlp.gate_proj | gate_proj | 3.0641 | 2.9719 | 2.9620 | 3.0107 | n/a | n/a | 0.0099 | 3.377974e-07 |
| model.layers.7.mlp.up_proj | up_proj | 3.0671 | 2.9718 | 2.9619 | 3.0188 | n/a | n/a | 0.0099 | 2.139913e-07 |
| model.layers.7.mlp.down_proj | down_proj | 3.0703 | 2.9813 | 2.9710 | 3.0201 | n/a | n/a | 0.0103 | 3.796196e-09 |
| model.layers.8.self_attn.q_proj | q_proj | 3.0734 | 2.8911 | 2.8750 | 2.9191 | n/a | n/a | 0.0161 | 1.074458e-06 |
| model.layers.8.self_attn.k_proj | k_proj | 3.0749 | 2.7884 | 2.7474 | 2.7846 | n/a | n/a | 0.0410 | 3.192160e-06 |
| model.layers.8.self_attn.v_proj | v_proj | 3.0756 | 2.7291 | 2.6881 | 2.7395 | n/a | n/a | 0.0410 | 3.625288e-07 |
| model.layers.8.self_attn.o_proj | o_proj | 3.0763 | 2.9413 | 2.9252 | 2.9738 | n/a | n/a | 0.0161 | 9.900574e-09 |
| model.layers.8.mlp.gate_proj | gate_proj | 3.0775 | 2.9607 | 2.9508 | 2.9982 | n/a | n/a | 0.0099 | 3.773685e-07 |
| model.layers.8.mlp.up_proj | up_proj | 3.0818 | 2.9944 | 2.9845 | 3.0412 | n/a | n/a | 0.0099 | 2.327155e-07 |
| model.layers.8.mlp.down_proj | down_proj | 3.0851 | 2.9946 | 2.9844 | 3.0345 | n/a | n/a | 0.0103 | 4.763609e-09 |
| model.layers.9.self_attn.q_proj | q_proj | 3.0887 | 2.9576 | 2.9415 | 2.9893 | n/a | n/a | 0.0161 | 1.197018e-06 |
| model.layers.9.self_attn.k_proj | k_proj | 3.0900 | 2.7289 | 2.6878 | 2.7275 | n/a | n/a | 0.0410 | 3.094182e-06 |
| model.layers.9.self_attn.v_proj | v_proj | 3.0909 | 2.7844 | 2.7434 | 2.7925 | n/a | n/a | 0.0410 | 3.845873e-07 |
| model.layers.9.self_attn.o_proj | o_proj | 3.0916 | 2.9514 | 2.9353 | 2.9814 | n/a | n/a | 0.0161 | 1.202128e-08 |
| model.layers.9.mlp.gate_proj | gate_proj | 3.0930 | 3.0110 | 3.0011 | 3.0508 | n/a | n/a | 0.0099 | 3.756581e-07 |
| model.layers.9.mlp.up_proj | up_proj | 3.0965 | 2.9961 | 2.9862 | 3.0419 | n/a | n/a | 0.0099 | 2.425715e-07 |
| model.layers.9.mlp.down_proj | down_proj | 3.1009 | 3.0118 | 3.0016 | 3.0530 | n/a | n/a | 0.0103 | 5.445611e-09 |
| model.layers.10.self_attn.q_proj | q_proj | 3.1050 | 2.9340 | 2.9178 | 2.9632 | n/a | n/a | 0.0161 | 1.130090e-06 |
| model.layers.10.self_attn.k_proj | k_proj | 3.1070 | 2.6792 | 2.6382 | 2.6803 | n/a | n/a | 0.0410 | 3.324255e-06 |
| model.layers.10.self_attn.v_proj | v_proj | 3.1082 | 2.7515 | 2.7105 | 2.7512 | n/a | n/a | 0.0410 | 3.554815e-07 |
| model.layers.10.self_attn.o_proj | o_proj | 3.1093 | 2.9623 | 2.9462 | 2.9934 | n/a | n/a | 0.0161 | 9.213727e-09 |
| model.layers.10.mlp.gate_proj | gate_proj | 3.1110 | 3.0606 | 3.0507 | 3.1052 | n/a | n/a | 0.0099 | 4.037451e-07 |
| model.layers.10.mlp.up_proj | up_proj | 3.1135 | 3.0116 | 3.0017 | 3.0599 | n/a | n/a | 0.0099 | 2.808825e-07 |
| model.layers.10.mlp.down_proj | down_proj | 3.1189 | 3.0271 | 3.0168 | 3.0683 | n/a | n/a | 0.0103 | 6.562347e-09 |
| model.layers.11.self_attn.q_proj | q_proj | 3.1239 | 2.9797 | 2.9636 | 3.0145 | n/a | n/a | 0.0161 | 1.092294e-06 |
| model.layers.11.self_attn.k_proj | k_proj | 3.1260 | 2.7321 | 2.6911 | 2.7289 | n/a | n/a | 0.0410 | 3.289829e-06 |
| model.layers.11.self_attn.v_proj | v_proj | 3.1273 | 2.7598 | 2.7188 | 2.7586 | n/a | n/a | 0.0410 | 3.698118e-07 |
| model.layers.11.self_attn.o_proj | o_proj | 3.1286 | 2.9809 | 2.9648 | 3.0129 | n/a | n/a | 0.0161 | 6.252699e-09 |
| model.layers.11.mlp.gate_proj | gate_proj | 3.1307 | 3.0439 | 3.0340 | 3.0874 | n/a | n/a | 0.0099 | 4.479407e-07 |
| model.layers.11.mlp.up_proj | up_proj | 3.1360 | 3.0482 | 3.0383 | 3.0981 | n/a | n/a | 0.0099 | 2.960455e-07 |
| model.layers.11.mlp.down_proj | down_proj | 3.1417 | 3.0495 | 3.0392 | 3.0927 | n/a | n/a | 0.0103 | 6.965081e-09 |
| model.layers.12.self_attn.q_proj | q_proj | 3.1480 | 3.0100 | 2.9939 | 3.0462 | n/a | n/a | 0.0161 | 8.746992e-07 |
| model.layers.12.self_attn.k_proj | k_proj | 3.1504 | 2.7406 | 2.6996 | 2.7459 | n/a | n/a | 0.0410 | 2.543679e-06 |
| model.layers.12.self_attn.v_proj | v_proj | 3.1523 | 2.8048 | 2.7638 | 2.8027 | n/a | n/a | 0.0410 | 3.368760e-07 |
| model.layers.12.self_attn.o_proj | o_proj | 3.1538 | 3.0062 | 2.9901 | 3.0428 | n/a | n/a | 0.0161 | 6.465003e-09 |
| model.layers.12.mlp.gate_proj | gate_proj | 3.1565 | 3.0731 | 3.0632 | 3.1200 | n/a | n/a | 0.0099 | 4.181217e-07 |
| model.layers.12.mlp.up_proj | up_proj | 3.1629 | 3.0721 | 3.0622 | 3.1169 | n/a | n/a | 0.0099 | 2.990706e-07 |
| model.layers.12.mlp.down_proj | down_proj | 3.1706 | 3.0821 | 3.0719 | 3.1289 | n/a | n/a | 0.0103 | 7.873819e-09 |
| model.layers.13.self_attn.q_proj | q_proj | 3.1787 | 3.0542 | 3.0381 | 3.0955 | n/a | n/a | 0.0161 | 9.528300e-07 |
| model.layers.13.self_attn.k_proj | k_proj | 3.1816 | 2.7571 | 2.7161 | 2.7535 | n/a | n/a | 0.0410 | 2.597787e-06 |
| model.layers.13.self_attn.v_proj | v_proj | 3.1842 | 2.8266 | 2.7855 | 2.8258 | n/a | n/a | 0.0410 | 5.276737e-07 |
| model.layers.13.self_attn.o_proj | o_proj | 3.1863 | 3.0421 | 3.0260 | 3.0809 | n/a | n/a | 0.0161 | 7.243255e-09 |
| model.layers.13.mlp.gate_proj | gate_proj | 3.1898 | 3.1116 | 3.1018 | 3.1513 | n/a | n/a | 0.0099 | 4.155321e-07 |
| model.layers.13.mlp.up_proj | up_proj | 3.1982 | 3.1063 | 3.0964 | 3.1449 | n/a | n/a | 0.0099 | 3.313557e-07 |
| model.layers.13.mlp.down_proj | down_proj | 3.2094 | 3.1213 | 3.1111 | 3.1621 | n/a | n/a | 0.0103 | 1.024294e-08 |
| model.layers.14.self_attn.q_proj | q_proj | 3.2215 | 3.0913 | 3.0752 | 3.1382 | n/a | n/a | 0.0161 | 7.496542e-07 |
| model.layers.14.self_attn.k_proj | k_proj | 3.2262 | 2.8550 | 2.8140 | 2.8591 | n/a | n/a | 0.0410 | 1.457472e-06 |
| model.layers.14.self_attn.v_proj | v_proj | 3.2295 | 2.8361 | 2.7950 | 2.8274 | n/a | n/a | 0.0410 | 7.859366e-07 |
| model.layers.14.self_attn.o_proj | o_proj | 3.2331 | 3.0870 | 3.0709 | 3.1231 | n/a | n/a | 0.0161 | 1.291829e-08 |
| model.layers.14.mlp.gate_proj | gate_proj | 3.2386 | 3.1375 | 3.1276 | 3.1745 | n/a | n/a | 0.0099 | 4.742878e-07 |
| model.layers.14.mlp.up_proj | up_proj | 3.2566 | 3.1742 | 3.1643 | 3.2032 | n/a | n/a | 0.0099 | 3.366065e-07 |
| model.layers.14.mlp.down_proj | down_proj | 3.2744 | 3.1858 | 3.1755 | 3.2174 | n/a | n/a | 0.0103 | 1.158448e-08 |
| model.layers.15.self_attn.q_proj | q_proj | 3.2989 | 3.1401 | 3.1240 | 3.1865 | n/a | n/a | 0.0161 | 6.689506e-07 |
| model.layers.15.self_attn.k_proj | k_proj | 3.3106 | 2.9223 | 2.8812 | 2.9180 | n/a | n/a | 0.0410 | 1.492906e-06 |
| model.layers.15.self_attn.v_proj | v_proj | 3.3180 | 3.0024 | 2.9614 | 3.0019 | n/a | n/a | 0.0410 | 6.636769e-07 |
| model.layers.15.self_attn.o_proj | o_proj | 3.3240 | 3.1747 | 3.1586 | 3.2181 | n/a | n/a | 0.0161 | 2.363178e-08 |
| model.layers.15.mlp.gate_proj | gate_proj | 3.3365 | 3.2477 | 3.2378 | 3.2750 | n/a | n/a | 0.0099 | 4.231003e-07 |
| model.layers.15.mlp.up_proj | up_proj | 3.3809 | 3.2946 | 3.2848 | 3.3162 | n/a | n/a | 0.0099 | 3.191250e-07 |
| model.layers.15.mlp.down_proj | down_proj | 3.4671 | 3.3748 | 3.3645 | 3.4032 | n/a | n/a | 0.0103 | 1.747499e-08 |

## Notes

- Exact Huffman shortest/longest symbol lengths are unavailable for this historical run because the integer Huffman symbols were not serialized in the saved artifact.
