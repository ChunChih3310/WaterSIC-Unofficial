# WaterSIC Run Report

- Timestamp: `2026-03-15T05:39:42Z`
- Git commit: `6173f2197521e0395fe99e0848569f39bf02fc4a`
- Environment: `watersic`
- Device: `cuda`
- Model: `meta-llama/Llama-3.2-1B`
- Model revision: `main`
- Tokenizer: `meta-llama/Llama-3.2-1B`
- Sequence length: `2048`
- Calibration sequences: `16`
- Target global bitwidth: `3.0000`
- Achieved global bitwidth: `2.9984`
- Raw average bitwidth: `8.4203`
- Entropy average bitwidth: `2.9865`
- Huffman average bitwidth: `3.0371`
- Huffman shortest symbol length: `n/a`
- Huffman longest symbol length: `n/a`
- Side-information overhead: `0.0119`
- Perplexity: `12.457374978830424`

## Layer Summary

| Layer | Kind | Target | Achieved | Entropy | Huffman | Huff Min | Huff Max | Side Info | Weighted Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model.layers.0.self_attn.q_proj | q_proj | 3.0000 | 2.8879 | 2.8718 | 2.9306 | n/a | n/a | 0.0161 | 3.479643e-07 |
| model.layers.0.self_attn.k_proj | k_proj | 3.0005 | 2.7047 | 2.6636 | 2.7302 | n/a | n/a | 0.0410 | 7.999128e-07 |
| model.layers.0.self_attn.v_proj | v_proj | 3.0008 | 2.6206 | 2.5795 | 2.6453 | n/a | n/a | 0.0410 | 2.949648e-08 |
| model.layers.0.self_attn.o_proj | o_proj | 3.0012 | 2.8654 | 2.8492 | 2.8823 | n/a | n/a | 0.0161 | 2.766085e-10 |
| model.layers.0.mlp.gate_proj | gate_proj | 3.0018 | 2.9110 | 2.9011 | 2.9563 | n/a | n/a | 0.0099 | 1.460601e-07 |
| model.layers.0.mlp.up_proj | up_proj | 3.0034 | 2.9165 | 2.9065 | 2.9634 | n/a | n/a | 0.0099 | 1.144743e-07 |
| model.layers.0.mlp.down_proj | down_proj | 3.0050 | 2.9144 | 2.9042 | 2.9523 | n/a | n/a | 0.0103 | 5.960778e-10 |
| model.layers.1.self_attn.q_proj | q_proj | 3.0067 | 2.8593 | 2.8431 | 2.8838 | n/a | n/a | 0.0162 | 7.155615e-07 |
| model.layers.1.self_attn.k_proj | k_proj | 3.0073 | 2.5627 | 2.5216 | 2.5907 | n/a | n/a | 0.0411 | 2.520627e-06 |
| model.layers.1.self_attn.v_proj | v_proj | 3.0079 | 2.6131 | 2.5720 | 2.6228 | n/a | n/a | 0.0411 | 1.297195e-07 |
| model.layers.1.self_attn.o_proj | o_proj | 3.0083 | 2.8820 | 2.8659 | 2.9031 | n/a | n/a | 0.0161 | 1.319506e-09 |
| model.layers.1.mlp.gate_proj | gate_proj | 3.0089 | 2.9204 | 2.9105 | 2.9650 | n/a | n/a | 0.0099 | 2.211586e-07 |
| model.layers.1.mlp.up_proj | up_proj | 3.0106 | 2.9221 | 2.9122 | 2.9695 | n/a | n/a | 0.0099 | 1.586819e-07 |
| model.layers.1.mlp.down_proj | down_proj | 3.0123 | 2.9056 | 2.8953 | 2.9428 | n/a | n/a | 0.0103 | 1.125255e-09 |
| model.layers.2.self_attn.q_proj | q_proj | 3.0144 | 2.9226 | 2.9065 | 2.9531 | n/a | n/a | 0.0161 | 1.195766e-06 |
| model.layers.2.self_attn.k_proj | k_proj | 3.0148 | 2.6368 | 2.5958 | 2.6433 | n/a | n/a | 0.0410 | 3.861227e-06 |
| model.layers.2.self_attn.v_proj | v_proj | 3.0153 | 2.6498 | 2.6088 | 2.6522 | n/a | n/a | 0.0410 | 3.210189e-07 |
| model.layers.2.self_attn.o_proj | o_proj | 3.0158 | 2.8865 | 2.8704 | 2.9181 | n/a | n/a | 0.0161 | 2.024736e-09 |
| model.layers.2.mlp.gate_proj | gate_proj | 3.0164 | 2.9320 | 2.9221 | 2.9755 | n/a | n/a | 0.0099 | 3.322867e-07 |
| model.layers.2.mlp.up_proj | up_proj | 3.0181 | 2.9280 | 2.9181 | 2.9767 | n/a | n/a | 0.0099 | 2.058679e-07 |
| model.layers.2.mlp.down_proj | down_proj | 3.0200 | 2.9274 | 2.9171 | 2.9653 | n/a | n/a | 0.0103 | 2.317566e-09 |
| model.layers.3.self_attn.q_proj | q_proj | 3.0220 | 2.8396 | 2.8235 | 2.8635 | n/a | n/a | 0.0161 | 1.185313e-06 |
| model.layers.3.self_attn.k_proj | k_proj | 3.0229 | 2.7183 | 2.6773 | 2.7166 | n/a | n/a | 0.0410 | 2.804451e-06 |
| model.layers.3.self_attn.v_proj | v_proj | 3.0233 | 2.7154 | 2.6743 | 2.7249 | n/a | n/a | 0.0410 | 3.584068e-07 |
| model.layers.3.self_attn.o_proj | o_proj | 3.0238 | 2.8922 | 2.8761 | 2.9270 | n/a | n/a | 0.0161 | 4.415568e-09 |
| model.layers.3.mlp.gate_proj | gate_proj | 3.0245 | 2.9256 | 2.9157 | 2.9658 | n/a | n/a | 0.0099 | 4.742663e-07 |
| model.layers.3.mlp.up_proj | up_proj | 3.0266 | 2.9376 | 2.9277 | 2.9869 | n/a | n/a | 0.0099 | 2.307784e-07 |
| model.layers.3.mlp.down_proj | down_proj | 3.0286 | 2.9377 | 2.9274 | 2.9756 | n/a | n/a | 0.0103 | 3.294755e-09 |
| model.layers.4.self_attn.q_proj | q_proj | 3.0307 | 2.8966 | 2.8805 | 2.9212 | n/a | n/a | 0.0161 | 9.856684e-07 |
| model.layers.4.self_attn.k_proj | k_proj | 3.0315 | 2.6766 | 2.6356 | 2.6770 | n/a | n/a | 0.0410 | 2.841895e-06 |
| model.layers.4.self_attn.v_proj | v_proj | 3.0320 | 2.7301 | 2.6890 | 2.7399 | n/a | n/a | 0.0410 | 3.142281e-07 |
| model.layers.4.self_attn.o_proj | o_proj | 3.0325 | 2.8993 | 2.8832 | 2.9247 | n/a | n/a | 0.0161 | 5.812144e-09 |
| model.layers.4.mlp.gate_proj | gate_proj | 3.0332 | 2.9490 | 2.9391 | 2.9899 | n/a | n/a | 0.0099 | 4.785014e-07 |
| model.layers.4.mlp.up_proj | up_proj | 3.0352 | 2.9359 | 2.9260 | 2.9842 | n/a | n/a | 0.0099 | 2.200904e-07 |
| model.layers.4.mlp.down_proj | down_proj | 3.0377 | 2.9486 | 2.9383 | 2.9864 | n/a | n/a | 0.0103 | 3.552521e-09 |
| model.layers.5.self_attn.q_proj | q_proj | 3.0399 | 2.9405 | 2.9244 | 2.9713 | n/a | n/a | 0.0161 | 1.015339e-06 |
| model.layers.5.self_attn.k_proj | k_proj | 3.0405 | 2.6059 | 2.5649 | 2.6185 | n/a | n/a | 0.0410 | 3.823719e-06 |
| model.layers.5.self_attn.v_proj | v_proj | 3.0412 | 2.6591 | 2.6181 | 2.6608 | n/a | n/a | 0.0410 | 2.902996e-07 |
| model.layers.5.self_attn.o_proj | o_proj | 3.0418 | 2.8974 | 2.8813 | 2.9297 | n/a | n/a | 0.0161 | 6.251301e-09 |
| model.layers.5.mlp.gate_proj | gate_proj | 3.0427 | 2.9592 | 2.9493 | 2.9997 | n/a | n/a | 0.0099 | 4.241032e-07 |
| model.layers.5.mlp.up_proj | up_proj | 3.0449 | 2.9569 | 2.9470 | 3.0067 | n/a | n/a | 0.0099 | 2.254398e-07 |
| model.layers.5.mlp.down_proj | down_proj | 3.0473 | 2.9623 | 2.9520 | 2.9998 | n/a | n/a | 0.0103 | 3.906443e-09 |
| model.layers.6.self_attn.q_proj | q_proj | 3.0496 | 2.9369 | 2.9208 | 2.9673 | n/a | n/a | 0.0161 | 9.350982e-07 |
| model.layers.6.self_attn.k_proj | k_proj | 3.0504 | 2.6094 | 2.5683 | 2.6190 | n/a | n/a | 0.0410 | 3.838576e-06 |
| model.layers.6.self_attn.v_proj | v_proj | 3.0512 | 2.6787 | 2.6377 | 2.6850 | n/a | n/a | 0.0410 | 3.838064e-07 |
| model.layers.6.self_attn.o_proj | o_proj | 3.0518 | 2.9114 | 2.8952 | 2.9436 | n/a | n/a | 0.0161 | 9.121450e-09 |
| model.layers.6.mlp.gate_proj | gate_proj | 3.0528 | 2.9458 | 2.9359 | 2.9848 | n/a | n/a | 0.0099 | 4.235385e-07 |
| model.layers.6.mlp.up_proj | up_proj | 3.0559 | 2.9719 | 2.9620 | 3.0224 | n/a | n/a | 0.0099 | 2.242277e-07 |
| model.layers.6.mlp.down_proj | down_proj | 3.0584 | 2.9711 | 2.9609 | 3.0095 | n/a | n/a | 0.0103 | 3.958499e-09 |
| model.layers.7.self_attn.q_proj | q_proj | 3.0611 | 2.9570 | 2.9409 | 2.9871 | n/a | n/a | 0.0161 | 1.048615e-06 |
| model.layers.7.self_attn.k_proj | k_proj | 3.0619 | 2.7316 | 2.6906 | 2.7311 | n/a | n/a | 0.0410 | 3.452558e-06 |
| model.layers.7.self_attn.v_proj | v_proj | 3.0625 | 2.6876 | 2.6466 | 2.6938 | n/a | n/a | 0.0410 | 4.391008e-07 |
| model.layers.7.self_attn.o_proj | o_proj | 3.0633 | 2.9259 | 2.9098 | 2.9571 | n/a | n/a | 0.0161 | 7.842053e-09 |
| model.layers.7.mlp.gate_proj | gate_proj | 3.0643 | 2.9724 | 2.9625 | 3.0112 | n/a | n/a | 0.0099 | 3.652341e-07 |
| model.layers.7.mlp.up_proj | up_proj | 3.0673 | 2.9718 | 2.9619 | 3.0206 | n/a | n/a | 0.0099 | 2.309954e-07 |
| model.layers.7.mlp.down_proj | down_proj | 3.0705 | 2.9830 | 2.9727 | 3.0211 | n/a | n/a | 0.0103 | 4.239141e-09 |
| model.layers.8.self_attn.q_proj | q_proj | 3.0735 | 2.8889 | 2.8728 | 2.9175 | n/a | n/a | 0.0161 | 1.152825e-06 |
| model.layers.8.self_attn.k_proj | k_proj | 3.0751 | 2.7898 | 2.7488 | 2.7866 | n/a | n/a | 0.0410 | 3.391230e-06 |
| model.layers.8.self_attn.v_proj | v_proj | 3.0757 | 2.7317 | 2.6907 | 2.7426 | n/a | n/a | 0.0410 | 3.862204e-07 |
| model.layers.8.self_attn.o_proj | o_proj | 3.0765 | 2.9385 | 2.9223 | 2.9717 | n/a | n/a | 0.0161 | 1.085038e-08 |
| model.layers.8.mlp.gate_proj | gate_proj | 3.0777 | 2.9603 | 2.9505 | 2.9979 | n/a | n/a | 0.0099 | 4.070876e-07 |
| model.layers.8.mlp.up_proj | up_proj | 3.0820 | 2.9950 | 2.9851 | 3.0433 | n/a | n/a | 0.0099 | 2.506606e-07 |
| model.layers.8.mlp.down_proj | down_proj | 3.0853 | 2.9964 | 2.9862 | 3.0354 | n/a | n/a | 0.0103 | 5.437995e-09 |
| model.layers.9.self_attn.q_proj | q_proj | 3.0888 | 2.9582 | 2.9421 | 2.9903 | n/a | n/a | 0.0161 | 1.291732e-06 |
| model.layers.9.self_attn.k_proj | k_proj | 3.0901 | 2.7271 | 2.6861 | 2.7266 | n/a | n/a | 0.0410 | 3.337576e-06 |
| model.layers.9.self_attn.v_proj | v_proj | 3.0910 | 2.7836 | 2.7426 | 2.7940 | n/a | n/a | 0.0410 | 4.147176e-07 |
| model.layers.9.self_attn.o_proj | o_proj | 3.0917 | 2.9522 | 2.9361 | 2.9822 | n/a | n/a | 0.0161 | 1.325666e-08 |
| model.layers.9.mlp.gate_proj | gate_proj | 3.0932 | 3.0114 | 3.0015 | 3.0514 | n/a | n/a | 0.0099 | 4.094734e-07 |
| model.layers.9.mlp.up_proj | up_proj | 3.0966 | 2.9958 | 2.9859 | 3.0431 | n/a | n/a | 0.0099 | 2.639968e-07 |
| model.layers.9.mlp.down_proj | down_proj | 3.1010 | 3.0124 | 3.0021 | 3.0523 | n/a | n/a | 0.0103 | 6.350060e-09 |
| model.layers.10.self_attn.q_proj | q_proj | 3.1051 | 2.9346 | 2.9185 | 2.9638 | n/a | n/a | 0.0161 | 1.251744e-06 |
| model.layers.10.self_attn.k_proj | k_proj | 3.1071 | 2.6788 | 2.6378 | 2.6797 | n/a | n/a | 0.0410 | 3.665857e-06 |
| model.layers.10.self_attn.v_proj | v_proj | 3.1083 | 2.7496 | 2.7086 | 2.7504 | n/a | n/a | 0.0410 | 3.941417e-07 |
| model.layers.10.self_attn.o_proj | o_proj | 3.1094 | 2.9630 | 2.9469 | 2.9948 | n/a | n/a | 0.0161 | 9.816126e-09 |
| model.layers.10.mlp.gate_proj | gate_proj | 3.1111 | 3.0606 | 3.0507 | 3.1055 | n/a | n/a | 0.0099 | 4.476003e-07 |
| model.layers.10.mlp.up_proj | up_proj | 3.1136 | 3.0123 | 3.0024 | 3.0632 | n/a | n/a | 0.0099 | 3.103461e-07 |
| model.layers.10.mlp.down_proj | down_proj | 3.1189 | 3.0286 | 3.0184 | 3.0694 | n/a | n/a | 0.0103 | 7.872619e-09 |
| model.layers.11.self_attn.q_proj | q_proj | 3.1239 | 2.9784 | 2.9623 | 3.0124 | n/a | n/a | 0.0161 | 1.225063e-06 |
| model.layers.11.self_attn.k_proj | k_proj | 3.1259 | 2.7324 | 2.6914 | 2.7291 | n/a | n/a | 0.0410 | 3.663229e-06 |
| model.layers.11.self_attn.v_proj | v_proj | 3.1273 | 2.7618 | 2.7208 | 2.7622 | n/a | n/a | 0.0410 | 4.119648e-07 |
| model.layers.11.self_attn.o_proj | o_proj | 3.1286 | 2.9797 | 2.9636 | 3.0123 | n/a | n/a | 0.0161 | 6.895591e-09 |
| model.layers.11.mlp.gate_proj | gate_proj | 3.1307 | 3.0437 | 3.0338 | 3.0880 | n/a | n/a | 0.0099 | 5.064471e-07 |
| model.layers.11.mlp.up_proj | up_proj | 3.1360 | 3.0488 | 3.0389 | 3.0989 | n/a | n/a | 0.0099 | 3.334823e-07 |
| model.layers.11.mlp.down_proj | down_proj | 3.1416 | 3.0513 | 3.0411 | 3.0947 | n/a | n/a | 0.0103 | 8.644678e-09 |
| model.layers.12.self_attn.q_proj | q_proj | 3.1479 | 3.0094 | 2.9933 | 3.0466 | n/a | n/a | 0.0161 | 1.000022e-06 |
| model.layers.12.self_attn.k_proj | k_proj | 3.1503 | 2.7409 | 2.6999 | 2.7457 | n/a | n/a | 0.0410 | 2.888490e-06 |
| model.layers.12.self_attn.v_proj | v_proj | 3.1521 | 2.8058 | 2.7648 | 2.8051 | n/a | n/a | 0.0410 | 3.838244e-07 |
| model.layers.12.self_attn.o_proj | o_proj | 3.1536 | 3.0085 | 2.9924 | 3.0460 | n/a | n/a | 0.0161 | 7.194857e-09 |
| model.layers.12.mlp.gate_proj | gate_proj | 3.1562 | 3.0727 | 3.0628 | 3.1190 | n/a | n/a | 0.0099 | 4.833256e-07 |
| model.layers.12.mlp.up_proj | up_proj | 3.1627 | 3.0715 | 3.0616 | 3.1171 | n/a | n/a | 0.0099 | 3.449997e-07 |
| model.layers.12.mlp.down_proj | down_proj | 3.1704 | 3.0813 | 3.0710 | 3.1263 | n/a | n/a | 0.0103 | 1.026244e-08 |
| model.layers.13.self_attn.q_proj | q_proj | 3.1786 | 3.0541 | 3.0380 | 3.0966 | n/a | n/a | 0.0161 | 1.097294e-06 |
| model.layers.13.self_attn.k_proj | k_proj | 3.1815 | 2.7589 | 2.7179 | 2.7551 | n/a | n/a | 0.0410 | 2.973558e-06 |
| model.layers.13.self_attn.v_proj | v_proj | 3.1840 | 2.8251 | 2.7840 | 2.8258 | n/a | n/a | 0.0410 | 6.074951e-07 |
| model.layers.13.self_attn.o_proj | o_proj | 3.1862 | 3.0438 | 3.0277 | 3.0826 | n/a | n/a | 0.0161 | 8.420580e-09 |
| model.layers.13.mlp.gate_proj | gate_proj | 3.1897 | 3.1120 | 3.1021 | 3.1509 | n/a | n/a | 0.0099 | 4.852776e-07 |
| model.layers.13.mlp.up_proj | up_proj | 3.1981 | 3.1058 | 3.0959 | 3.1454 | n/a | n/a | 0.0099 | 3.864766e-07 |
| model.layers.13.mlp.down_proj | down_proj | 3.2092 | 3.1204 | 3.1102 | 3.1579 | n/a | n/a | 0.0103 | 1.358396e-08 |
| model.layers.14.self_attn.q_proj | q_proj | 3.2215 | 3.0910 | 3.0749 | 3.1377 | n/a | n/a | 0.0161 | 8.659942e-07 |
| model.layers.14.self_attn.k_proj | k_proj | 3.2261 | 2.8562 | 2.8152 | 2.8588 | n/a | n/a | 0.0410 | 1.671536e-06 |
| model.layers.14.self_attn.v_proj | v_proj | 3.2295 | 2.8363 | 2.7953 | 2.8283 | n/a | n/a | 0.0410 | 9.042292e-07 |
| model.layers.14.self_attn.o_proj | o_proj | 3.2331 | 3.0872 | 3.0711 | 3.1230 | n/a | n/a | 0.0161 | 1.565350e-08 |
| model.layers.14.mlp.gate_proj | gate_proj | 3.2386 | 3.1380 | 3.1281 | 3.1740 | n/a | n/a | 0.0099 | 5.576860e-07 |
| model.layers.14.mlp.up_proj | up_proj | 3.2564 | 3.1744 | 3.1645 | 3.2044 | n/a | n/a | 0.0099 | 3.949980e-07 |
| model.layers.14.mlp.down_proj | down_proj | 3.2742 | 3.1843 | 3.1741 | 3.2126 | n/a | n/a | 0.0103 | 1.552448e-08 |
| model.layers.15.self_attn.q_proj | q_proj | 3.2990 | 3.1391 | 3.1230 | 3.1852 | n/a | n/a | 0.0161 | 7.791598e-07 |
| model.layers.15.self_attn.k_proj | k_proj | 3.3108 | 2.9222 | 2.8812 | 2.9173 | n/a | n/a | 0.0410 | 1.734712e-06 |
| model.layers.15.self_attn.v_proj | v_proj | 3.3181 | 3.0010 | 2.9600 | 3.0017 | n/a | n/a | 0.0410 | 7.725162e-07 |
| model.layers.15.self_attn.o_proj | o_proj | 3.3242 | 3.1739 | 3.1578 | 3.2181 | n/a | n/a | 0.0161 | 2.810611e-08 |
| model.layers.15.mlp.gate_proj | gate_proj | 3.3368 | 3.2483 | 3.2384 | 3.2746 | n/a | n/a | 0.0099 | 4.998177e-07 |
| model.layers.15.mlp.up_proj | up_proj | 3.3810 | 3.2948 | 3.2849 | 3.3174 | n/a | n/a | 0.0099 | 3.760620e-07 |
| model.layers.15.mlp.down_proj | down_proj | 3.4673 | 3.3748 | 3.3645 | 3.3985 | n/a | n/a | 0.0103 | 2.314424e-08 |

## Notes

- Exact Huffman shortest/longest symbol lengths are unavailable for this historical run because the integer Huffman symbols were not serialized in the saved artifact.
