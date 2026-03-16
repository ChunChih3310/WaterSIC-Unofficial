# WaterSIC Run Report

- Timestamp: `2026-03-14T18:00:39Z`
- Git commit: `efda59d419fc818c6ad4becb63d0d3d195813335`
- Environment: `watersic`
- Device: `cuda`
- Model: `meta-llama/Llama-3.2-1B`
- Model revision: `main`
- Tokenizer: `meta-llama/Llama-3.2-1B`
- Sequence length: `2048`
- Calibration sequences: `8`
- Target global bitwidth: `3.0000`
- Achieved global bitwidth: `2.9984`
- Raw average bitwidth: `8.4601`
- Entropy average bitwidth: `2.9865`
- Huffman average bitwidth: `3.0368`
- Huffman shortest symbol length: `n/a`
- Huffman longest symbol length: `n/a`
- Side-information overhead: `0.0119`
- Perplexity: `16.27963264787442`

## Layer Summary

| Layer | Kind | Target | Achieved | Entropy | Huffman | Huff Min | Huff Max | Side Info | Weighted Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model.layers.0.self_attn.q_proj | q_proj | 3.0000 | 2.8881 | 2.8720 | 2.9296 | n/a | n/a | 0.0161 | 3.268353e-07 |
| model.layers.0.self_attn.k_proj | k_proj | 3.0005 | 2.7032 | 2.6621 | 2.7284 | n/a | n/a | 0.0410 | 7.578682e-07 |
| model.layers.0.self_attn.v_proj | v_proj | 3.0008 | 2.6211 | 2.5800 | 2.6443 | n/a | n/a | 0.0410 | 2.802020e-08 |
| model.layers.0.self_attn.o_proj | o_proj | 3.0012 | 2.8674 | 2.8513 | 2.8801 | n/a | n/a | 0.0161 | 2.510545e-10 |
| model.layers.0.mlp.gate_proj | gate_proj | 3.0018 | 2.9117 | 2.9018 | 2.9543 | n/a | n/a | 0.0099 | 1.312833e-07 |
| model.layers.0.mlp.up_proj | up_proj | 3.0034 | 2.9171 | 2.9072 | 2.9625 | n/a | n/a | 0.0099 | 1.029727e-07 |
| model.layers.0.mlp.down_proj | down_proj | 3.0050 | 2.9145 | 2.9043 | 2.9447 | n/a | n/a | 0.0103 | 4.457244e-10 |
| model.layers.1.self_attn.q_proj | q_proj | 3.0066 | 2.8579 | 2.8417 | 2.8825 | n/a | n/a | 0.0162 | 6.755639e-07 |
| model.layers.1.self_attn.k_proj | k_proj | 3.0073 | 2.5638 | 2.5228 | 2.5926 | n/a | n/a | 0.0411 | 2.378357e-06 |
| model.layers.1.self_attn.v_proj | v_proj | 3.0078 | 2.6126 | 2.5715 | 2.6223 | n/a | n/a | 0.0411 | 1.225806e-07 |
| model.layers.1.self_attn.o_proj | o_proj | 3.0083 | 2.8828 | 2.8667 | 2.9043 | n/a | n/a | 0.0161 | 1.206843e-09 |
| model.layers.1.mlp.gate_proj | gate_proj | 3.0089 | 2.9201 | 2.9102 | 2.9629 | n/a | n/a | 0.0099 | 2.037743e-07 |
| model.layers.1.mlp.up_proj | up_proj | 3.0105 | 2.9216 | 2.9117 | 2.9683 | n/a | n/a | 0.0099 | 1.462962e-07 |
| model.layers.1.mlp.down_proj | down_proj | 3.0123 | 2.9086 | 2.8983 | 2.9454 | n/a | n/a | 0.0103 | 8.166147e-10 |
| model.layers.2.self_attn.q_proj | q_proj | 3.0143 | 2.9222 | 2.9061 | 2.9524 | n/a | n/a | 0.0161 | 1.128117e-06 |
| model.layers.2.self_attn.k_proj | k_proj | 3.0148 | 2.6347 | 2.5937 | 2.6420 | n/a | n/a | 0.0410 | 3.649666e-06 |
| model.layers.2.self_attn.v_proj | v_proj | 3.0152 | 2.6494 | 2.6084 | 2.6527 | n/a | n/a | 0.0410 | 3.032410e-07 |
| model.layers.2.self_attn.o_proj | o_proj | 3.0157 | 2.8917 | 2.8756 | 2.9248 | n/a | n/a | 0.0161 | 1.847310e-09 |
| model.layers.2.mlp.gate_proj | gate_proj | 3.0163 | 2.9324 | 2.9225 | 2.9746 | n/a | n/a | 0.0099 | 3.085562e-07 |
| model.layers.2.mlp.up_proj | up_proj | 3.0180 | 2.9280 | 2.9181 | 2.9758 | n/a | n/a | 0.0099 | 1.912631e-07 |
| model.layers.2.mlp.down_proj | down_proj | 3.0199 | 2.9313 | 2.9210 | 2.9695 | n/a | n/a | 0.0103 | 1.925126e-09 |
| model.layers.3.self_attn.q_proj | q_proj | 3.0218 | 2.8399 | 2.8238 | 2.8630 | n/a | n/a | 0.0161 | 1.144715e-06 |
| model.layers.3.self_attn.k_proj | k_proj | 3.0227 | 2.7169 | 2.6759 | 2.7150 | n/a | n/a | 0.0410 | 2.701754e-06 |
| model.layers.3.self_attn.v_proj | v_proj | 3.0231 | 2.7133 | 2.6723 | 2.7219 | n/a | n/a | 0.0410 | 3.458532e-07 |
| model.layers.3.self_attn.o_proj | o_proj | 3.0236 | 2.8936 | 2.8775 | 2.9284 | n/a | n/a | 0.0161 | 4.023646e-09 |
| model.layers.3.mlp.gate_proj | gate_proj | 3.0243 | 2.9248 | 2.9149 | 2.9644 | n/a | n/a | 0.0099 | 4.408039e-07 |
| model.layers.3.mlp.up_proj | up_proj | 3.0264 | 2.9381 | 2.9282 | 2.9869 | n/a | n/a | 0.0099 | 2.143389e-07 |
| model.layers.3.mlp.down_proj | down_proj | 3.0284 | 2.9405 | 2.9303 | 2.9787 | n/a | n/a | 0.0103 | 2.792356e-09 |
| model.layers.4.self_attn.q_proj | q_proj | 3.0304 | 2.8956 | 2.8795 | 2.9200 | n/a | n/a | 0.0161 | 9.349844e-07 |
| model.layers.4.self_attn.k_proj | k_proj | 3.0312 | 2.6752 | 2.6341 | 2.6753 | n/a | n/a | 0.0410 | 2.701130e-06 |
| model.layers.4.self_attn.v_proj | v_proj | 3.0317 | 2.7318 | 2.6908 | 2.7419 | n/a | n/a | 0.0410 | 2.956465e-07 |
| model.layers.4.self_attn.o_proj | o_proj | 3.0322 | 2.8996 | 2.8834 | 2.9249 | n/a | n/a | 0.0161 | 5.393174e-09 |
| model.layers.4.mlp.gate_proj | gate_proj | 3.0330 | 2.9487 | 2.9388 | 2.9889 | n/a | n/a | 0.0099 | 4.423790e-07 |
| model.layers.4.mlp.up_proj | up_proj | 3.0350 | 2.9358 | 2.9259 | 2.9834 | n/a | n/a | 0.0099 | 2.037139e-07 |
| model.layers.4.mlp.down_proj | down_proj | 3.0374 | 2.9481 | 2.9379 | 2.9871 | n/a | n/a | 0.0103 | 3.067800e-09 |
| model.layers.5.self_attn.q_proj | q_proj | 3.0396 | 2.9403 | 2.9242 | 2.9708 | n/a | n/a | 0.0161 | 9.563022e-07 |
| model.layers.5.self_attn.k_proj | k_proj | 3.0403 | 2.6074 | 2.5664 | 2.6204 | n/a | n/a | 0.0410 | 3.578254e-06 |
| model.layers.5.self_attn.v_proj | v_proj | 3.0409 | 2.6583 | 2.6173 | 2.6610 | n/a | n/a | 0.0410 | 2.750498e-07 |
| model.layers.5.self_attn.o_proj | o_proj | 3.0415 | 2.9003 | 2.8842 | 2.9323 | n/a | n/a | 0.0161 | 5.912990e-09 |
| model.layers.5.mlp.gate_proj | gate_proj | 3.0424 | 2.9592 | 2.9493 | 2.9989 | n/a | n/a | 0.0099 | 3.914445e-07 |
| model.layers.5.mlp.up_proj | up_proj | 3.0446 | 2.9561 | 2.9462 | 3.0040 | n/a | n/a | 0.0099 | 2.085768e-07 |
| model.layers.5.mlp.down_proj | down_proj | 3.0470 | 2.9616 | 2.9513 | 2.9999 | n/a | n/a | 0.0103 | 3.386002e-09 |
| model.layers.6.self_attn.q_proj | q_proj | 3.0493 | 2.9366 | 2.9205 | 2.9665 | n/a | n/a | 0.0161 | 8.800186e-07 |
| model.layers.6.self_attn.k_proj | k_proj | 3.0501 | 2.6089 | 2.5679 | 2.6194 | n/a | n/a | 0.0410 | 3.600178e-06 |
| model.layers.6.self_attn.v_proj | v_proj | 3.0509 | 2.6814 | 2.6404 | 2.6893 | n/a | n/a | 0.0410 | 3.586798e-07 |
| model.layers.6.self_attn.o_proj | o_proj | 3.0515 | 2.9102 | 2.8941 | 2.9422 | n/a | n/a | 0.0161 | 8.594528e-09 |
| model.layers.6.mlp.gate_proj | gate_proj | 3.0525 | 2.9459 | 2.9360 | 2.9848 | n/a | n/a | 0.0099 | 3.905980e-07 |
| model.layers.6.mlp.up_proj | up_proj | 3.0556 | 2.9720 | 2.9621 | 3.0207 | n/a | n/a | 0.0099 | 2.069494e-07 |
| model.layers.6.mlp.down_proj | down_proj | 3.0581 | 2.9720 | 2.9618 | 3.0107 | n/a | n/a | 0.0103 | 3.524173e-09 |
| model.layers.7.self_attn.q_proj | q_proj | 3.0607 | 2.9566 | 2.9405 | 2.9862 | n/a | n/a | 0.0161 | 9.838250e-07 |
| model.layers.7.self_attn.k_proj | k_proj | 3.0615 | 2.7298 | 2.6887 | 2.7286 | n/a | n/a | 0.0410 | 3.249456e-06 |
| model.layers.7.self_attn.v_proj | v_proj | 3.0622 | 2.6874 | 2.6464 | 2.6949 | n/a | n/a | 0.0410 | 4.122290e-07 |
| model.layers.7.self_attn.o_proj | o_proj | 3.0629 | 2.9229 | 2.9068 | 2.9515 | n/a | n/a | 0.0161 | 7.276751e-09 |
| model.layers.7.mlp.gate_proj | gate_proj | 3.0640 | 2.9717 | 2.9618 | 3.0105 | n/a | n/a | 0.0099 | 3.378218e-07 |
| model.layers.7.mlp.up_proj | up_proj | 3.0670 | 2.9714 | 2.9615 | 3.0184 | n/a | n/a | 0.0099 | 2.140831e-07 |
| model.layers.7.mlp.down_proj | down_proj | 3.0702 | 2.9828 | 2.9726 | 3.0218 | n/a | n/a | 0.0103 | 3.789834e-09 |
| model.layers.8.self_attn.q_proj | q_proj | 3.0732 | 2.8897 | 2.8736 | 2.9181 | n/a | n/a | 0.0161 | 1.098868e-06 |
| model.layers.8.self_attn.k_proj | k_proj | 3.0748 | 2.7881 | 2.7470 | 2.7847 | n/a | n/a | 0.0410 | 3.256590e-06 |
| model.layers.8.self_attn.v_proj | v_proj | 3.0754 | 2.7295 | 2.6885 | 2.7399 | n/a | n/a | 0.0410 | 3.704960e-07 |
| model.layers.8.self_attn.o_proj | o_proj | 3.0762 | 2.9393 | 2.9231 | 2.9715 | n/a | n/a | 0.0161 | 9.914054e-09 |
| model.layers.8.mlp.gate_proj | gate_proj | 3.0774 | 2.9604 | 2.9505 | 2.9981 | n/a | n/a | 0.0099 | 3.771325e-07 |
| model.layers.8.mlp.up_proj | up_proj | 3.0816 | 2.9941 | 2.9843 | 3.0408 | n/a | n/a | 0.0099 | 2.327135e-07 |
| model.layers.8.mlp.down_proj | down_proj | 3.0850 | 2.9971 | 2.9868 | 3.0370 | n/a | n/a | 0.0103 | 4.747093e-09 |
| model.layers.9.self_attn.q_proj | q_proj | 3.0884 | 2.9569 | 2.9408 | 2.9887 | n/a | n/a | 0.0161 | 1.207835e-06 |
| model.layers.9.self_attn.k_proj | k_proj | 3.0897 | 2.7265 | 2.6855 | 2.7253 | n/a | n/a | 0.0410 | 3.129066e-06 |
| model.layers.9.self_attn.v_proj | v_proj | 3.0906 | 2.7823 | 2.7413 | 2.7908 | n/a | n/a | 0.0410 | 3.902053e-07 |
| model.layers.9.self_attn.o_proj | o_proj | 3.0914 | 2.9518 | 2.9357 | 2.9820 | n/a | n/a | 0.0161 | 1.203212e-08 |
| model.layers.9.mlp.gate_proj | gate_proj | 3.0928 | 3.0114 | 3.0015 | 3.0512 | n/a | n/a | 0.0099 | 3.753390e-07 |
| model.layers.9.mlp.up_proj | up_proj | 3.0962 | 2.9958 | 2.9859 | 3.0416 | n/a | n/a | 0.0099 | 2.427092e-07 |
| model.layers.9.mlp.down_proj | down_proj | 3.1007 | 3.0122 | 3.0020 | 3.0534 | n/a | n/a | 0.0103 | 5.438652e-09 |
| model.layers.10.self_attn.q_proj | q_proj | 3.1047 | 2.9338 | 2.9177 | 2.9630 | n/a | n/a | 0.0161 | 1.166112e-06 |
| model.layers.10.self_attn.k_proj | k_proj | 3.1067 | 2.6780 | 2.6370 | 2.6790 | n/a | n/a | 0.0410 | 3.428235e-06 |
| model.layers.10.self_attn.v_proj | v_proj | 3.1080 | 2.7523 | 2.7113 | 2.7522 | n/a | n/a | 0.0410 | 3.668395e-07 |
| model.layers.10.self_attn.o_proj | o_proj | 3.1090 | 2.9630 | 2.9469 | 2.9940 | n/a | n/a | 0.0161 | 9.190102e-09 |
| model.layers.10.mlp.gate_proj | gate_proj | 3.1107 | 3.0600 | 3.0501 | 3.1046 | n/a | n/a | 0.0099 | 4.040740e-07 |
| model.layers.10.mlp.up_proj | up_proj | 3.1132 | 3.0114 | 3.0015 | 3.0598 | n/a | n/a | 0.0099 | 2.809700e-07 |
| model.layers.10.mlp.down_proj | down_proj | 3.1186 | 3.0272 | 3.0170 | 3.0686 | n/a | n/a | 0.0103 | 6.557107e-09 |
| model.layers.11.self_attn.q_proj | q_proj | 3.1236 | 2.9793 | 2.9632 | 3.0139 | n/a | n/a | 0.0161 | 1.126944e-06 |
| model.layers.11.self_attn.k_proj | k_proj | 3.1256 | 2.7295 | 2.6885 | 2.7263 | n/a | n/a | 0.0410 | 3.402606e-06 |
| model.layers.11.self_attn.v_proj | v_proj | 3.1270 | 2.7595 | 2.7185 | 2.7583 | n/a | n/a | 0.0410 | 3.813317e-07 |
| model.layers.11.self_attn.o_proj | o_proj | 3.1283 | 2.9777 | 2.9616 | 3.0093 | n/a | n/a | 0.0161 | 6.268100e-09 |
| model.layers.11.mlp.gate_proj | gate_proj | 3.1305 | 3.0433 | 3.0334 | 3.0867 | n/a | n/a | 0.0099 | 4.485192e-07 |
| model.layers.11.mlp.up_proj | up_proj | 3.1357 | 3.0477 | 3.0378 | 3.0977 | n/a | n/a | 0.0099 | 2.962288e-07 |
| model.layers.11.mlp.down_proj | down_proj | 3.1414 | 3.0502 | 3.0399 | 3.0935 | n/a | n/a | 0.0103 | 6.955102e-09 |
| model.layers.12.self_attn.q_proj | q_proj | 3.1477 | 3.0097 | 2.9936 | 3.0459 | n/a | n/a | 0.0161 | 8.798883e-07 |
| model.layers.12.self_attn.k_proj | k_proj | 3.1501 | 2.7410 | 2.6999 | 2.7467 | n/a | n/a | 0.0410 | 2.547077e-06 |
| model.layers.12.self_attn.v_proj | v_proj | 3.1519 | 2.8049 | 2.7639 | 2.8026 | n/a | n/a | 0.0410 | 3.389288e-07 |
| model.layers.12.self_attn.o_proj | o_proj | 3.1535 | 3.0077 | 2.9916 | 3.0446 | n/a | n/a | 0.0161 | 6.442168e-09 |
| model.layers.12.mlp.gate_proj | gate_proj | 3.1561 | 3.0727 | 3.0629 | 3.1198 | n/a | n/a | 0.0099 | 4.181132e-07 |
| model.layers.12.mlp.up_proj | up_proj | 3.1626 | 3.0714 | 3.0615 | 3.1164 | n/a | n/a | 0.0099 | 2.994513e-07 |
| model.layers.12.mlp.down_proj | down_proj | 3.1703 | 3.0813 | 3.0710 | 3.1280 | n/a | n/a | 0.0103 | 7.880151e-09 |
| model.layers.13.self_attn.q_proj | q_proj | 3.1784 | 3.0539 | 3.0378 | 3.0949 | n/a | n/a | 0.0161 | 9.984398e-07 |
| model.layers.13.self_attn.k_proj | k_proj | 3.1814 | 2.7574 | 2.7164 | 2.7541 | n/a | n/a | 0.0410 | 2.714037e-06 |
| model.layers.13.self_attn.v_proj | v_proj | 3.1839 | 2.8266 | 2.7856 | 2.8257 | n/a | n/a | 0.0410 | 5.524723e-07 |
| model.layers.13.self_attn.o_proj | o_proj | 3.1860 | 3.0438 | 3.0277 | 3.0828 | n/a | n/a | 0.0161 | 7.220608e-09 |
| model.layers.13.mlp.gate_proj | gate_proj | 3.1895 | 3.1118 | 3.1019 | 3.1514 | n/a | n/a | 0.0099 | 4.154668e-07 |
| model.layers.13.mlp.up_proj | up_proj | 3.1979 | 3.1060 | 3.0961 | 3.1447 | n/a | n/a | 0.0099 | 3.314567e-07 |
| model.layers.13.mlp.down_proj | down_proj | 3.2090 | 3.1212 | 3.1110 | 3.1620 | n/a | n/a | 0.0103 | 1.023947e-08 |
| model.layers.14.self_attn.q_proj | q_proj | 3.2211 | 3.0912 | 3.0751 | 3.1381 | n/a | n/a | 0.0161 | 7.742713e-07 |
| model.layers.14.self_attn.k_proj | k_proj | 3.2258 | 2.8557 | 2.8146 | 2.8601 | n/a | n/a | 0.0410 | 1.498166e-06 |
| model.layers.14.self_attn.v_proj | v_proj | 3.2291 | 2.8336 | 2.7926 | 2.8253 | n/a | n/a | 0.0410 | 8.145003e-07 |
| model.layers.14.self_attn.o_proj | o_proj | 3.2327 | 3.0876 | 3.0715 | 3.1238 | n/a | n/a | 0.0161 | 1.290374e-08 |
| model.layers.14.mlp.gate_proj | gate_proj | 3.2382 | 3.1374 | 3.1275 | 3.1744 | n/a | n/a | 0.0099 | 4.744266e-07 |
| model.layers.14.mlp.up_proj | up_proj | 3.2561 | 3.1740 | 3.1641 | 3.2029 | n/a | n/a | 0.0099 | 3.367017e-07 |
| model.layers.14.mlp.down_proj | down_proj | 3.2738 | 3.1844 | 3.1741 | 3.2161 | n/a | n/a | 0.0103 | 1.160710e-08 |
| model.layers.15.self_attn.q_proj | q_proj | 3.2985 | 3.1392 | 3.1231 | 3.1856 | n/a | n/a | 0.0161 | 6.926393e-07 |
| model.layers.15.self_attn.k_proj | k_proj | 3.3103 | 2.9225 | 2.8815 | 2.9181 | n/a | n/a | 0.0410 | 1.540793e-06 |
| model.layers.15.self_attn.v_proj | v_proj | 3.3176 | 3.0015 | 2.9605 | 3.0011 | n/a | n/a | 0.0410 | 6.853031e-07 |
| model.layers.15.self_attn.o_proj | o_proj | 3.3237 | 3.1745 | 3.1584 | 3.2181 | n/a | n/a | 0.0161 | 2.365332e-08 |
| model.layers.15.mlp.gate_proj | gate_proj | 3.3361 | 3.2470 | 3.2371 | 3.2744 | n/a | n/a | 0.0099 | 4.234964e-07 |
| model.layers.15.mlp.up_proj | up_proj | 3.3807 | 3.2948 | 3.2850 | 3.3163 | n/a | n/a | 0.0099 | 3.188940e-07 |
| model.layers.15.mlp.down_proj | down_proj | 3.4666 | 3.3759 | 3.3656 | 3.4044 | n/a | n/a | 0.0103 | 1.745522e-08 |

## Notes

- Exact Huffman shortest/longest symbol lengths are unavailable for this historical run because the integer Huffman symbols were not serialized in the saved artifact.
