# WaterSIC Run Report

- Timestamp: `2026-03-13T19:32:55Z`
- Git commit: `cef970ceae851dbd6297b2b4f3abc3dad76c78f8`
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
- Side-information overhead: `0.0119`
- Perplexity: `16.609645721988343`

## Layer Summary

| Layer | Kind | Target | Achieved | Entropy | Huffman | Side Info | Weighted Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model.layers.0.self_attn.q_proj | q_proj | 3.0000 | 2.8878 | 2.8717 | 2.9297 | 0.0161 | 2.877172e-07 |
| model.layers.0.self_attn.k_proj | k_proj | 3.0005 | 2.7002 | 2.6592 | 2.7243 | 0.0410 | 6.706741e-07 |
| model.layers.0.self_attn.v_proj | v_proj | 3.0008 | 2.6229 | 2.5819 | 2.6452 | 0.0410 | 2.457768e-08 |
| model.layers.0.self_attn.o_proj | o_proj | 3.0012 | 2.8649 | 2.8488 | 2.8778 | 0.0161 | 2.519698e-10 |
| model.layers.0.mlp.gate_proj | gate_proj | 3.0018 | 2.9110 | 2.9011 | 2.9535 | 0.0099 | 1.314158e-07 |
| model.layers.0.mlp.up_proj | up_proj | 3.0034 | 2.9162 | 2.9063 | 2.9617 | 0.0099 | 1.030898e-07 |
| model.layers.0.mlp.down_proj | down_proj | 3.0050 | 2.9138 | 2.9035 | 2.9438 | 0.0103 | 4.457875e-10 |
| model.layers.1.self_attn.q_proj | q_proj | 3.0067 | 2.8579 | 2.8417 | 2.8826 | 0.0162 | 5.628793e-07 |
| model.layers.1.self_attn.k_proj | k_proj | 3.0074 | 2.5632 | 2.5222 | 2.5933 | 0.0411 | 1.983797e-06 |
| model.layers.1.self_attn.v_proj | v_proj | 3.0079 | 2.6118 | 2.5708 | 2.6221 | 0.0411 | 1.023197e-07 |
| model.layers.1.self_attn.o_proj | o_proj | 3.0083 | 2.8832 | 2.8671 | 2.9049 | 0.0161 | 1.207722e-09 |
| model.layers.1.mlp.gate_proj | gate_proj | 3.0089 | 2.9199 | 2.9100 | 2.9626 | 0.0099 | 2.039106e-07 |
| model.layers.1.mlp.up_proj | up_proj | 3.0106 | 2.9221 | 2.9122 | 2.9687 | 0.0099 | 1.462550e-07 |
| model.layers.1.mlp.down_proj | down_proj | 3.0123 | 2.9137 | 2.9034 | 2.9514 | 0.0103 | 8.040071e-10 |
| model.layers.2.self_attn.q_proj | q_proj | 3.0143 | 2.9224 | 2.9063 | 2.9526 | 0.0161 | 1.042892e-06 |
| model.layers.2.self_attn.k_proj | k_proj | 3.0147 | 2.6345 | 2.5935 | 2.6418 | 0.0410 | 3.380532e-06 |
| model.layers.2.self_attn.v_proj | v_proj | 3.0152 | 2.6512 | 2.6102 | 2.6547 | 0.0410 | 2.800136e-07 |
| model.layers.2.self_attn.o_proj | o_proj | 3.0156 | 2.8909 | 2.8747 | 2.9240 | 0.0161 | 1.847752e-09 |
| model.layers.2.mlp.gate_proj | gate_proj | 3.0163 | 2.9317 | 2.9218 | 2.9739 | 0.0099 | 3.087490e-07 |
| model.layers.2.mlp.up_proj | up_proj | 3.0180 | 2.9278 | 2.9179 | 2.9756 | 0.0099 | 1.913898e-07 |
| model.layers.2.mlp.down_proj | down_proj | 3.0199 | 2.9306 | 2.9203 | 2.9689 | 0.0103 | 1.921754e-09 |
| model.layers.3.self_attn.q_proj | q_proj | 3.0218 | 2.8398 | 2.8237 | 2.8628 | 0.0161 | 1.060771e-06 |
| model.layers.3.self_attn.k_proj | k_proj | 3.0227 | 2.7185 | 2.6775 | 2.7167 | 0.0410 | 2.504296e-06 |
| model.layers.3.self_attn.v_proj | v_proj | 3.0231 | 2.7110 | 2.6700 | 2.7199 | 0.0410 | 3.222395e-07 |
| model.layers.3.self_attn.o_proj | o_proj | 3.0235 | 2.8935 | 2.8774 | 2.9286 | 0.0161 | 4.015682e-09 |
| model.layers.3.mlp.gate_proj | gate_proj | 3.0242 | 2.9250 | 2.9151 | 2.9644 | 0.0099 | 4.407092e-07 |
| model.layers.3.mlp.up_proj | up_proj | 3.0264 | 2.9382 | 2.9283 | 2.9869 | 0.0099 | 2.144359e-07 |
| model.layers.3.mlp.down_proj | down_proj | 3.0284 | 2.9383 | 2.9280 | 2.9765 | 0.0103 | 2.799366e-09 |
| model.layers.4.self_attn.q_proj | q_proj | 3.0305 | 2.8960 | 2.8799 | 2.9202 | 0.0161 | 8.715101e-07 |
| model.layers.4.self_attn.k_proj | k_proj | 3.0313 | 2.6783 | 2.6373 | 2.6780 | 0.0410 | 2.505195e-06 |
| model.layers.4.self_attn.v_proj | v_proj | 3.0318 | 2.7301 | 2.6891 | 2.7397 | 0.0410 | 2.770959e-07 |
| model.layers.4.self_attn.o_proj | o_proj | 3.0322 | 2.8989 | 2.8828 | 2.9242 | 0.0161 | 5.390098e-09 |
| model.layers.4.mlp.gate_proj | gate_proj | 3.0330 | 2.9486 | 2.9388 | 2.9890 | 0.0099 | 4.424975e-07 |
| model.layers.4.mlp.up_proj | up_proj | 3.0350 | 2.9356 | 2.9257 | 2.9831 | 0.0099 | 2.037535e-07 |
| model.layers.4.mlp.down_proj | down_proj | 3.0374 | 2.9473 | 2.9370 | 2.9863 | 0.0103 | 3.067168e-09 |
| model.layers.5.self_attn.q_proj | q_proj | 3.0397 | 2.9394 | 2.9233 | 2.9696 | 0.0161 | 9.010847e-07 |
| model.layers.5.self_attn.k_proj | k_proj | 3.0403 | 2.6070 | 2.5660 | 2.6208 | 0.0410 | 3.373767e-06 |
| model.layers.5.self_attn.v_proj | v_proj | 3.0410 | 2.6607 | 2.6197 | 2.6637 | 0.0410 | 2.583084e-07 |
| model.layers.5.self_attn.o_proj | o_proj | 3.0416 | 2.8993 | 2.8832 | 2.9313 | 0.0161 | 5.918469e-09 |
| model.layers.5.mlp.gate_proj | gate_proj | 3.0425 | 2.9597 | 2.9498 | 2.9995 | 0.0099 | 3.910807e-07 |
| model.layers.5.mlp.up_proj | up_proj | 3.0447 | 2.9564 | 2.9465 | 3.0042 | 0.0099 | 2.086368e-07 |
| model.layers.5.mlp.down_proj | down_proj | 3.0470 | 2.9614 | 2.9511 | 2.9998 | 0.0103 | 3.387741e-09 |
| model.layers.6.self_attn.q_proj | q_proj | 3.0494 | 2.9364 | 2.9203 | 2.9658 | 0.0161 | 8.825402e-07 |
| model.layers.6.self_attn.k_proj | k_proj | 3.0502 | 2.6082 | 2.5672 | 2.6194 | 0.0410 | 3.621830e-06 |
| model.layers.6.self_attn.v_proj | v_proj | 3.0510 | 2.6799 | 2.6389 | 2.6870 | 0.0410 | 3.608143e-07 |
| model.layers.6.self_attn.o_proj | o_proj | 3.0516 | 2.9096 | 2.8935 | 2.9417 | 0.0161 | 8.579524e-09 |
| model.layers.6.mlp.gate_proj | gate_proj | 3.0526 | 2.9461 | 2.9362 | 2.9849 | 0.0099 | 3.907501e-07 |
| model.layers.6.mlp.up_proj | up_proj | 3.0557 | 2.9718 | 2.9619 | 3.0204 | 0.0099 | 2.070961e-07 |
| model.layers.6.mlp.down_proj | down_proj | 3.0582 | 2.9715 | 2.9613 | 3.0102 | 0.0103 | 3.523863e-09 |
| model.layers.7.self_attn.q_proj | q_proj | 3.0608 | 2.9554 | 2.9393 | 2.9848 | 0.0161 | 9.630186e-07 |
| model.layers.7.self_attn.k_proj | k_proj | 3.0617 | 2.7309 | 2.6899 | 2.7299 | 0.0410 | 3.163887e-06 |
| model.layers.7.self_attn.v_proj | v_proj | 3.0623 | 2.6867 | 2.6456 | 2.6942 | 0.0410 | 4.038417e-07 |
| model.layers.7.self_attn.o_proj | o_proj | 3.0630 | 2.9231 | 2.9069 | 2.9517 | 0.0161 | 7.267208e-09 |
| model.layers.7.mlp.gate_proj | gate_proj | 3.0641 | 2.9719 | 2.9620 | 3.0106 | 0.0099 | 3.378992e-07 |
| model.layers.7.mlp.up_proj | up_proj | 3.0671 | 2.9719 | 2.9620 | 3.0190 | 0.0099 | 2.140111e-07 |
| model.layers.7.mlp.down_proj | down_proj | 3.0703 | 2.9819 | 2.9717 | 3.0209 | 0.0103 | 3.790857e-09 |
| model.layers.8.self_attn.q_proj | q_proj | 3.0733 | 2.8904 | 2.8743 | 2.9186 | 0.0161 | 1.069757e-06 |
| model.layers.8.self_attn.k_proj | k_proj | 3.0749 | 2.7872 | 2.7462 | 2.7832 | 0.0410 | 3.171510e-06 |
| model.layers.8.self_attn.v_proj | v_proj | 3.0755 | 2.7306 | 2.6896 | 2.7410 | 0.0410 | 3.610525e-07 |
| model.layers.8.self_attn.o_proj | o_proj | 3.0763 | 2.9390 | 2.9228 | 2.9713 | 0.0161 | 9.920814e-09 |
| model.layers.8.mlp.gate_proj | gate_proj | 3.0775 | 2.9604 | 2.9505 | 2.9979 | 0.0099 | 3.772327e-07 |
| model.layers.8.mlp.up_proj | up_proj | 3.0818 | 2.9943 | 2.9844 | 3.0409 | 0.0099 | 2.326988e-07 |
| model.layers.8.mlp.down_proj | down_proj | 3.0851 | 2.9953 | 2.9850 | 3.0350 | 0.0103 | 4.757846e-09 |
| model.layers.9.self_attn.q_proj | q_proj | 3.0886 | 2.9576 | 2.9415 | 2.9895 | 0.0161 | 1.210369e-06 |
| model.layers.9.self_attn.k_proj | k_proj | 3.0899 | 2.7286 | 2.6876 | 2.7272 | 0.0410 | 3.122623e-06 |
| model.layers.9.self_attn.v_proj | v_proj | 3.0908 | 2.7830 | 2.7420 | 2.7917 | 0.0410 | 3.906549e-07 |
| model.layers.9.self_attn.o_proj | o_proj | 3.0916 | 2.9503 | 2.9342 | 2.9800 | 0.0161 | 1.204981e-08 |
| model.layers.9.mlp.gate_proj | gate_proj | 3.0930 | 3.0113 | 3.0014 | 3.0511 | 0.0099 | 3.752429e-07 |
| model.layers.9.mlp.up_proj | up_proj | 3.0965 | 2.9958 | 2.9860 | 3.0416 | 0.0099 | 2.425787e-07 |
| model.layers.9.mlp.down_proj | down_proj | 3.1009 | 3.0127 | 3.0024 | 3.0540 | 0.0103 | 5.437119e-09 |
| model.layers.10.self_attn.q_proj | q_proj | 3.1049 | 2.9334 | 2.9172 | 2.9626 | 0.0161 | 1.135257e-06 |
| model.layers.10.self_attn.k_proj | k_proj | 3.1069 | 2.6809 | 2.6399 | 2.6817 | 0.0410 | 3.321395e-06 |
| model.layers.10.self_attn.v_proj | v_proj | 3.1082 | 2.7505 | 2.7095 | 2.7499 | 0.0410 | 3.578412e-07 |
| model.layers.10.self_attn.o_proj | o_proj | 3.1092 | 2.9622 | 2.9461 | 2.9930 | 0.0161 | 9.212110e-09 |
| model.layers.10.mlp.gate_proj | gate_proj | 3.1110 | 3.0601 | 3.0502 | 3.1047 | 0.0099 | 4.039951e-07 |
| model.layers.10.mlp.up_proj | up_proj | 3.1135 | 3.0119 | 3.0020 | 3.0604 | 0.0099 | 2.807804e-07 |
| model.layers.10.mlp.down_proj | down_proj | 3.1188 | 3.0275 | 3.0172 | 3.0688 | 0.0103 | 6.562775e-09 |
| model.layers.11.self_attn.q_proj | q_proj | 3.1238 | 2.9791 | 2.9629 | 3.0138 | 0.0161 | 1.084505e-06 |
| model.layers.11.self_attn.k_proj | k_proj | 3.1259 | 2.7316 | 2.6906 | 2.7280 | 0.0410 | 3.268078e-06 |
| model.layers.11.self_attn.v_proj | v_proj | 3.1273 | 2.7623 | 2.7213 | 2.7610 | 0.0410 | 3.664678e-07 |
| model.layers.11.self_attn.o_proj | o_proj | 3.1285 | 2.9803 | 2.9642 | 3.0121 | 0.0161 | 6.253897e-09 |
| model.layers.11.mlp.gate_proj | gate_proj | 3.1307 | 3.0436 | 3.0337 | 3.0871 | 0.0099 | 4.482689e-07 |
| model.layers.11.mlp.up_proj | up_proj | 3.1359 | 3.0481 | 3.0382 | 3.0979 | 0.0099 | 2.958558e-07 |
| model.layers.11.mlp.down_proj | down_proj | 3.1416 | 3.0493 | 3.0390 | 3.0926 | 0.0103 | 6.960737e-09 |
| model.layers.12.self_attn.q_proj | q_proj | 3.1480 | 3.0101 | 2.9940 | 3.0462 | 0.0161 | 8.327998e-07 |
| model.layers.12.self_attn.k_proj | k_proj | 3.1504 | 2.7406 | 2.6996 | 2.7461 | 0.0410 | 2.421430e-06 |
| model.layers.12.self_attn.v_proj | v_proj | 3.1522 | 2.8040 | 2.7630 | 2.8016 | 0.0410 | 3.217045e-07 |
| model.layers.12.self_attn.o_proj | o_proj | 3.1537 | 3.0061 | 2.9900 | 3.0428 | 0.0161 | 6.470605e-09 |
| model.layers.12.mlp.gate_proj | gate_proj | 3.1564 | 3.0728 | 3.0629 | 3.1198 | 0.0099 | 4.181268e-07 |
| model.layers.12.mlp.up_proj | up_proj | 3.1629 | 3.0719 | 3.0620 | 3.1168 | 0.0099 | 2.990500e-07 |
| model.layers.12.mlp.down_proj | down_proj | 3.1705 | 3.0820 | 3.0717 | 3.1287 | 0.0103 | 7.879580e-09 |
| model.layers.13.self_attn.q_proj | q_proj | 3.1787 | 3.0535 | 3.0374 | 3.0948 | 0.0161 | 9.255834e-07 |
| model.layers.13.self_attn.k_proj | k_proj | 3.1816 | 2.7579 | 2.7169 | 2.7546 | 0.0410 | 2.512916e-06 |
| model.layers.13.self_attn.v_proj | v_proj | 3.1841 | 2.8258 | 2.7848 | 2.8246 | 0.0410 | 5.129575e-07 |
| model.layers.13.self_attn.o_proj | o_proj | 3.1863 | 3.0423 | 3.0261 | 3.0809 | 0.0161 | 7.226593e-09 |
| model.layers.13.mlp.gate_proj | gate_proj | 3.1898 | 3.1117 | 3.1018 | 3.1514 | 0.0099 | 4.154305e-07 |
| model.layers.13.mlp.up_proj | up_proj | 3.1982 | 3.1061 | 3.0962 | 3.1447 | 0.0099 | 3.313672e-07 |
| model.layers.13.mlp.down_proj | down_proj | 3.2094 | 3.1210 | 3.1108 | 3.1619 | 0.0103 | 1.024527e-08 |
| model.layers.14.self_attn.q_proj | q_proj | 3.2216 | 3.0916 | 3.0755 | 3.1385 | 0.0161 | 6.953729e-07 |
| model.layers.14.self_attn.k_proj | k_proj | 3.2262 | 2.8564 | 2.8153 | 2.8610 | 0.0410 | 1.344898e-06 |
| model.layers.14.self_attn.v_proj | v_proj | 3.2296 | 2.8333 | 2.7923 | 2.8247 | 0.0410 | 7.322993e-07 |
| model.layers.14.self_attn.o_proj | o_proj | 3.2332 | 3.0874 | 3.0713 | 3.1236 | 0.0161 | 1.290150e-08 |
| model.layers.14.mlp.gate_proj | gate_proj | 3.2387 | 3.1378 | 3.1279 | 3.1748 | 0.0099 | 4.741071e-07 |
| model.layers.14.mlp.up_proj | up_proj | 3.2566 | 3.1744 | 3.1645 | 3.2033 | 0.0099 | 3.365575e-07 |
| model.layers.14.mlp.down_proj | down_proj | 3.2744 | 3.1853 | 3.1750 | 3.2170 | 0.0103 | 1.159852e-08 |
| model.layers.15.self_attn.q_proj | q_proj | 3.2989 | 3.1398 | 3.1237 | 3.1862 | 0.0161 | 6.184076e-07 |
| model.layers.15.self_attn.k_proj | k_proj | 3.3107 | 2.9212 | 2.8802 | 2.9172 | 0.0410 | 1.381357e-06 |
| model.layers.15.self_attn.v_proj | v_proj | 3.3181 | 3.0017 | 2.9606 | 3.0006 | 0.0410 | 6.134919e-07 |
| model.layers.15.self_attn.o_proj | o_proj | 3.3242 | 3.1740 | 3.1579 | 3.2174 | 0.0161 | 2.365846e-08 |
| model.layers.15.mlp.gate_proj | gate_proj | 3.3367 | 3.2479 | 3.2380 | 3.2752 | 0.0099 | 4.227373e-07 |
| model.layers.15.mlp.up_proj | up_proj | 3.3811 | 3.2950 | 3.2851 | 3.3165 | 0.0099 | 3.187532e-07 |
| model.layers.15.mlp.down_proj | down_proj | 3.4671 | 3.3758 | 3.3656 | 3.4043 | 0.0103 | 1.747039e-08 |
