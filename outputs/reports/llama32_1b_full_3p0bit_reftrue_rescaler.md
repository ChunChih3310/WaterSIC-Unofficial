# WaterSIC Run Report

- Timestamp: `2026-03-13T14:20:41Z`
- Git commit: `3299c4124f436e394c8cefa7c4a395a363af1df5`
- Environment: `watersic`
- Device: `cuda`
- Model: `meta-llama/Llama-3.2-1B`
- Model revision: `main`
- Tokenizer: `meta-llama/Llama-3.2-1B`
- Sequence length: `2048`
- Calibration sequences: `8`
- Target global bitwidth: `3.0000`
- Achieved global bitwidth: `2.9984`
- Raw average bitwidth: `8.4806`
- Entropy average bitwidth: `2.9865`
- Huffman average bitwidth: `3.0368`
- Side-information overhead: `0.0119`
- Perplexity: `15.702942918276795`

## Layer Summary

| Layer | Kind | Target | Achieved | Entropy | Huffman | Side Info | Weighted Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model.layers.0.self_attn.q_proj | q_proj | 3.0000 | 2.8879 | 2.8717 | 2.9296 | 0.0161 | 3.011346e-07 |
| model.layers.0.self_attn.k_proj | k_proj | 3.0005 | 2.7010 | 2.6600 | 2.7259 | 0.0410 | 7.000186e-07 |
| model.layers.0.self_attn.v_proj | v_proj | 3.0008 | 2.6206 | 2.5795 | 2.6434 | 0.0410 | 2.583929e-08 |
| model.layers.0.self_attn.o_proj | o_proj | 3.0012 | 2.8671 | 2.8510 | 2.8800 | 0.0161 | 2.513894e-10 |
| model.layers.0.mlp.gate_proj | gate_proj | 3.0018 | 2.9117 | 2.9018 | 2.9542 | 0.0099 | 1.313176e-07 |
| model.layers.0.mlp.up_proj | up_proj | 3.0034 | 2.9167 | 2.9068 | 2.9622 | 0.0099 | 1.030131e-07 |
| model.layers.0.mlp.down_proj | down_proj | 3.0050 | 2.9148 | 2.9046 | 2.9451 | 0.0103 | 4.450017e-10 |
| model.layers.1.self_attn.q_proj | q_proj | 3.0066 | 2.8592 | 2.8431 | 2.8839 | 0.0162 | 6.559996e-07 |
| model.layers.1.self_attn.k_proj | k_proj | 3.0073 | 2.5629 | 2.5218 | 2.5922 | 0.0411 | 2.318084e-06 |
| model.layers.1.self_attn.v_proj | v_proj | 3.0078 | 2.6103 | 2.5692 | 2.6203 | 0.0411 | 1.197145e-07 |
| model.layers.1.self_attn.o_proj | o_proj | 3.0083 | 2.8829 | 2.8668 | 2.9044 | 0.0161 | 1.206863e-09 |
| model.layers.1.mlp.gate_proj | gate_proj | 3.0089 | 2.9203 | 2.9104 | 2.9631 | 0.0099 | 2.037886e-07 |
| model.layers.1.mlp.up_proj | up_proj | 3.0105 | 2.9218 | 2.9118 | 2.9683 | 0.0099 | 1.462934e-07 |
| model.layers.1.mlp.down_proj | down_proj | 3.0123 | 2.9110 | 2.9008 | 2.9484 | 0.0103 | 8.099281e-10 |
| model.layers.2.self_attn.q_proj | q_proj | 3.0143 | 2.9242 | 2.9080 | 2.9544 | 0.0161 | 1.109791e-06 |
| model.layers.2.self_attn.k_proj | k_proj | 3.0147 | 2.6346 | 2.5936 | 2.6415 | 0.0410 | 3.610456e-06 |
| model.layers.2.self_attn.v_proj | v_proj | 3.0152 | 2.6483 | 2.6072 | 2.6516 | 0.0410 | 2.998265e-07 |
| model.layers.2.self_attn.o_proj | o_proj | 3.0156 | 2.8896 | 2.8735 | 2.9230 | 0.0161 | 1.851148e-09 |
| model.layers.2.mlp.gate_proj | gate_proj | 3.0163 | 2.9317 | 2.9218 | 2.9739 | 0.0099 | 3.086547e-07 |
| model.layers.2.mlp.up_proj | up_proj | 3.0180 | 2.9282 | 2.9183 | 2.9761 | 0.0099 | 1.912687e-07 |
| model.layers.2.mlp.down_proj | down_proj | 3.0198 | 2.9279 | 2.9177 | 2.9662 | 0.0103 | 1.932173e-09 |
| model.layers.3.self_attn.q_proj | q_proj | 3.0218 | 2.8401 | 2.8240 | 2.8632 | 0.0161 | 1.106754e-06 |
| model.layers.3.self_attn.k_proj | k_proj | 3.0228 | 2.7175 | 2.6764 | 2.7155 | 0.0410 | 2.619530e-06 |
| model.layers.3.self_attn.v_proj | v_proj | 3.0232 | 2.7126 | 2.6716 | 2.7219 | 0.0410 | 3.357793e-07 |
| model.layers.3.self_attn.o_proj | o_proj | 3.0236 | 2.8928 | 2.8767 | 2.9279 | 0.0161 | 4.020936e-09 |
| model.layers.3.mlp.gate_proj | gate_proj | 3.0243 | 2.9253 | 2.9154 | 2.9649 | 0.0099 | 4.404494e-07 |
| model.layers.3.mlp.up_proj | up_proj | 3.0265 | 2.9383 | 2.9284 | 2.9870 | 0.0099 | 2.143297e-07 |
| model.layers.3.mlp.down_proj | down_proj | 3.0284 | 2.9399 | 2.9296 | 2.9782 | 0.0103 | 2.790891e-09 |
| model.layers.4.self_attn.q_proj | q_proj | 3.0305 | 2.8959 | 2.8797 | 2.9200 | 0.0161 | 9.186348e-07 |
| model.layers.4.self_attn.k_proj | k_proj | 3.0313 | 2.6754 | 2.6344 | 2.6755 | 0.0410 | 2.648172e-06 |
| model.layers.4.self_attn.v_proj | v_proj | 3.0318 | 2.7297 | 2.6887 | 2.7399 | 0.0410 | 2.922774e-07 |
| model.layers.4.self_attn.o_proj | o_proj | 3.0322 | 2.9006 | 2.8845 | 2.9260 | 0.0161 | 5.375870e-09 |
| model.layers.4.mlp.gate_proj | gate_proj | 3.0330 | 2.9487 | 2.9388 | 2.9891 | 0.0099 | 4.424266e-07 |
| model.layers.4.mlp.up_proj | up_proj | 3.0350 | 2.9351 | 2.9253 | 2.9827 | 0.0099 | 2.039283e-07 |
| model.layers.4.mlp.down_proj | down_proj | 3.0374 | 2.9471 | 2.9368 | 2.9860 | 0.0103 | 3.066887e-09 |
| model.layers.5.self_attn.q_proj | q_proj | 3.0397 | 2.9408 | 2.9247 | 2.9714 | 0.0161 | 9.394417e-07 |
| model.layers.5.self_attn.k_proj | k_proj | 3.0403 | 2.6039 | 2.5629 | 2.6180 | 0.0410 | 3.565116e-06 |
| model.layers.5.self_attn.v_proj | v_proj | 3.0410 | 2.6597 | 2.6187 | 2.6625 | 0.0410 | 2.695019e-07 |
| model.layers.5.self_attn.o_proj | o_proj | 3.0416 | 2.8984 | 2.8823 | 2.9304 | 0.0161 | 5.934337e-09 |
| model.layers.5.mlp.gate_proj | gate_proj | 3.0425 | 2.9595 | 2.9496 | 2.9995 | 0.0099 | 3.911109e-07 |
| model.layers.5.mlp.up_proj | up_proj | 3.0447 | 2.9565 | 2.9466 | 3.0043 | 0.0099 | 2.085160e-07 |
| model.layers.5.mlp.down_proj | down_proj | 3.0471 | 2.9618 | 2.9515 | 3.0002 | 0.0103 | 3.381690e-09 |
| model.layers.6.self_attn.q_proj | q_proj | 3.0494 | 2.9362 | 2.9201 | 2.9660 | 0.0161 | 8.675922e-07 |
| model.layers.6.self_attn.k_proj | k_proj | 3.0502 | 2.6099 | 2.5689 | 2.6202 | 0.0410 | 3.558145e-06 |
| model.layers.6.self_attn.v_proj | v_proj | 3.0510 | 2.6801 | 2.6391 | 2.6874 | 0.0410 | 3.545803e-07 |
| model.layers.6.self_attn.o_proj | o_proj | 3.0516 | 2.9110 | 2.8949 | 2.9431 | 0.0161 | 8.552745e-09 |
| model.layers.6.mlp.gate_proj | gate_proj | 3.0526 | 2.9458 | 2.9360 | 2.9846 | 0.0099 | 3.911234e-07 |
| model.layers.6.mlp.up_proj | up_proj | 3.0557 | 2.9724 | 2.9625 | 3.0212 | 0.0099 | 2.070244e-07 |
| model.layers.6.mlp.down_proj | down_proj | 3.0582 | 2.9709 | 2.9607 | 3.0094 | 0.0103 | 3.521678e-09 |
| model.layers.7.self_attn.q_proj | q_proj | 3.0608 | 2.9555 | 2.9394 | 2.9848 | 0.0161 | 9.822050e-07 |
| model.layers.7.self_attn.k_proj | k_proj | 3.0617 | 2.7288 | 2.6878 | 2.7277 | 0.0410 | 3.237711e-06 |
| model.layers.7.self_attn.v_proj | v_proj | 3.0623 | 2.6867 | 2.6457 | 2.6937 | 0.0410 | 4.126225e-07 |
| model.layers.7.self_attn.o_proj | o_proj | 3.0630 | 2.9253 | 2.9092 | 2.9540 | 0.0161 | 7.238559e-09 |
| model.layers.7.mlp.gate_proj | gate_proj | 3.0641 | 2.9719 | 2.9620 | 3.0107 | 0.0099 | 3.380325e-07 |
| model.layers.7.mlp.up_proj | up_proj | 3.0671 | 2.9711 | 2.9612 | 3.0180 | 0.0099 | 2.142666e-07 |
| model.layers.7.mlp.down_proj | down_proj | 3.0703 | 2.9831 | 2.9729 | 3.0221 | 0.0103 | 3.782564e-09 |
| model.layers.8.self_attn.q_proj | q_proj | 3.0733 | 2.8911 | 2.8750 | 2.9193 | 0.0161 | 1.074961e-06 |
| model.layers.8.self_attn.k_proj | k_proj | 3.0749 | 2.7875 | 2.7465 | 2.7836 | 0.0410 | 3.196050e-06 |
| model.layers.8.self_attn.v_proj | v_proj | 3.0755 | 2.7262 | 2.6852 | 2.7360 | 0.0410 | 3.643631e-07 |
| model.layers.8.self_attn.o_proj | o_proj | 3.0763 | 2.9383 | 2.9222 | 2.9706 | 0.0161 | 9.909117e-09 |
| model.layers.8.mlp.gate_proj | gate_proj | 3.0775 | 2.9604 | 2.9505 | 2.9981 | 0.0099 | 3.773214e-07 |
| model.layers.8.mlp.up_proj | up_proj | 3.0817 | 2.9940 | 2.9841 | 3.0407 | 0.0099 | 2.329168e-07 |
| model.layers.8.mlp.down_proj | down_proj | 3.0851 | 2.9961 | 2.9859 | 3.0360 | 0.0103 | 4.747395e-09 |
| model.layers.9.self_attn.q_proj | q_proj | 3.0886 | 2.9566 | 2.9405 | 2.9883 | 0.0161 | 1.197680e-06 |
| model.layers.9.self_attn.k_proj | k_proj | 3.0899 | 2.7286 | 2.6876 | 2.7275 | 0.0410 | 3.099314e-06 |
| model.layers.9.self_attn.v_proj | v_proj | 3.0908 | 2.7838 | 2.7428 | 2.7915 | 0.0410 | 3.841858e-07 |
| model.layers.9.self_attn.o_proj | o_proj | 3.0916 | 2.9526 | 2.9365 | 2.9828 | 0.0161 | 1.200194e-08 |
| model.layers.9.mlp.gate_proj | gate_proj | 3.0930 | 3.0113 | 3.0014 | 3.0511 | 0.0099 | 3.755683e-07 |
| model.layers.9.mlp.up_proj | up_proj | 3.0964 | 2.9956 | 2.9857 | 3.0413 | 0.0099 | 2.427569e-07 |
| model.layers.9.mlp.down_proj | down_proj | 3.1008 | 3.0124 | 3.0021 | 3.0535 | 0.0103 | 5.434143e-09 |
| model.layers.10.self_attn.q_proj | q_proj | 3.1049 | 2.9334 | 2.9173 | 2.9626 | 0.0161 | 1.132150e-06 |
| model.layers.10.self_attn.k_proj | k_proj | 3.1069 | 2.6791 | 2.6381 | 2.6802 | 0.0410 | 3.321591e-06 |
| model.layers.10.self_attn.v_proj | v_proj | 3.1081 | 2.7498 | 2.7088 | 2.7492 | 0.0410 | 3.565941e-07 |
| model.layers.10.self_attn.o_proj | o_proj | 3.1092 | 2.9641 | 2.9480 | 2.9953 | 0.0161 | 9.182765e-09 |
| model.layers.10.mlp.gate_proj | gate_proj | 3.1109 | 3.0605 | 3.0506 | 3.1050 | 0.0099 | 4.036262e-07 |
| model.layers.10.mlp.up_proj | up_proj | 3.1134 | 3.0116 | 3.0017 | 3.0600 | 0.0099 | 2.808899e-07 |
| model.layers.10.mlp.down_proj | down_proj | 3.1187 | 3.0283 | 3.0180 | 3.0697 | 0.0103 | 6.545342e-09 |
| model.layers.11.self_attn.q_proj | q_proj | 3.1237 | 2.9787 | 2.9626 | 3.0133 | 0.0161 | 1.094649e-06 |
| model.layers.11.self_attn.k_proj | k_proj | 3.1258 | 2.7321 | 2.6910 | 2.7284 | 0.0410 | 3.300187e-06 |
| model.layers.11.self_attn.v_proj | v_proj | 3.1271 | 2.7610 | 2.7200 | 2.7599 | 0.0410 | 3.694659e-07 |
| model.layers.11.self_attn.o_proj | o_proj | 3.1284 | 2.9797 | 2.9636 | 3.0117 | 0.0161 | 6.247354e-09 |
| model.layers.11.mlp.gate_proj | gate_proj | 3.1306 | 3.0439 | 3.0340 | 3.0874 | 0.0099 | 4.482308e-07 |
| model.layers.11.mlp.up_proj | up_proj | 3.1358 | 3.0480 | 3.0381 | 3.0978 | 0.0099 | 2.960815e-07 |
| model.layers.11.mlp.down_proj | down_proj | 3.1415 | 3.0506 | 3.0403 | 3.0938 | 0.0103 | 6.939306e-09 |
| model.layers.12.self_attn.q_proj | q_proj | 3.1477 | 3.0100 | 2.9939 | 3.0461 | 0.0161 | 8.754708e-07 |
| model.layers.12.self_attn.k_proj | k_proj | 3.1502 | 2.7411 | 2.7001 | 2.7468 | 0.0410 | 2.537482e-06 |
| model.layers.12.self_attn.v_proj | v_proj | 3.1520 | 2.8062 | 2.7651 | 2.8037 | 0.0410 | 3.368151e-07 |
| model.layers.12.self_attn.o_proj | o_proj | 3.1535 | 3.0051 | 2.9889 | 3.0416 | 0.0161 | 6.475628e-09 |
| model.layers.12.mlp.gate_proj | gate_proj | 3.1562 | 3.0730 | 3.0631 | 3.1200 | 0.0099 | 4.180486e-07 |
| model.layers.12.mlp.up_proj | up_proj | 3.1626 | 3.0717 | 3.0618 | 3.1166 | 0.0099 | 2.992701e-07 |
| model.layers.12.mlp.down_proj | down_proj | 3.1703 | 3.0812 | 3.0710 | 3.1279 | 0.0103 | 7.881062e-09 |
| model.layers.13.self_attn.q_proj | q_proj | 3.1785 | 3.0537 | 3.0375 | 3.0948 | 0.0161 | 9.533938e-07 |
| model.layers.13.self_attn.k_proj | k_proj | 3.1814 | 2.7585 | 2.7175 | 2.7552 | 0.0410 | 2.592474e-06 |
| model.layers.13.self_attn.v_proj | v_proj | 3.1839 | 2.8265 | 2.7855 | 2.8254 | 0.0410 | 5.267811e-07 |
| model.layers.13.self_attn.o_proj | o_proj | 3.1860 | 3.0439 | 3.0278 | 3.0830 | 0.0161 | 7.219176e-09 |
| model.layers.13.mlp.gate_proj | gate_proj | 3.1895 | 3.1116 | 3.1018 | 3.1514 | 0.0099 | 4.155279e-07 |
| model.layers.13.mlp.up_proj | up_proj | 3.1979 | 3.1057 | 3.0958 | 3.1445 | 0.0099 | 3.317764e-07 |
| model.layers.13.mlp.down_proj | down_proj | 3.2091 | 3.1208 | 3.1105 | 3.1616 | 0.0103 | 1.024751e-08 |
| model.layers.14.self_attn.q_proj | q_proj | 3.2213 | 3.0917 | 3.0756 | 3.1386 | 0.0161 | 7.495252e-07 |
| model.layers.14.self_attn.k_proj | k_proj | 3.2259 | 2.8564 | 2.8154 | 2.8607 | 0.0410 | 1.450196e-06 |
| model.layers.14.self_attn.v_proj | v_proj | 3.2292 | 2.8345 | 2.7935 | 2.8260 | 0.0410 | 7.877876e-07 |
| model.layers.14.self_attn.o_proj | o_proj | 3.2328 | 3.0854 | 3.0693 | 3.1222 | 0.0161 | 1.294524e-08 |
| model.layers.14.mlp.gate_proj | gate_proj | 3.2384 | 3.1372 | 3.1273 | 3.1743 | 0.0099 | 4.744675e-07 |
| model.layers.14.mlp.up_proj | up_proj | 3.2564 | 3.1741 | 3.1642 | 3.2029 | 0.0099 | 3.367545e-07 |
| model.layers.14.mlp.down_proj | down_proj | 3.2741 | 3.1847 | 3.1745 | 3.2165 | 0.0103 | 1.159747e-08 |
| model.layers.15.self_attn.q_proj | q_proj | 3.2988 | 3.1391 | 3.1230 | 3.1855 | 0.0161 | 6.695446e-07 |
| model.layers.15.self_attn.k_proj | k_proj | 3.3106 | 2.9219 | 2.8809 | 2.9177 | 0.0410 | 1.493759e-06 |
| model.layers.15.self_attn.v_proj | v_proj | 3.3180 | 3.0014 | 2.9604 | 3.0009 | 0.0410 | 6.635193e-07 |
| model.layers.15.self_attn.o_proj | o_proj | 3.3241 | 3.1737 | 3.1576 | 3.2169 | 0.0161 | 2.368635e-08 |
| model.layers.15.mlp.gate_proj | gate_proj | 3.3366 | 3.2479 | 3.2380 | 3.2752 | 0.0099 | 4.229352e-07 |
| model.layers.15.mlp.up_proj | up_proj | 3.3810 | 3.2949 | 3.2850 | 3.3163 | 0.0099 | 3.190187e-07 |
| model.layers.15.mlp.down_proj | down_proj | 3.4670 | 3.3751 | 3.3648 | 3.4036 | 0.0103 | 1.745963e-08 |
