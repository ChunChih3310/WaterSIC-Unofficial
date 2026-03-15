# WaterSIC Run Report

- Timestamp: `2026-03-15T12:57:26Z`
- Git commit: `318001582b89194a7bd6b733043497a96fca5e9f`
- Environment: `watersic`
- Device: `cuda`
- Model: `meta-llama/Llama-3.2-1B`
- Model revision: `main`
- Tokenizer: `meta-llama/Llama-3.2-1B`
- Sequence length: `2048`
- Calibration sequences: `32`
- Target global bitwidth: `3.0000`
- Achieved global bitwidth: `2.9984`
- Raw average bitwidth: `8.4246`
- Entropy average bitwidth: `2.9865`
- Huffman average bitwidth: `3.0373`
- Side-information overhead: `0.0119`
- Perplexity: `11.780607603943109`

## Layer Summary

| Layer | Kind | Target | Achieved | Entropy | Huffman | Side Info | Weighted Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model.layers.0.self_attn.q_proj | q_proj | 3.0000 | 2.8885 | 2.8724 | 2.9319 | 0.0161 | 2.271092e-07 |
| model.layers.0.self_attn.k_proj | k_proj | 3.0005 | 2.7063 | 2.6652 | 2.7300 | 0.0410 | 5.225797e-07 |
| model.layers.0.self_attn.v_proj | v_proj | 3.0008 | 2.6216 | 2.5805 | 2.6456 | 0.0410 | 1.932079e-08 |
| model.layers.0.self_attn.o_proj | o_proj | 3.0012 | 2.8656 | 2.8495 | 2.8840 | 0.0161 | 2.980589e-10 |
| model.layers.0.mlp.gate_proj | gate_proj | 3.0018 | 2.9111 | 2.9012 | 2.9571 | 0.0099 | 1.580005e-07 |
| model.layers.0.mlp.up_proj | up_proj | 3.0034 | 2.9165 | 2.9066 | 2.9641 | 0.0099 | 1.237946e-07 |
| model.layers.0.mlp.down_proj | down_proj | 3.0050 | 2.9128 | 2.9025 | 2.9504 | 0.0103 | 7.148361e-10 |
| model.layers.1.self_attn.q_proj | q_proj | 3.0067 | 2.8584 | 2.8423 | 2.8834 | 0.0162 | 4.691998e-07 |
| model.layers.1.self_attn.k_proj | k_proj | 3.0074 | 2.5636 | 2.5225 | 2.5926 | 0.0411 | 1.653160e-06 |
| model.layers.1.self_attn.v_proj | v_proj | 3.0079 | 2.6139 | 2.5728 | 2.6236 | 0.0411 | 8.496151e-08 |
| model.layers.1.self_attn.o_proj | o_proj | 3.0083 | 2.8822 | 2.8661 | 2.9047 | 0.0161 | 1.587690e-09 |
| model.layers.1.mlp.gate_proj | gate_proj | 3.0089 | 2.9202 | 2.9103 | 2.9657 | 0.0099 | 2.321875e-07 |
| model.layers.1.mlp.up_proj | up_proj | 3.0106 | 2.9217 | 2.9118 | 2.9696 | 0.0099 | 1.665525e-07 |
| model.layers.1.mlp.down_proj | down_proj | 3.0123 | 2.9057 | 2.8954 | 2.9433 | 0.0103 | 1.327301e-09 |
| model.layers.2.self_attn.q_proj | q_proj | 3.0144 | 2.9234 | 2.9073 | 2.9535 | 0.0161 | 7.198652e-07 |
| model.layers.2.self_attn.k_proj | k_proj | 3.0149 | 2.6352 | 2.5942 | 2.6422 | 0.0410 | 2.345973e-06 |
| model.layers.2.self_attn.v_proj | v_proj | 3.0153 | 2.6477 | 2.6067 | 2.6499 | 0.0410 | 1.943074e-07 |
| model.layers.2.self_attn.o_proj | o_proj | 3.0158 | 2.8861 | 2.8700 | 2.9181 | 0.0161 | 2.086887e-09 |
| model.layers.2.mlp.gate_proj | gate_proj | 3.0164 | 2.9326 | 2.9227 | 2.9767 | 0.0099 | 3.434885e-07 |
| model.layers.2.mlp.up_proj | up_proj | 3.0182 | 2.9283 | 2.9184 | 2.9775 | 0.0099 | 2.126070e-07 |
| model.layers.2.mlp.down_proj | down_proj | 3.0200 | 2.9267 | 2.9164 | 2.9652 | 0.0103 | 2.527483e-09 |
| model.layers.3.self_attn.q_proj | q_proj | 3.0220 | 2.8418 | 2.8257 | 2.8651 | 0.0161 | 8.788069e-07 |
| model.layers.3.self_attn.k_proj | k_proj | 3.0230 | 2.7193 | 2.6783 | 2.7172 | 0.0410 | 2.086581e-06 |
| model.layers.3.self_attn.v_proj | v_proj | 3.0234 | 2.7131 | 2.6721 | 2.7221 | 0.0410 | 2.685957e-07 |
| model.layers.3.self_attn.o_proj | o_proj | 3.0238 | 2.8912 | 2.8751 | 2.9257 | 0.0161 | 4.671562e-09 |
| model.layers.3.mlp.gate_proj | gate_proj | 3.0245 | 2.9253 | 2.9154 | 2.9657 | 0.0099 | 4.850805e-07 |
| model.layers.3.mlp.up_proj | up_proj | 3.0267 | 2.9386 | 2.9287 | 2.9883 | 0.0099 | 2.352402e-07 |
| model.layers.3.mlp.down_proj | down_proj | 3.0287 | 2.9367 | 2.9265 | 2.9754 | 0.0103 | 3.509485e-09 |
| model.layers.4.self_attn.q_proj | q_proj | 3.0308 | 2.8969 | 2.8808 | 2.9212 | 0.0161 | 7.608070e-07 |
| model.layers.4.self_attn.k_proj | k_proj | 3.0315 | 2.6751 | 2.6340 | 2.6753 | 0.0410 | 2.206924e-06 |
| model.layers.4.self_attn.v_proj | v_proj | 3.0321 | 2.7290 | 2.6880 | 2.7387 | 0.0410 | 2.441761e-07 |
| model.layers.4.self_attn.o_proj | o_proj | 3.0325 | 2.8993 | 2.8832 | 2.9256 | 0.0161 | 6.050696e-09 |
| model.layers.4.mlp.gate_proj | gate_proj | 3.0333 | 2.9489 | 2.9390 | 2.9902 | 0.0099 | 4.880970e-07 |
| model.layers.4.mlp.up_proj | up_proj | 3.0353 | 2.9357 | 2.9259 | 2.9842 | 0.0099 | 2.240658e-07 |
| model.layers.4.mlp.down_proj | down_proj | 3.0377 | 2.9472 | 2.9370 | 2.9857 | 0.0103 | 3.779711e-09 |
| model.layers.5.self_attn.q_proj | q_proj | 3.0400 | 2.9412 | 2.9251 | 2.9718 | 0.0161 | 8.050317e-07 |
| model.layers.5.self_attn.k_proj | k_proj | 3.0406 | 2.6067 | 2.5657 | 2.6205 | 0.0410 | 3.052473e-06 |
| model.layers.5.self_attn.v_proj | v_proj | 3.0413 | 2.6591 | 2.6181 | 2.6614 | 0.0410 | 2.317681e-07 |
| model.layers.5.self_attn.o_proj | o_proj | 3.0419 | 2.8979 | 2.8817 | 2.9304 | 0.0161 | 6.632950e-09 |
| model.layers.5.mlp.gate_proj | gate_proj | 3.0428 | 2.9598 | 2.9500 | 3.0006 | 0.0099 | 4.345695e-07 |
| model.layers.5.mlp.up_proj | up_proj | 3.0450 | 2.9572 | 2.9473 | 3.0076 | 0.0099 | 2.307809e-07 |
| model.layers.5.mlp.down_proj | down_proj | 3.0474 | 2.9615 | 2.9512 | 2.9994 | 0.0103 | 4.150765e-09 |
| model.layers.6.self_attn.q_proj | q_proj | 3.0497 | 2.9372 | 2.9211 | 2.9666 | 0.0161 | 7.844350e-07 |
| model.layers.6.self_attn.k_proj | k_proj | 3.0505 | 2.6111 | 2.5701 | 2.6215 | 0.0410 | 3.227346e-06 |
| model.layers.6.self_attn.v_proj | v_proj | 3.0513 | 2.6801 | 2.6391 | 2.6874 | 0.0410 | 3.231524e-07 |
| model.layers.6.self_attn.o_proj | o_proj | 3.0519 | 2.9098 | 2.8937 | 2.9423 | 0.0161 | 9.778717e-09 |
| model.layers.6.mlp.gate_proj | gate_proj | 3.0529 | 2.9458 | 2.9359 | 2.9852 | 0.0099 | 4.363930e-07 |
| model.layers.6.mlp.up_proj | up_proj | 3.0560 | 2.9721 | 2.9622 | 3.0236 | 0.0099 | 2.305133e-07 |
| model.layers.6.mlp.down_proj | down_proj | 3.0585 | 2.9717 | 2.9615 | 3.0107 | 0.0103 | 4.228719e-09 |
| model.layers.7.self_attn.q_proj | q_proj | 3.0612 | 2.9567 | 2.9406 | 2.9864 | 0.0161 | 9.073178e-07 |
| model.layers.7.self_attn.k_proj | k_proj | 3.0620 | 2.7307 | 2.6897 | 2.7298 | 0.0410 | 2.996899e-06 |
| model.layers.7.self_attn.v_proj | v_proj | 3.0626 | 2.6856 | 2.6445 | 2.6921 | 0.0410 | 3.841240e-07 |
| model.layers.7.self_attn.o_proj | o_proj | 3.0633 | 2.9258 | 2.9097 | 2.9571 | 0.0161 | 8.510892e-09 |
| model.layers.7.mlp.gate_proj | gate_proj | 3.0644 | 2.9727 | 2.9629 | 3.0118 | 0.0099 | 3.788547e-07 |
| model.layers.7.mlp.up_proj | up_proj | 3.0674 | 2.9716 | 2.9617 | 3.0213 | 0.0099 | 2.394507e-07 |
| model.layers.7.mlp.down_proj | down_proj | 3.0706 | 2.9833 | 2.9731 | 3.0216 | 0.0103 | 4.600002e-09 |
| model.layers.8.self_attn.q_proj | q_proj | 3.0736 | 2.8892 | 2.8731 | 2.9173 | 0.0161 | 1.005544e-06 |
| model.layers.8.self_attn.k_proj | k_proj | 3.0752 | 2.7896 | 2.7485 | 2.7858 | 0.0410 | 2.974726e-06 |
| model.layers.8.self_attn.v_proj | v_proj | 3.0758 | 2.7314 | 2.6903 | 2.7415 | 0.0410 | 3.376067e-07 |
| model.layers.8.self_attn.o_proj | o_proj | 3.0765 | 2.9398 | 2.9237 | 2.9726 | 0.0161 | 1.180009e-08 |
| model.layers.8.mlp.gate_proj | gate_proj | 3.0777 | 2.9606 | 2.9507 | 2.9983 | 0.0099 | 4.243808e-07 |
| model.layers.8.mlp.up_proj | up_proj | 3.0820 | 2.9947 | 2.9848 | 3.0439 | 0.0099 | 2.611287e-07 |
| model.layers.8.mlp.down_proj | down_proj | 3.0853 | 2.9963 | 2.9860 | 3.0353 | 0.0103 | 5.951165e-09 |
| model.layers.9.self_attn.q_proj | q_proj | 3.0888 | 2.9578 | 2.9416 | 2.9896 | 0.0161 | 1.096251e-06 |
| model.layers.9.self_attn.k_proj | k_proj | 3.0902 | 2.7297 | 2.6887 | 2.7282 | 0.0410 | 2.834276e-06 |
| model.layers.9.self_attn.v_proj | v_proj | 3.0911 | 2.7842 | 2.7432 | 2.7923 | 0.0410 | 3.542739e-07 |
| model.layers.9.self_attn.o_proj | o_proj | 3.0918 | 2.9524 | 2.9363 | 2.9830 | 0.0161 | 1.434756e-08 |
| model.layers.9.mlp.gate_proj | gate_proj | 3.0932 | 3.0113 | 3.0014 | 3.0511 | 0.0099 | 4.275176e-07 |
| model.layers.9.mlp.up_proj | up_proj | 3.0967 | 2.9957 | 2.9859 | 3.0439 | 0.0099 | 2.754139e-07 |
| model.layers.9.mlp.down_proj | down_proj | 3.1011 | 3.0119 | 3.0017 | 3.0525 | 0.0103 | 6.936120e-09 |
| model.layers.10.self_attn.q_proj | q_proj | 3.1052 | 2.9333 | 2.9172 | 2.9628 | 0.0161 | 9.673122e-07 |
| model.layers.10.self_attn.k_proj | k_proj | 3.1072 | 2.6807 | 2.6397 | 2.6817 | 0.0410 | 2.833510e-06 |
| model.layers.10.self_attn.v_proj | v_proj | 3.1085 | 2.7503 | 2.7093 | 2.7504 | 0.0410 | 3.037268e-07 |
| model.layers.10.self_attn.o_proj | o_proj | 3.1095 | 2.9632 | 2.9470 | 2.9952 | 0.0161 | 1.039846e-08 |
| model.layers.10.mlp.gate_proj | gate_proj | 3.1112 | 3.0609 | 3.0510 | 3.1060 | 0.0099 | 4.661426e-07 |
| model.layers.10.mlp.up_proj | up_proj | 3.1137 | 3.0118 | 3.0019 | 3.0638 | 0.0099 | 3.228751e-07 |
| model.layers.10.mlp.down_proj | down_proj | 3.1191 | 3.0292 | 3.0190 | 3.0709 | 0.0103 | 8.557462e-09 |
| model.layers.11.self_attn.q_proj | q_proj | 3.1240 | 2.9798 | 2.9637 | 3.0144 | 0.0161 | 8.644864e-07 |
| model.layers.11.self_attn.k_proj | k_proj | 3.1260 | 2.7339 | 2.6928 | 2.7298 | 0.0410 | 2.598199e-06 |
| model.layers.11.self_attn.v_proj | v_proj | 3.1274 | 2.7595 | 2.7185 | 2.7584 | 0.0410 | 2.921723e-07 |
| model.layers.11.self_attn.o_proj | o_proj | 3.1287 | 2.9795 | 2.9634 | 3.0124 | 0.0161 | 7.424925e-09 |
| model.layers.11.mlp.gate_proj | gate_proj | 3.1308 | 3.0432 | 3.0333 | 3.0875 | 0.0099 | 5.223703e-07 |
| model.layers.11.mlp.up_proj | up_proj | 3.1361 | 3.0485 | 3.0386 | 3.0990 | 0.0099 | 3.434423e-07 |
| model.layers.11.mlp.down_proj | down_proj | 3.1418 | 3.0519 | 3.0416 | 3.0960 | 0.0103 | 9.291675e-09 |
| model.layers.12.self_attn.q_proj | q_proj | 3.1480 | 3.0083 | 2.9922 | 3.0447 | 0.0161 | 6.721939e-07 |
| model.layers.12.self_attn.k_proj | k_proj | 3.1505 | 2.7406 | 2.6996 | 2.7452 | 0.0410 | 1.950215e-06 |
| model.layers.12.self_attn.v_proj | v_proj | 3.1523 | 2.8062 | 2.7652 | 2.8043 | 0.0410 | 2.576575e-07 |
| model.layers.12.self_attn.o_proj | o_proj | 3.1538 | 3.0074 | 2.9913 | 3.0443 | 0.0161 | 7.497222e-09 |
| model.layers.12.mlp.gate_proj | gate_proj | 3.1564 | 3.0737 | 3.0638 | 3.1197 | 0.0099 | 4.924526e-07 |
| model.layers.12.mlp.up_proj | up_proj | 3.1629 | 3.0717 | 3.0618 | 3.1176 | 0.0099 | 3.513326e-07 |
| model.layers.12.mlp.down_proj | down_proj | 3.1705 | 3.0820 | 3.0718 | 3.1267 | 0.0103 | 1.098227e-08 |
| model.layers.13.self_attn.q_proj | q_proj | 3.1787 | 3.0536 | 3.0375 | 3.0949 | 0.0161 | 6.885321e-07 |
| model.layers.13.self_attn.k_proj | k_proj | 3.1816 | 2.7582 | 2.7172 | 2.7541 | 0.0410 | 1.875442e-06 |
| model.layers.13.self_attn.v_proj | v_proj | 3.1841 | 2.8252 | 2.7842 | 2.8240 | 0.0410 | 3.806637e-07 |
| model.layers.13.self_attn.o_proj | o_proj | 3.1863 | 3.0420 | 3.0259 | 3.0810 | 0.0161 | 8.965682e-09 |
| model.layers.13.mlp.gate_proj | gate_proj | 3.1898 | 3.1122 | 3.1023 | 3.1512 | 0.0099 | 4.921887e-07 |
| model.layers.13.mlp.up_proj | up_proj | 3.1982 | 3.1058 | 3.0959 | 3.1456 | 0.0099 | 3.920671e-07 |
| model.layers.13.mlp.down_proj | down_proj | 3.2094 | 3.1211 | 3.1109 | 3.1577 | 0.0103 | 1.460898e-08 |
| model.layers.14.self_attn.q_proj | q_proj | 3.2215 | 3.0914 | 3.0752 | 3.1382 | 0.0161 | 5.998114e-07 |
| model.layers.14.self_attn.k_proj | k_proj | 3.2262 | 2.8539 | 2.8129 | 2.8565 | 0.0410 | 1.171484e-06 |
| model.layers.14.self_attn.v_proj | v_proj | 3.2295 | 2.8353 | 2.7943 | 2.8264 | 0.0410 | 6.288965e-07 |
| model.layers.14.self_attn.o_proj | o_proj | 3.2331 | 3.0867 | 3.0706 | 3.1223 | 0.0161 | 1.706045e-08 |
| model.layers.14.mlp.gate_proj | gate_proj | 3.2386 | 3.1377 | 3.1278 | 3.1737 | 0.0099 | 5.700333e-07 |
| model.layers.14.mlp.up_proj | up_proj | 3.2566 | 3.1740 | 3.1642 | 3.2042 | 0.0099 | 4.032557e-07 |
| model.layers.14.mlp.down_proj | down_proj | 3.2744 | 3.1839 | 3.1736 | 3.2120 | 0.0103 | 1.697666e-08 |
| model.layers.15.self_attn.q_proj | q_proj | 3.2994 | 3.1400 | 3.1239 | 3.1863 | 0.0161 | 5.603621e-07 |
| model.layers.15.self_attn.k_proj | k_proj | 3.3112 | 2.9237 | 2.8827 | 2.9188 | 0.0410 | 1.253340e-06 |
| model.layers.15.self_attn.v_proj | v_proj | 3.3185 | 3.0017 | 2.9607 | 3.0011 | 0.0410 | 5.581652e-07 |
| model.layers.15.self_attn.o_proj | o_proj | 3.3246 | 3.1735 | 3.1574 | 3.2176 | 0.0161 | 3.007127e-08 |
| model.layers.15.mlp.gate_proj | gate_proj | 3.3372 | 3.2482 | 3.2383 | 3.2744 | 0.0099 | 5.131254e-07 |
| model.layers.15.mlp.up_proj | up_proj | 3.3818 | 3.2957 | 3.2858 | 3.3188 | 0.0099 | 3.852994e-07 |
| model.layers.15.mlp.down_proj | down_proj | 3.4678 | 3.3750 | 3.3648 | 3.3980 | 0.0103 | 2.554641e-08 |
