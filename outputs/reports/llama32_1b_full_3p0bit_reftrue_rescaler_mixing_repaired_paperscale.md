# WaterSIC Run Report

- Timestamp: `2026-03-18T05:15:17Z`
- Git commit: `cafd15ceb110e89de83a324f94c05c7cd6775f1f`
- Environment: `watersic`
- Device: `cuda`
- Model: `meta-llama/Llama-3.2-1B`
- Model revision: `main`
- Tokenizer: `meta-llama/Llama-3.2-1B`
- Sequence length: `2048`
- Calibration sequences: `1188`
- Target global bitwidth: `3.0000`
- Achieved global bitwidth: `2.9984`
- Raw average bitwidth: `8.3761`
- Entropy average bitwidth: `2.9864`
- Huffman average bitwidth: `3.0379`
- Huffman shortest symbol length: `1`
- Huffman longest symbol length: `25`
- Side-information overhead: `0.0119`
- Perplexity: `10.6030816224561`

## Layer Summary

| Layer | Kind | Target | Achieved | Entropy | Huffman | Huff Min | Huff Max | Side Info | Weighted Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model.layers.0.self_attn.q_proj | q_proj | 3.0000 | 2.8887 | 2.8725 | 2.9326 | 2 | 22 | 0.0161 | 4.694733e-07 |
| model.layers.0.self_attn.k_proj | k_proj | 3.0005 | 2.7036 | 2.6625 | 2.7289 | 1 | 20 | 0.0410 | 1.075672e-06 |
| model.layers.0.self_attn.v_proj | v_proj | 3.0008 | 2.6191 | 2.5781 | 2.6472 | 2 | 18 | 0.0410 | 3.940038e-08 |
| model.layers.0.self_attn.o_proj | o_proj | 3.0012 | 2.8660 | 2.8499 | 2.8889 | 1 | 22 | 0.0161 | 3.508499e-10 |
| model.layers.0.mlp.gate_proj | gate_proj | 3.0018 | 2.9110 | 2.9011 | 2.9580 | 2 | 24 | 0.0099 | 1.770481e-07 |
| model.layers.0.mlp.up_proj | up_proj | 3.0034 | 2.9167 | 2.9068 | 2.9652 | 2 | 25 | 0.0099 | 1.386178e-07 |
| model.layers.0.mlp.down_proj | down_proj | 3.0050 | 2.9133 | 2.9031 | 2.9546 | 2 | 23 | 0.0103 | 9.437522e-10 |
| model.layers.1.self_attn.q_proj | q_proj | 3.0067 | 2.8580 | 2.8418 | 2.8828 | 2 | 22 | 0.0162 | 8.507071e-07 |
| model.layers.1.self_attn.k_proj | k_proj | 3.0074 | 2.5628 | 2.5217 | 2.5891 | 2 | 19 | 0.0411 | 2.979060e-06 |
| model.layers.1.self_attn.v_proj | v_proj | 3.0079 | 2.6115 | 2.5704 | 2.6212 | 2 | 20 | 0.0411 | 1.536823e-07 |
| model.layers.1.self_attn.o_proj | o_proj | 3.0083 | 2.8816 | 2.8655 | 2.9038 | 2 | 23 | 0.0161 | 1.571348e-09 |
| model.layers.1.mlp.gate_proj | gate_proj | 3.0089 | 2.9203 | 2.9104 | 2.9668 | 2 | 24 | 0.0099 | 2.536700e-07 |
| model.layers.1.mlp.up_proj | up_proj | 3.0106 | 2.9218 | 2.9119 | 2.9708 | 2 | 24 | 0.0099 | 1.818421e-07 |
| model.layers.1.mlp.down_proj | down_proj | 3.0123 | 2.9050 | 2.8947 | 2.9447 | 2 | 23 | 0.0103 | 1.610779e-09 |
| model.layers.2.self_attn.q_proj | q_proj | 3.0144 | 2.9235 | 2.9074 | 2.9544 | 2 | 22 | 0.0161 | 1.158272e-06 |
| model.layers.2.self_attn.k_proj | k_proj | 3.0149 | 2.6359 | 2.5949 | 2.6418 | 2 | 20 | 0.0410 | 3.744202e-06 |
| model.layers.2.self_attn.v_proj | v_proj | 3.0153 | 2.6491 | 2.6081 | 2.6501 | 2 | 20 | 0.0410 | 3.093959e-07 |
| model.layers.2.self_attn.o_proj | o_proj | 3.0158 | 2.8837 | 2.8676 | 2.9151 | 2 | 22 | 0.0161 | 2.173011e-09 |
| model.layers.2.mlp.gate_proj | gate_proj | 3.0165 | 2.9323 | 2.9224 | 2.9773 | 2 | 22 | 0.0099 | 3.628861e-07 |
| model.layers.2.mlp.up_proj | up_proj | 3.0182 | 2.9279 | 2.9180 | 2.9778 | 2 | 24 | 0.0099 | 2.245314e-07 |
| model.layers.2.mlp.down_proj | down_proj | 3.0200 | 2.9253 | 2.9151 | 2.9650 | 2 | 23 | 0.0103 | 2.840904e-09 |
| model.layers.3.self_attn.q_proj | q_proj | 3.0221 | 2.8397 | 2.8236 | 2.8642 | 2 | 22 | 0.0161 | 1.220463e-06 |
| model.layers.3.self_attn.k_proj | k_proj | 3.0230 | 2.7196 | 2.6786 | 2.7179 | 2 | 19 | 0.0410 | 2.881325e-06 |
| model.layers.3.self_attn.v_proj | v_proj | 3.0234 | 2.7147 | 2.6737 | 2.7249 | 2 | 21 | 0.0410 | 3.686746e-07 |
| model.layers.3.self_attn.o_proj | o_proj | 3.0238 | 2.8890 | 2.8729 | 2.9231 | 2 | 21 | 0.0161 | 5.123306e-09 |
| model.layers.3.mlp.gate_proj | gate_proj | 3.0246 | 2.9254 | 2.9155 | 2.9662 | 2 | 23 | 0.0099 | 5.118719e-07 |
| model.layers.3.mlp.up_proj | up_proj | 3.0268 | 2.9384 | 2.9285 | 2.9888 | 2 | 24 | 0.0099 | 2.479789e-07 |
| model.layers.3.mlp.down_proj | down_proj | 3.0287 | 2.9350 | 2.9247 | 2.9744 | 2 | 24 | 0.0103 | 3.914370e-09 |
| model.layers.4.self_attn.q_proj | q_proj | 3.0309 | 2.8977 | 2.8816 | 2.9232 | 2 | 22 | 0.0161 | 1.088590e-06 |
| model.layers.4.self_attn.k_proj | k_proj | 3.0317 | 2.6747 | 2.6337 | 2.6752 | 2 | 19 | 0.0410 | 3.149818e-06 |
| model.layers.4.self_attn.v_proj | v_proj | 3.0322 | 2.7282 | 2.6872 | 2.7387 | 2 | 20 | 0.0410 | 3.498586e-07 |
| model.layers.4.self_attn.o_proj | o_proj | 3.0326 | 2.8988 | 2.8827 | 2.9259 | 2 | 23 | 0.0161 | 6.561060e-09 |
| model.layers.4.mlp.gate_proj | gate_proj | 3.0334 | 2.9493 | 2.9394 | 2.9910 | 2 | 23 | 0.0099 | 5.170084e-07 |
| model.layers.4.mlp.up_proj | up_proj | 3.0354 | 2.9356 | 2.9257 | 2.9847 | 2 | 25 | 0.0099 | 2.371872e-07 |
| model.layers.4.mlp.down_proj | down_proj | 3.0378 | 2.9457 | 2.9354 | 2.9848 | 2 | 23 | 0.0103 | 4.220562e-09 |
| model.layers.5.self_attn.q_proj | q_proj | 3.0402 | 2.9414 | 2.9253 | 2.9732 | 2 | 22 | 0.0161 | 1.127365e-06 |
| model.layers.5.self_attn.k_proj | k_proj | 3.0408 | 2.6083 | 2.5673 | 2.6203 | 2 | 19 | 0.0410 | 4.204181e-06 |
| model.layers.5.self_attn.v_proj | v_proj | 3.0415 | 2.6594 | 2.6184 | 2.6602 | 2 | 20 | 0.0410 | 3.233950e-07 |
| model.layers.5.self_attn.o_proj | o_proj | 3.0421 | 2.8970 | 2.8809 | 2.9311 | 2 | 22 | 0.0161 | 6.847432e-09 |
| model.layers.5.mlp.gate_proj | gate_proj | 3.0430 | 2.9595 | 2.9497 | 3.0006 | 2 | 24 | 0.0099 | 4.604320e-07 |
| model.layers.5.mlp.up_proj | up_proj | 3.0452 | 2.9570 | 2.9471 | 3.0081 | 2 | 24 | 0.0099 | 2.440401e-07 |
| model.layers.5.mlp.down_proj | down_proj | 3.0475 | 2.9618 | 2.9516 | 3.0005 | 2 | 24 | 0.0103 | 4.501603e-09 |
| model.layers.6.self_attn.q_proj | q_proj | 3.0499 | 2.9370 | 2.9209 | 2.9690 | 2 | 22 | 0.0161 | 1.045137e-06 |
| model.layers.6.self_attn.k_proj | k_proj | 3.0507 | 2.6091 | 2.5681 | 2.6188 | 2 | 19 | 0.0410 | 4.283943e-06 |
| model.layers.6.self_attn.v_proj | v_proj | 3.0515 | 2.6789 | 2.6379 | 2.6838 | 2 | 21 | 0.0410 | 4.286144e-07 |
| model.layers.6.self_attn.o_proj | o_proj | 3.0521 | 2.9101 | 2.8940 | 2.9425 | 2 | 22 | 0.0161 | 1.012336e-08 |
| model.layers.6.mlp.gate_proj | gate_proj | 3.0531 | 2.9459 | 2.9360 | 2.9854 | 2 | 24 | 0.0099 | 4.552931e-07 |
| model.layers.6.mlp.up_proj | up_proj | 3.0562 | 2.9720 | 2.9621 | 3.0246 | 2 | 24 | 0.0099 | 2.400491e-07 |
| model.layers.6.mlp.down_proj | down_proj | 3.0587 | 2.9726 | 2.9623 | 3.0122 | 2 | 24 | 0.0103 | 4.433622e-09 |
| model.layers.7.self_attn.q_proj | q_proj | 3.0613 | 2.9580 | 2.9419 | 2.9893 | 2 | 22 | 0.0161 | 1.125864e-06 |
| model.layers.7.self_attn.k_proj | k_proj | 3.0621 | 2.7314 | 2.6904 | 2.7314 | 2 | 19 | 0.0410 | 3.698548e-06 |
| model.layers.7.self_attn.v_proj | v_proj | 3.0628 | 2.6860 | 2.6449 | 2.6917 | 2 | 21 | 0.0410 | 4.736102e-07 |
| model.layers.7.self_attn.o_proj | o_proj | 3.0635 | 2.9268 | 2.9107 | 2.9593 | 2 | 22 | 0.0161 | 8.905960e-09 |
| model.layers.7.mlp.gate_proj | gate_proj | 3.0646 | 2.9726 | 2.9627 | 3.0117 | 2 | 24 | 0.0099 | 3.912517e-07 |
| model.layers.7.mlp.up_proj | up_proj | 3.0676 | 2.9722 | 2.9624 | 3.0232 | 2 | 22 | 0.0099 | 2.466428e-07 |
| model.layers.7.mlp.down_proj | down_proj | 3.0707 | 2.9841 | 2.9738 | 3.0226 | 2 | 24 | 0.0103 | 4.729061e-09 |
| model.layers.8.self_attn.q_proj | q_proj | 3.0737 | 2.8895 | 2.8733 | 2.9193 | 2 | 22 | 0.0161 | 1.228160e-06 |
| model.layers.8.self_attn.k_proj | k_proj | 3.0753 | 2.7888 | 2.7478 | 2.7867 | 2 | 20 | 0.0410 | 3.614475e-06 |
| model.layers.8.self_attn.v_proj | v_proj | 3.0759 | 2.7309 | 2.6899 | 2.7427 | 2 | 20 | 0.0410 | 4.133051e-07 |
| model.layers.8.self_attn.o_proj | o_proj | 3.0767 | 2.9409 | 2.9248 | 2.9735 | 2 | 23 | 0.0161 | 1.208497e-08 |
| model.layers.8.mlp.gate_proj | gate_proj | 3.0779 | 2.9604 | 2.9505 | 2.9981 | 2 | 24 | 0.0099 | 4.348973e-07 |
| model.layers.8.mlp.up_proj | up_proj | 3.0822 | 2.9953 | 2.9854 | 3.0454 | 2 | 23 | 0.0099 | 2.668273e-07 |
| model.layers.8.mlp.down_proj | down_proj | 3.0855 | 2.9965 | 2.9863 | 3.0358 | 2 | 24 | 0.0103 | 6.206840e-09 |
| model.layers.9.self_attn.q_proj | q_proj | 3.0890 | 2.9576 | 2.9415 | 2.9905 | 2 | 22 | 0.0161 | 1.445749e-06 |
| model.layers.9.self_attn.k_proj | k_proj | 3.0903 | 2.7273 | 2.6863 | 2.7282 | 2 | 19 | 0.0410 | 3.704198e-06 |
| model.layers.9.self_attn.v_proj | v_proj | 3.0912 | 2.7839 | 2.7428 | 2.7974 | 2 | 19 | 0.0410 | 4.640353e-07 |
| model.layers.9.self_attn.o_proj | o_proj | 3.0920 | 2.9516 | 2.9355 | 2.9812 | 2 | 23 | 0.0161 | 1.548155e-08 |
| model.layers.9.mlp.gate_proj | gate_proj | 3.0934 | 3.0117 | 3.0018 | 3.0517 | 2 | 24 | 0.0099 | 4.491842e-07 |
| model.layers.9.mlp.up_proj | up_proj | 3.0968 | 2.9957 | 2.9859 | 3.0448 | 2 | 23 | 0.0099 | 2.888431e-07 |
| model.layers.9.mlp.down_proj | down_proj | 3.1013 | 3.0131 | 3.0028 | 3.0541 | 2 | 24 | 0.0103 | 7.689159e-09 |
| model.layers.10.self_attn.q_proj | q_proj | 3.1053 | 2.9339 | 2.9178 | 2.9632 | 2 | 23 | 0.0161 | 1.455524e-06 |
| model.layers.10.self_attn.k_proj | k_proj | 3.1073 | 2.6786 | 2.6376 | 2.6795 | 2 | 19 | 0.0410 | 4.222215e-06 |
| model.layers.10.self_attn.v_proj | v_proj | 3.1086 | 2.7506 | 2.7096 | 2.7535 | 2 | 19 | 0.0410 | 4.551528e-07 |
| model.layers.10.self_attn.o_proj | o_proj | 3.1096 | 2.9617 | 2.9456 | 2.9937 | 2 | 23 | 0.0161 | 1.050262e-08 |
| model.layers.10.mlp.gate_proj | gate_proj | 3.1114 | 3.0610 | 3.0512 | 3.1064 | 2 | 25 | 0.0099 | 5.049315e-07 |
| model.layers.10.mlp.up_proj | up_proj | 3.1139 | 3.0118 | 3.0020 | 3.0655 | 2 | 23 | 0.0099 | 3.485840e-07 |
| model.layers.10.mlp.down_proj | down_proj | 3.1192 | 3.0304 | 3.0201 | 3.0727 | 2 | 25 | 0.0103 | 9.688674e-09 |
| model.layers.11.self_attn.q_proj | q_proj | 3.1241 | 2.9782 | 2.9621 | 3.0116 | 2 | 22 | 0.0161 | 1.474619e-06 |
| model.layers.11.self_attn.k_proj | k_proj | 3.1261 | 2.7317 | 2.6907 | 2.7288 | 2 | 19 | 0.0410 | 4.392153e-06 |
| model.layers.11.self_attn.v_proj | v_proj | 3.1275 | 2.7626 | 2.7216 | 2.7655 | 2 | 19 | 0.0410 | 4.957440e-07 |
| model.layers.11.self_attn.o_proj | o_proj | 3.1288 | 2.9800 | 2.9639 | 3.0140 | 2 | 22 | 0.0161 | 7.980761e-09 |
| model.layers.11.mlp.gate_proj | gate_proj | 3.1309 | 3.0441 | 3.0342 | 3.0890 | 2 | 24 | 0.0099 | 5.800719e-07 |
| model.layers.11.mlp.up_proj | up_proj | 3.1362 | 3.0487 | 3.0388 | 3.0997 | 2 | 22 | 0.0099 | 3.805968e-07 |
| model.layers.11.mlp.down_proj | down_proj | 3.1419 | 3.0534 | 3.0431 | 3.0984 | 2 | 24 | 0.0103 | 1.111993e-08 |
| model.layers.12.self_attn.q_proj | q_proj | 3.1480 | 3.0102 | 2.9941 | 3.0489 | 2 | 22 | 0.0161 | 1.198159e-06 |
| model.layers.12.self_attn.k_proj | k_proj | 3.1504 | 2.7392 | 2.6982 | 2.7434 | 2 | 20 | 0.0410 | 3.459682e-06 |
| model.layers.12.self_attn.v_proj | v_proj | 3.1522 | 2.8064 | 2.7654 | 2.8075 | 2 | 20 | 0.0410 | 4.595429e-07 |
| model.layers.12.self_attn.o_proj | o_proj | 3.1537 | 3.0089 | 2.9928 | 3.0452 | 2 | 22 | 0.0161 | 8.214720e-09 |
| model.layers.12.mlp.gate_proj | gate_proj | 3.1563 | 3.0733 | 3.0634 | 3.1190 | 2 | 23 | 0.0099 | 5.649978e-07 |
| model.layers.12.mlp.up_proj | up_proj | 3.1628 | 3.0718 | 3.0619 | 3.1184 | 2 | 21 | 0.0099 | 4.024262e-07 |
| model.layers.12.mlp.down_proj | down_proj | 3.1704 | 3.0823 | 3.0720 | 3.1256 | 2 | 23 | 0.0103 | 1.329564e-08 |
| model.layers.13.self_attn.q_proj | q_proj | 3.1785 | 3.0539 | 3.0378 | 3.0962 | 2 | 23 | 0.0161 | 1.348511e-06 |
| model.layers.13.self_attn.k_proj | k_proj | 3.1815 | 2.7575 | 2.7165 | 2.7542 | 2 | 20 | 0.0410 | 3.647426e-06 |
| model.layers.13.self_attn.v_proj | v_proj | 3.1840 | 2.8273 | 2.7863 | 2.8309 | 2 | 19 | 0.0410 | 7.498064e-07 |
| model.layers.13.self_attn.o_proj | o_proj | 3.1861 | 3.0442 | 3.0281 | 3.0820 | 2 | 23 | 0.0161 | 1.101792e-08 |
| model.layers.13.mlp.gate_proj | gate_proj | 3.1896 | 3.1119 | 3.1020 | 3.1506 | 2 | 24 | 0.0099 | 5.693063e-07 |
| model.layers.13.mlp.up_proj | up_proj | 3.1980 | 3.1058 | 3.0959 | 3.1466 | 2 | 21 | 0.0099 | 4.525465e-07 |
| model.layers.13.mlp.down_proj | down_proj | 3.2091 | 3.1205 | 3.1102 | 3.1566 | 2 | 23 | 0.0103 | 1.788735e-08 |
| model.layers.14.self_attn.q_proj | q_proj | 3.2214 | 3.0899 | 3.0738 | 3.1364 | 2 | 22 | 0.0161 | 1.061782e-06 |
| model.layers.14.self_attn.k_proj | k_proj | 3.2261 | 2.8555 | 2.8145 | 2.8572 | 2 | 20 | 0.0410 | 2.043057e-06 |
| model.layers.14.self_attn.v_proj | v_proj | 3.2294 | 2.8343 | 2.7933 | 2.8272 | 2 | 20 | 0.0410 | 1.106757e-06 |
| model.layers.14.self_attn.o_proj | o_proj | 3.2330 | 3.0859 | 3.0698 | 3.1205 | 2 | 21 | 0.0161 | 2.298619e-08 |
| model.layers.14.mlp.gate_proj | gate_proj | 3.2385 | 3.1379 | 3.1280 | 3.1733 | 2 | 24 | 0.0099 | 6.599820e-07 |
| model.layers.14.mlp.up_proj | up_proj | 3.2564 | 3.1743 | 3.1644 | 3.2056 | 2 | 22 | 0.0099 | 4.658891e-07 |
| model.layers.14.mlp.down_proj | down_proj | 3.2742 | 3.1829 | 3.1726 | 3.2110 | 2 | 23 | 0.0103 | 2.064491e-08 |
| model.layers.15.self_attn.q_proj | q_proj | 3.2994 | 3.1392 | 3.1231 | 3.1824 | 2 | 22 | 0.0161 | 9.493683e-07 |
| model.layers.15.self_attn.k_proj | k_proj | 3.3112 | 2.9220 | 2.8810 | 2.9164 | 2 | 20 | 0.0410 | 2.105997e-06 |
| model.layers.15.self_attn.v_proj | v_proj | 3.3186 | 3.0041 | 2.9631 | 3.0069 | 2 | 20 | 0.0410 | 9.349814e-07 |
| model.layers.15.self_attn.o_proj | o_proj | 3.3246 | 3.1740 | 3.1579 | 3.2188 | 2 | 22 | 0.0161 | 3.714294e-08 |
| model.layers.15.mlp.gate_proj | gate_proj | 3.3372 | 3.2480 | 3.2381 | 3.2736 | 2 | 24 | 0.0099 | 5.891041e-07 |
| model.layers.15.mlp.up_proj | up_proj | 3.3818 | 3.2959 | 3.2861 | 3.3202 | 2 | 23 | 0.0099 | 4.415318e-07 |
| model.layers.15.mlp.down_proj | down_proj | 3.4676 | 3.3732 | 3.3630 | 3.3960 | 2 | 25 | 0.0103 | 3.076200e-08 |
