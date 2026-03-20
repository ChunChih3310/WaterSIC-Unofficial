# Layer-0 Attention Debug Report

- Timestamp: `2026-03-13T03:26:38Z`
- Git commit: `1d495d7207c3da085f65706fa229c0010787597b`
- Model: `meta-llama/Llama-3.2-1B`
- Baseline small-eval PPL: `8.9880`

| Stage | Small-Eval PPL | Block Rel MSE | q | k | v | o | Entropy bw | Huffman bw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Post-core-fix safe path | 9.0138 | 5.742931e-04 | 4.597645e-04 | 9.624289e-04 | 1.909851e-02 | 1.217846e-02 | 2.8087 | 2.8561 |
