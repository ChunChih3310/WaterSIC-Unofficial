# AGENTS.md

## Mission

Reproduce the WaterSIC paper as faithfully as possible inside this repository, with a complete, runnable, documented pipeline for:

1. calibration data collection,
2. WaterSIC quantization,
3. saved quantized artifacts,
4. benchmark/evaluation,
5. bitrate / entropy / Huffman reporting,
6. reproducible experiment configs and reports.

The first priority is faithful reproduction of the paper.  
The second priority is runtime and engineering quality.  
Do not optimize for novelty before the paper-faithful implementation is working end-to-end.

---

## Repository Scope: Absolute Non-Negotiable Rules

### 1) Never modify files outside this repository
You are **strictly forbidden** from modifying any file outside:

`/nfs_tmp/Compression_team/src/WaterSIC`

This includes, but is not limited to:

- parent directories
- sibling repositories
- user home dotfiles
- system Python
- system packages
- shared scripts
- other conda environments
- shell startup files
- external caches that require writing outside this repo, unless they are read-only and unavoidable

If a task appears to require editing something outside this repo, do **not** do it.  
Instead, solve it locally inside this repo.

### 2) Paper is available locally
The paper PDF is already available under the repo root `paper/` directory.  
Consult it whenever needed.  
Do not invent algorithm details that are not supported by the paper.

### 3) Hugging Face credentials
A Hugging Face API key is already available in `.env`.  
Use it securely. Never print it. Never commit it. Never copy it into logs, markdown, JSON, or shell history.

### 4) Conda isolation
You must create a **new** conda environment for this project.  
You must **not** modify any existing conda environment.

### 5) Use git properly
You must use git throughout the work.

- Make logical, reviewable commits.
- Do not create a single giant commit.
- Do not commit large model weights, caches, or datasets unless explicitly intended and safely ignored/managed.
- Keep `.gitignore` correct.

### 6) Reproduction before embellishment
Do not replace WaterSIC with a simpler or “close enough” quantizer.  
Do not silently fall back to GPTQ or RTN and call it WaterSIC.  
If any part of WaterSIC is missing, state it clearly in docs and reports.

### 7) Runtime matters
The implementation must be careful about runtime and memory use.  
However, runtime optimization must never invalidate the paper-faithful algorithm.

---

## What “Paper-Faithful” Means Here

Implement the WaterSIC pipeline in the spirit of the paper, not an unrelated approximation.

At minimum, the implementation must include the following core ideas:

1. **Integer-grid quantization with entropy coding instead of bounded scaling quantization**
   - Base quantizer is round-to-nearest on an equispaced grid.
   - Quantized values are unbounded integers.
   - Compression rate is controlled by grid spacing and entropy coding, not by clipping to a fixed small integer range.

2. **Per-column unequal rate allocation**
   - WaterSIC must allocate different effective rates to different input channels / columns.
   - This is the core algorithmic idea and must not be removed.

3. **ZSIC-style sequential quantization**
   - Use lower-triangular structure from Cholesky factorization.
   - Quantize from the last column backward while subtracting already committed interference.

4. **LMMSE correction**
   - After integer rounding, compute the shrinkage factor and use it in the recursive update.

5. **Activation drift correction**
   - Use statistics involving quantized-model activations, not only original activations.

6. **Residual stream correction**
   - For residual-contributing down-projection layers, account for residual stream discrepancy.

7. **Diagonal rescaler optimization**
   - Optimize row and column diagonal rescalers after ZSIC reconstruction.

8. **Attention-weighted calibration for QKV**
   - Apply only to Q/K/V projections.
   - Weight token contributions by attention importance.

9. **Adaptive mixing**
   - Optimize the mixing between quantized and unquantized statistics for attention projections.

10. **Dead feature erasure**
    - Detect near-zero variance input dimensions and handle them explicitly.

11. **Rate assignment by binary search**
    - Solve for the layer-level rate by searching the grid parameter.
    - Maintain a running global remaining budget across layers.

12. **Sequential layerwise quantization**
    - Quantize layers in model order.
    - Collect downstream statistics using already-quantized earlier layers.

---

## Primary Reproduction Targets

### First-priority models
Prioritize the models that the paper emphasizes most strongly:

1. `Llama-3.2-1B`
2. `Qwen3-8B`

Do not start with unrelated models before these are supported.

### Second-priority models
If the first-priority targets are working and runtime/resources allow, extend to:

3. `Llama-3-8B`
4. `Llama-2-7B`

### First-priority benchmark
Implement and report:

- WikiText-2 perplexity
- context length = 2048

This is the minimum required benchmark for a paper-faithful first milestone.

### Required bitrate reporting
For every completed run, report all of the following:

1. target average bitwidth
2. actual effective average bitwidth
3. pre-entropy/raw storage bitwidth
4. entropy-estimated bitwidth
5. Huffman-coded bitwidth
6. side-information overhead
7. final total effective bitwidth

Important:
- The **paper-faithful primary metric** is the entropy-derived effective rate.
- Because this project explicitly requires Huffman reporting, also report empirical or canonical Huffman bitrate.
- Do **not** replace the paper’s effective-rate metric with only Huffman rate.

---

## Required High-Level Deliverables

By the end, this repository must contain:

1. A working quantization pipeline for WaterSIC
2. A working benchmark pipeline
3. Reproducible configs for each experiment
4. Saved quantized artifacts and metadata
5. A complete `README.md`
6. Full documentation under `docs/`
7. Final experiment reports with bitrate + benchmark results
8. A clean repository structure
9. Tests / verification scripts for core algorithmic correctness
10. Git history with meaningful commits

---

## Required Repository Structure

Use a clear structure similar to the following.  
You may adapt names slightly if necessary, but keep the same spirit.

```text
WaterSIC/
├── AGENTS.md
├── README.md
├── .gitignore
├── .env                         # already exists; never print or commit secrets
├── paper/
│   └── ...pdf
├── configs/
│   ├── env/
│   │   └── watersic_conda.yml
│   ├── models/
│   │   ├── llama32_1b.yaml
│   │   ├── qwen3_8b.yaml
│   │   ├── llama3_8b.yaml
│   │   └── llama2_7b.yaml
│   ├── quant/
│   │   ├── watersic_default.yaml
│   │   ├── watersic_llama32_1b.yaml
│   │   ├── watersic_qwen3_8b.yaml
│   │   └── ...
│   └── eval/
│       ├── wikitext2.yaml
│       └── ...
├── src/
│   └── watersic/
│       ├── __init__.py
│       ├── utils/
│       │   ├── io.py
│       │   ├── logging.py
│       │   ├── env.py
│       │   ├── seed.py
│       │   ├── device.py
│       │   ├── path_guard.py
│       │   └── huffman.py
│       ├── data/
│       │   ├── wikitext2.py
│       │   └── calibration.py
│       ├── models/
│       │   ├── registry.py
│       │   ├── llama.py
│       │   ├── qwen.py
│       │   └── hooks.py
│       ├── stats/
│       │   ├── covariance.py
│       │   ├── attention_weighting.py
│       │   ├── drift.py
│       │   ├── residual.py
│       │   └── dead_features.py
│       ├── quant/
│       │   ├── zsic.py
│       │   ├── lmmse.py
│       │   ├── rescalers.py
│       │   ├── watersic_layer.py
│       │   ├── watersic_model.py
│       │   ├── rate_search.py
│       │   └── serialization.py
│       ├── eval/
│       │   ├── perplexity.py
│       │   ├── metrics.py
│       │   └── runner.py
│       └── report/
│           ├── schema.py
│           ├── summarize.py
│           └── render_markdown.py
├── scripts/
│   ├── setup_env.sh
│   ├── download_model.py
│   ├── collect_calibration.py
│   ├── quantize_model.py
│   ├── benchmark_model.py
│   ├── run_experiment.py
│   └── make_report.py
├── tests/
│   ├── test_zsic.py
│   ├── test_lmmse.py
│   ├── test_dead_features.py
│   ├── test_rate_search.py
│   ├── test_rescalers.py
│   ├── test_huffman.py
│   └── test_path_guard.py
├── docs/
│   ├── implementation_plan.md
│   ├── algorithm_notes.md
│   ├── calibration_pipeline.md
│   ├── quantization_pipeline.md
│   ├── benchmarking.md
│   ├── report_format.md
│   ├── reproduction_log.md
│   └── known_issues.md
└── outputs/
    ├── logs/
    ├── stats/
    ├── quantized/
    ├── eval/
    └── reports/
````

### Notes on large files

* Quantized weights, downloaded original weights, caches, and intermediate tensors should **not** be committed unless explicitly intended.
* Use `.gitignore` aggressively for:

  * model checkpoints
  * HF cache mirrors inside repo
  * temporary tensors
  * benchmark outputs that are too large
* Keep lightweight metadata, configs, logs, and markdown reports versioned.

---

## Conda Environment Rules

Create a new environment dedicated to this repository only.

### Requirements

* Do not modify existing environments.
* Store the environment spec in:

  * `configs/env/watersic_conda.yml`
* Provide a small setup script:

  * `scripts/setup_env.sh`

### Environment contents

Use only what is necessary. Typical packages may include:

* python
* pytorch + cuda-compatible build
* transformers
* datasets
* accelerate
* sentencepiece / tokenizers as needed
* numpy
* scipy
* pyyaml
* tqdm
* pandas
* matplotlib
* safetensors
* python-dotenv
* pytest

Avoid bloated or unrelated dependencies.

### Environment naming

Use a project-specific name, for example:

`watersic`

If a different name is chosen, document it clearly and use it consistently.

---

## GPU Usage Requirements

An A6000 may be used.

### Mandatory behavior

The code must include **automatic idle-GPU selection**.

### Required GPU selection policy

Implement a utility that behaves as follows:

1. If `CUDA_VISIBLE_DEVICES` is already set by the caller, respect it.
2. Otherwise, inspect visible GPUs using a local mechanism such as `nvidia-smi`.
3. Choose the GPU with the best idle score, based on:

   * low memory used
   * low utilization
4. Log the chosen GPU clearly.
5. Allow manual override via config or CLI.
6. Fail gracefully to CPU if CUDA is unavailable, but log that this is slower and not the intended path.

### Additional GPU engineering requirements

* Never hardcode GPU id `0`.
* Never assume a single-GPU machine.
* Keep tensors on the intended device.
* Free temporary tensors aggressively.
* Use `torch.no_grad()` for calibration and evaluation unless gradients are explicitly needed.
* Use BF16/FP16 inference where safe, but keep numerically sensitive statistics accumulation in sufficiently stable precision.
* Avoid unnecessary device-host transfers.

---

## Path Safety Guard

Implement a path-guard utility to reduce accidental writes outside this repo.

### Requirements

* Centralize all write paths through a helper.
* Resolve and validate output paths.
* Refuse to write if the final resolved path is outside:
  `/nfs_tmp/Compression_team/src/WaterSIC`
* Add tests for this behavior.

This is mandatory.

---

## WaterSIC Algorithm Requirements

This section is the heart of the implementation.

### A. Calibration data collection

Implement a calibration pipeline that can collect:

* original layer input activations `X`
* quantized-model layer input activations `X_hat`
* residual stream states `R`
* quantized-model residual states `R_hat`

### Dataset handling

For the first faithful implementation:

* use WikiText-2 training split for calibration
* concatenate into a single token stream
* partition into non-overlapping chunks of length 2048
* keep the pipeline deterministic
* document exact tokenizer/model revision used

### Sequential dependency

Because later-layer statistics depend on earlier quantization decisions:

* quantize layers sequentially
* collect quantized-model statistics using the partially quantized model state

Do not fake this by collecting all stats only from the full-precision model.

---

### B. Core covariance/statistics objects

Implement and store the statistics required by the paper:

* `Σ_X`
* `Σ_X_hat`
* `Σ_X,X_hat`
* `Σ_Δ,X_hat`, where `Δ = R - R_hat`

These should be collected per quantized matrix or per logical layer group as needed.

### Numerical requirements

* Use stable accumulation.
* Avoid explicit matrix inversion whenever possible.
* Prefer triangular solves or linear solves.
* Guard against singular / ill-conditioned covariance matrices.

---

### C. Plain WaterSIC foundation

Before full WaterSIC, implement a clean and testable plain version.

At the layer level:

1. compute lower-triangular `L` from Cholesky factorization,
2. set per-column spacing from the diagonal of `L`,
3. run ZSIC from right to left,
4. serialize the resulting integer matrix,
5. estimate entropy / bitrate.

This is the foundational milestone.

---

### D. ZSIC implementation details

Implement ZSIC in a way that is:

* explicit
* well-tested
* vectorized where reasonable
* easy to debug

For a given matrix input:

* iterate columns from `n-1` down to `0`
* quantize the current transformed column
* subtract its contribution to the remaining unquantized coordinates

Do not hide the algorithm in overly compressed code.
Correctness and readability matter here.

---

### E. LMMSE correction

Implement the LMMSE shrinkage factor after integer rounding.

Requirements:

* compute the shrinkage factor per column
* use it in the recursive update
* store it for later reporting / rescaler initialization if needed

Be careful about divide-by-zero or tiny-norm cases.
Handle these explicitly and deterministically.

---

### F. Activation drift correction

Implement the drift-corrected objective using quantized-model activations.

This is not optional.

The code should clearly distinguish:

* original-model statistics
* quantized-model statistics
* mixed statistics used for the current quantization decision

Avoid ambiguous variable naming.

---

### G. Residual stream correction

Implement residual stream correction for residual-contributing layers.

In transformer terms, this especially matters for down-projection style layers such as:

* attention output projection (`wo`)
* MLP down projection (`w2`)

The implementation should clearly identify which layer types receive residual correction and why.

Do not apply this blindly to all layers without documentation.

---

### H. Diagonal rescaler optimization

Implement the alternating optimization of:

* row rescaler `T`
* column rescaler `Γ`

Requirements:

* initialize from the LMMSE-derived values where appropriate
* normalize scale consistently
* alternate until convergence or a small fixed iteration cap
* use a ridge / numerical safeguard when solving linear systems
* log optimization diagnostics when debug logging is enabled

Also provide tests that verify the objective does not obviously worsen on controlled synthetic inputs.

---

### I. Entropy coding and bitrate accounting

Implement bitrate accounting carefully.

At minimum, the reporting stack must include:

1. raw uncoded storage size
2. symbol frequency entropy
3. side-information overhead
4. canonical Huffman-coded size
5. final effective average bitwidth

### Important rule

The implementation must separate:

* **algorithmic quantization output**
* **bitrate estimation/reporting**
* **artifact serialization**

Do not entangle them into one opaque function.

### Huffman requirement

Because this project requires Huffman reporting:

* implement canonical Huffman coding or an equivalently precise expected-code-length calculator
* report average bits per weight after Huffman coding
* clearly distinguish this from entropy and from raw integer storage

### Optional additional codecs

If practical, also report:

* zstd
* lzma

But these are optional extras after the required metrics above are correct.

---

### J. Rate assignment

Implement layer rate assignment via binary search over the WaterSIC scale parameter.

Requirements:

* rate target is per-layer but coordinated through a global remaining budget
* the search variable must be the grid parameter controlling effective rate
* rate must be monotone enough in that variable for binary search to work
* record:

  * target rate
  * searched parameter bounds
  * selected parameter
  * actual achieved rate

### Runtime-faithful defaults

Use the paper-inspired search strategy as the default starting point:

* binary search iterations: around 30
* rate-search row sampling fraction: around 10%
* after selecting the parameter, rerun on the full matrix for final output

These should be configurable, but the default should prioritize faithful reproduction.

---

### K. Attention-weighted calibration

For QKV projections only:

* compute token importance scores from attention probabilities
* weight the covariance estimates accordingly
* substitute these weighted covariances into the quantization objective

This must be implemented clearly and documented carefully.

### Important scope rule

Do not apply attention weighting to non-QKV matrices unless explicitly documented as an experiment.

---

### L. Adaptive mixing

Implement the two mixing parameters:

* `epsilon_qr`
* `epsilon_aw`

Use a lightweight search procedure for the attention block.

### Required search behavior

* optimize `epsilon_qr` first with `epsilon_aw` fixed
* optimize `epsilon_aw` second using the chosen `epsilon_qr`
* use a robust one-dimensional search such as golden-section search
* evaluate candidates using attention-block distortion at the `wo` input

### Runtime-faithful defaults

Use paper-inspired defaults as the starting point:

* ~15 iterations for each golden-section search

These should be configurable but must default to the faithful setting.

---

### M. Dead feature erasure

Implement dead-feature detection and removal.

Requirements:

* detect near-zero variance input dimensions
* use a robust threshold based on the **median** variance, not the mean
* remove dead features before the reduced quantization solve
* re-expand the quantized weight back to original dimensionality by inserting zeros

### Important rule

This is not just a numerical trick.
It changes effective dimensionality and therefore affects rate budgeting.
That must be reflected in the bookkeeping.

---

### N. Damping / numerical stabilization

Implement damping in a clear, explicit, configurable way.

Requirements:

* apply damping after the final mixed statistics are formed
* document exactly where damping enters
* keep WaterSIC defaults conservative and close to the paper-faithful settings
* do not bury damping logic deep inside random helper functions

For the first faithful implementation, use small damping defaults for WaterSIC and expose them through config.

---

## Runtime and Memory Requirements

This project must be careful about runtime.

### Required engineering practices

1. Use triangular solves instead of explicit matrix inversion wherever possible.
2. Cache reusable calibration statistics.
3. Avoid storing unnecessary full activations for all layers at once if a streaming/chunked alternative is possible.
4. Reuse buffers where safe.
5. Keep quantization code vectorized where reasonable.
6. Keep debug logging optional, not always-on.
7. Separate “smoke test mode” from “full paper-faithful run”, but do not change the algorithm itself.
8. Make batch sizes configurable.
9. Write progress logs for long-running steps.
10. Ensure all expensive searches can be resumed or at least diagnosed from logs.

### Required execution modes

Support at least:

* `smoke` mode
  For quick sanity checks on a tiny subset.

* `full` mode
  For the actual paper-faithful runs.

The algorithmic logic must remain the same.
Only dataset size / row sampling / logging frequency may differ.

---

## Model Download / Loading Rules

### Requirements

* Prefer the exact model identifiers/revisions that match the intended paper reproduction as closely as possible.
* Record model revision and tokenizer revision in the run metadata.
* Never hardcode private credentials into code.
* Keep download logic scriptable and documented.

### Artifact handling

If the repository stores local checkpoints or converted artifacts, do so under repo-local paths only, for example:

* `outputs/original/`
* `outputs/quantized/`

Do not write checkpoints elsewhere.

---

## Benchmarking Requirements

The first required benchmark is WikiText-2 perplexity.

### Minimum benchmark spec

* dataset: WikiText-2 test split
* context length: 2048
* report perplexity clearly
* compare against the unquantized baseline for the same model/tokenizer/context setup

### Benchmark runner requirements

* one command should benchmark a saved quantized artifact
* logs should include:

  * model id
  * git commit
  * config path
  * runtime device
  * evaluation dataset
  * context length
  * dtype
  * batch size
  * final perplexity

### Reproducibility

The benchmark report must be reproducible from a saved config and saved quantized artifact.

---

## Reporting Requirements

Every quantize+benchmark run must generate a structured report.

### Required report fields

At minimum include:

* timestamp
* git commit hash
* environment name
* CUDA / GPU info
* model id and revision
* tokenizer id and revision
* quantization config path
* evaluation config path
* dataset names and splits
* sequence length
* calibration size
* target global bitwidth
* achieved global bitwidth
* per-layer target bitwidth
* per-layer achieved bitwidth
* raw/pre-entropy average bitwidth
* entropy average bitwidth
* Huffman average bitwidth
* side-information overhead
* perplexity
* optional extra metrics if implemented
* wall-clock runtime
* notes on warnings/fallbacks

### Required report outputs

For each run, generate:

1. machine-readable report
   e.g. JSON

2. human-readable summary
   e.g. Markdown

### Final summary report

Also generate an aggregate report that compares runs across rates and models.

---

## README Requirements

A complete `README.md` is mandatory.

It must include:

1. What WaterSIC is
2. What is implemented in this repository
3. What is paper-faithful and what is not yet implemented
4. Environment setup
5. How to download models
6. How to collect calibration data
7. How to quantize a model
8. How to benchmark a quantized model
9. How to generate reports
10. Expected output locations
11. Common troubleshooting notes

The README must be enough for a technically capable reader to run the pipeline.

---

## Documentation Requirements

Add and maintain markdown documents under `docs/`.

### Required docs

At minimum:

* `docs/implementation_plan.md`
* `docs/algorithm_notes.md`
* `docs/calibration_pipeline.md`
* `docs/quantization_pipeline.md`
* `docs/benchmarking.md`
* `docs/report_format.md`
* `docs/reproduction_log.md`
* `docs/known_issues.md`

### Documentation standard

Each document should be concise but real.
Do not create empty placeholder files.

### Reproduction log

Maintain a chronological log of what was run, what succeeded, what failed, and what remains.

---

## Git Workflow Requirements

### Commit rules

Make small, meaningful commits, such as:

* environment and setup
* data pipeline
* core ZSIC
* LMMSE
* rate search
* full layer quantizer
* benchmarking
* reporting
* docs

### Branching

Use the repository’s existing workflow if one exists.
If not, keep the history clean and linear.

### Commit hygiene

* Do not commit secrets.
* Do not commit giant raw checkpoints.
* Do not commit random debug dumps.

---

## Recommended Execution Order

Follow this order unless there is a very strong reason not to.

### Phase 0 — Repository audit

* inspect current repo structure
* inspect existing code
* inspect paper PDF in `paper/`
* document the implementation plan in `docs/implementation_plan.md`

### Phase 1 — Safe environment + guard rails

* create conda env spec
* add setup script
* add path guard
* add `.gitignore`
* add logging / config loading utilities

### Phase 2 — Data and model I/O

* implement model loading
* implement WikiText-2 loading
* implement calibration chunking
* implement hooks for activations / residuals / attention data

### Phase 3 — Core math modules

* implement covariance accumulation
* implement ZSIC
* implement LMMSE
* implement dead feature handling
* implement rate / entropy / Huffman utilities

### Phase 4 — Layer quantization

* implement plain WaterSIC layer quantizer
* implement full WaterSIC layer quantizer
* implement diagonal rescaler optimization
* verify on synthetic/unit tests

### Phase 5 — Model-level sequential quantization

* implement sequential layer traversal
* implement downstream stat refresh
* implement global remaining budget
* save quantized artifacts + metadata

### Phase 6 — Benchmarking

* implement WikiText-2 perplexity runner
* benchmark original and quantized models
* save structured metrics

### Phase 7 — Reporting

* generate per-run report
* generate aggregate report
* update README and docs

### Phase 8 — Reproduction runs

* first reproduce `Llama-3.2-1B`
* then reproduce `Qwen3-8B`
* only then consider additional models

---

## Testing and Verification Requirements

This repository must include tests.

### Required unit/integration tests

At minimum:

1. **ZSIC sanity**

   * output shapes
   * integer output
   * deterministic behavior under fixed seed
   * residual update is consistent

2. **LMMSE sanity**

   * no NaN on small-norm edge cases
   * shrinkage factor behaves sensibly

3. **Dead feature handling**

   * dead features are removed and correctly reinserted as zero columns

4. **Rate search**

   * chosen parameter moves achieved rate in the expected direction

5. **Huffman**

   * expected code length is correct on controlled distributions

6. **Path guard**

   * writing outside repo is refused

7. **Smoke quantization**

   * a tiny synthetic or tiny real subset run completes end-to-end

### Required synthetic verification

Add at least one synthetic test where anisotropic covariance makes unequal per-column spacing meaningfully different from uniform spacing.
This helps verify that the WaterSIC-specific logic is not accidentally collapsed into uniform GPTQ-like behavior.

---

## Logging Requirements

Implement structured logging.

### Logs must show

* start/end of each major phase
* selected device
* selected model
* layer currently being quantized
* target and achieved per-layer rate
* search progress when enabled
* benchmark progress
* output artifact paths

### Logging modes

Support at least:

* normal
* verbose/debug

Debug mode should be informative without being absurdly slow.

---

## Configuration Requirements

Use explicit config files for experiments.

### Configs must capture

* model id/revision
* tokenizer id/revision
* calibration dataset/split
* eval dataset/split
* sequence length
* layer order
* target rates
* damping
* binary search iterations
* row sampling fraction
* golden-section iterations
* dead-feature threshold
* output paths
* device policy
* seeds

A completed run must be reproducible from config + code + saved artifact.

---

## Serialization Requirements

Quantized artifacts must be saved with enough metadata to benchmark them later without ambiguity.

### Save at least

* integer-coded weights or final reconstructed weights, depending on artifact type
* per-layer metadata
* per-layer rates
* scale / spacing info
* rescaler info
* dead-feature masks
* model config snapshot
* tokenizer info
* quantization config snapshot

Make the artifact format explicit and documented.

---

## Honesty / Failure Policy

If you hit a blocker:

* do not pretend the reproduction is complete
* do not silently substitute another algorithm
* do not hide numerical issues
* document the problem clearly in:

  * logs
  * `docs/known_issues.md`
  * final report

If only part of the target is achieved, leave the repo in a state that is:

* runnable
* documented
* testable
* easy to continue from

Partial but honest completion is better than a misleading “done”.

---

## Definition of Done

The task is only “done” when all of the following are true:

1. A new dedicated conda env spec exists.
2. No file outside `/nfs_tmp/Compression_team/src/WaterSIC` was modified.
3. WaterSIC quantization runs end-to-end on at least one paper-priority model.
4. WikiText-2 perplexity benchmarking runs end-to-end on the produced artifact.
5. Reports include raw / entropy / Huffman / effective bitwidth metrics.
6. Reproducible configs are saved.
7. README is complete.
8. Docs under `docs/` are real and useful.
9. Tests exist for the core algorithmic pieces.
10. Git history is clean and meaningful.

The **target** completion standard is stronger:

* end-to-end reproduction support for both:

  * `Llama-3.2-1B`
  * `Qwen3-8B`

with report generation for multiple target rates.

---

## Final Instruction

Be meticulous.

This repository should end up as a serious reproduction codebase, not a quick prototype.

When choosing between:

* “fast but vague”
* “slower but faithful and reproducible”

prefer:

* **faithful and reproducible**

unless the slower path is clearly unnecessary.

Never compromise the paper’s central algorithmic ideas.
Never modify files outside this repository.
Always leave clear docs for the next person.
