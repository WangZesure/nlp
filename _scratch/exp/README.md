# Reliability-Tax experiments

Black-box budget sweep: for each model, vary the thinking budget and measure
accuracy plus three reliability axes (calibration, selective prediction,
abstention-vs-fabrication) across four task types.

No GPU. Pure API, standard library only.

## 1. Prerequisites

- Python 3.13 (stdlib only; no numpy/pandas needed)
- API keys for the providers you use, as environment variables:

| provider    | env var                | note                          |
|-------------|------------------------|-------------------------------|
| deepseek    | `DEEPSEEK_API_KEY`     | R1 (`deepseek-reasoner`) + V3 |
| volcengine  | `ARK_API_KEY`          | Doubao; pass your endpoint id |
| dashscope   | `DASHSCOPE_API_KEY`    | Qwen3                         |
| siliconflow | `SILICONFLOW_API_KEY`  | Qwen3 / others                |
| moonshot    | `MOONSHOT_API_KEY`     | Kimi                          |
| zhipu       | `ZHIPU_API_KEY`        | GLM                           |

## 2. Run order

```bash
cd _scratch/exp

# 0. offline pipeline check — no API, no cost
python -I run_sweep.py --mock --n 20 --k 2
python -I score.py && python -I analyze.py

# 1. fetch evaluation sets  (writes data/*.jsonl)
python -I fetch_data.py --n 200

# 2. smoke test on ONE model / task / two budgets  (small cost)
python -I run_sweep.py --models deepseek-chat --tasks math --budgets b1,b4 --n 20 --k 2

# 3. full sweep  (resumable; re-run to continue after any interruption)
python -I run_sweep.py

# 4. score + analyse
python -I score.py
python -I analyze.py --eps 0.10
```

## 3. What each file does

| file | role |
|---|---|
| `providers.py`  | one OpenAI-compatible client, retry + timeout |
| `config.json`   | models x budgets x tasks x templates x k |
| `fetch_data.py` | downloads & normalises GSM8K / NQ / HotpotQA / SciFact-false-premise |
| `run_sweep.py`  | orchestrates calls; concurrent, resumable, logs cost |
| `score.py`      | parses replies -> accuracy / ECE / overconfidence / fabrication |
| `analyze.py`    | profile, reliability tax, oracle budget, usable accuracy |

Outputs: `runs/*.jsonl` (raw), `results/scores.{json,csv}`,
`results/profile.json`.

## 4. Design notes

- **Budget** is nominal `max_tokens`; results are reported against the
  **realized** token count, because budget knobs are not comparable across
  providers.
- **Thinking toggle** is passed as `enable_thinking` where the provider
  supports it (Qwen3, GLM, Doubao); DeepSeek uses distinct models
  (`reasoner` vs `chat`).
- **Fabrication** = the model *endorses* a premise known to be false.
  Refuting (or hedging) is not fabrication. The three-way judgement needs a
  gold standard; see the paper's limitations.
- **Resumability**: each run file is keyed by `(item_id, sample_idx)`;
  re-running skips what is already recorded.

## 5. Cost

Full sweep at `n=200, k=4, 4 budgets, 2 templates, 6 models, 4 tasks` is
about 1.5x10^5 calls. On domestic APIs that is roughly **$20-80**.
Always run the mock check and a one-model smoke test first.
