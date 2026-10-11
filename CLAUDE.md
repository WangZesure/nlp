# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Coursework workspace for a Chinese NLP course. It holds **two Markdown specs, their CSV datasets, and a chalkboard photo — no code yet.** Everything under `HW-1/` still has to be written from scratch. The specs are graded contracts: read them before implementing, and follow stated hyperparameters literally.

`AGENTS.md` in this folder covers the same ground for other agent tools. Keep the two in sync if either changes.

## Sources of truth

| File | Governs |
|---|---|
| `HW-1/作业1要求.md` | **Homework 1** — the operative spec for current work |
| `大作业要求.md` | **Course project** (Survey / Demo / Research tracks) — parallel, separate deliverable |
| `HW-1/image.png` | Chalkboard photo restating HW1 — fully redundant with the `.md`; don't re-analyze |
服务器：ssh -p 53607 root@connect.westb.seetacloud.com
pwd:kFVAnrEFL7ts
## Deadlines — both fall on 2026-10-12

- **HW1**: due **11:55 AM**. Deliver: report PDF + full code + run instructions, submitted as a **GitHub link**. This folder is not a git repo yet — creating and populating one is part of the submission.
- **Course Project Phase 1 (Proposal)**: 1–2 page PDF, ACL LaTeX template, → course Feishu folder as `NLP2026_TeamXX_Track/TeamXX_Proposal.pdf`.

## HW1 pipeline

The spec fixes the data flow. Deviating breaks cross-method comparability, which is explicitly graded.

```
NYT (text,label) ──shuffle, ONE fixed seed──> 80 / 10 / 10
                                              │
                    train ────────────────────┤──> Logistic Regression (T1, T2)
                                              │    BERT fine-tune (T3)
                    val ──> model selection only
                    test ──> EVERY reported number, same set for all methods
```

**Every method must share one identical split.** Fix a seed once, or persist the split to disk. Report Accuracy **and** Macro-F1 on the NYT test set for every method.

| Task | Pts | Methods (all on the shared split) |
|---|---|---|
| **T1** BoW + LR | 30 | Binary BoW (`x_i = 1` if present) · Word Frequency (`x_i = count`) |
| **T2** Word2Vec/GloVe + LR | 50 | Pre-trained `glove.6B.100d` · Word2Vec trained on AG News · Word2Vec trained on NYT |
| **T3** BERT | 20 | `google-bert/bert-base-uncased` fine-tune |

T2 document representation is specified: **mean of the document's word vectors**, then LR. All T2 vectors are 100-dim.

## Graded literal settings

Stated exactly in the spec, and graded — use as written, don't tune: 100-dim vectors, `glove.6B.100d`, `bert-base-uncased`, `max_length=64`, **3 epochs**, Logistic Regression as classifier for every non-BERT method.

## Open questions in the spec — resolve before finalizing

1. **T1 says "三种" (three) BoW methods but enumerates only two** — Binary BoW and Word Frequency. The chalkboard also shows two. This governs 30 points: confirm whether a third (e.g. TF-IDF) is expected or the count is a typo.
2. Tokenization: spec suggests `nltk.word_tokenize`; nltk is not installed and needs a `punkt` download. Install it or use a documented substitute.
3. Stopword removal / lowercasing / lemmatization are never mentioned. The spec is silent.

## Environment

Verified 2026-10-07: `python` 3.13.0 with **numpy 2.2.6, torch 2.14.0+cpu, scipy 1.16.2** only. **Missing**: pandas, scikit-learn, nltk, gensim, transformers, datasets. The submitted README must record install steps — the spec requires it.

Two consequences worth planning around:

- **`scikit-learn` is absent, yet LR is required for every non-BERT method.** Either install it or implement LR directly.
- **Torch is CPU-only** (24 threads, no CUDA). BERT fine-tuning over ~9.2k training docs × 3 epochs dominates runtime — size the batch accordingly, and state hardware needs honestly in the README rather than silently shrinking the setting.

Windows + PowerShell; the workspace path contains CJK (`水课文件`) — quote paths in shell commands. GloVe additionally requires downloading `glove.6B.zip` (~822 MB) from `nlp.stanford.edu`.

## Dataset facts (full-file, measured)

`HW-1/nyt.csv` — 11,519 data rows, columns `text,label`:

- Labels: **sports 8639, politics 1451, business 1429** → a 75.0% majority class. An always-`sports` baseline scores Accuracy 0.75 with Macro-F1 ≈ 0.36. **Accuracy alone is uninformative here; Macro-F1 is the discriminating metric.**
- Long documents: mean 655 tokens, p90 969, max 5260. At `max_length=64` only **0.16% of documents fit**, retaining **9.8%** of all tokens. Heavy truncation is inherent to the spec's setting, not a bug — say so in the report.
- ~73.7k lowercased vocabulary types over ~7.55M tokens. Vocabulary size matters for T1's `|V|`-dimensional vectors.

`HW-1/ag.csv` — 90,000 data rows, `text` column **only, no labels**. Per spec, use it *exclusively* to train Word2Vec. Never classify or evaluate on it.

## Gotchas

- Both CSVs use **CRLF** line endings and have **no trailing newline**. No embedded newlines, so `wc -l` happens to equal the data-row count for both — but stream with stdlib `csv` anyway rather than trusting that coincidence.
- Raw CSVs total ~65 MB (nyt 43.4 MB, ag 21.7 MB). Never paste samples into context.
- There is no `.gitignore`. Decide deliberately whether the CSVs belong in the GitHub submission — the spec only requires clear data-acquisition instructions (see `AGENTS.md` for the same open question).
- A shared seeded split is worth creating *first*. Retrofitting one after T1/T2/T3 have each split independently means every cross-method comparison is invalid.

## No build system

There is no package manifest, test suite, or linter. Scripts will be plain Python run directly, e.g. `python HW-1/train_bow.py`. Any entrypoint you add should be runnable standalone with documented dependencies, since the README's instructions are part of the grade.
