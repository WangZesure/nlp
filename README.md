# The Reliability Tax

Course project (NLP, Research track): *when does making a language model
"think longer" make it less trustworthy?*

We treat the thinking budget as a control knob and measure, for each model and
task, how accuracy and three reliability axes (calibration, selective
prediction, abstention-versus-fabrication) move as the budget changes. The
budget–reliability profile is the object; the *reliability tax* is the
reliability given up per unit of accuracy gained.

## Layout

```
acl/                     paper sources + compiled PDFs
  proposal.tex  proposal.pdf     Phase-1 proposal (1-2 pp, refs excluded)
  main.tex      main.pdf         full paper draft
  refs.bib                       verified references
  acl.sty acl_natbib.bst         ACL style files (required to build)
_scratch/
  exp/                   core experiment code (see exp/README.md)
  pilot/run_pilot.py     minimal end-to-end pipeline check
  *.md                   survey notes and findings
```

Not in version control (see `.gitignore`): the corpus crawler and its
downloads, the evaluation datasets, raw API responses, and LaTeX build
artefacts.

## Experiment code (`_scratch/exp/`)

```bash
cd _scratch/exp
python -I run_sweep.py --mock --n 20 --k 2   # offline pipeline check
python -I fetch_data.py --n 200              # download evaluation sets
python -I run_sweep.py                       # budget sweep (needs API keys)
python -I score.py && python -I analyze.py   # metrics and profiles
```

No GPU; Python 3.13 standard library only. Provider keys go in environment
variables (`DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`, `ZHIPU_API_KEY`, ...).

## Building the paper

```bash
cd acl
pdflatex proposal && bibtex proposal && pdflatex proposal && pdflatex proposal
```

## Status

- Protocol, code and pipeline: done and validated end-to-end.
- First real run: `deepseek-flash`, 1,280 calls — see
  `_scratch/exp/results/FINDINGS.md`.
- Next: scale to 200 items x k=4, add Qwen3 / GLM / Doubao for cross-family
  coverage, then evaluate the stopping rule.
