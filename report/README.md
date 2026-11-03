# BTP-1 Report (IEEE format, Overleaf)

IEEE conference-format LaTeX (`IEEEtran`) for the BTP-1 interim report.

## Layout
```
report/
├── main.tex              # preamble, title/authors, \input of each section
├── refs.bib              # bibliography (IEEEtran style)
├── sections/             # one file per section; edit these
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_background.tex
│   ├── 03_dataset.tex     # setup, protocol, policies, groups, fields
│   ├── 04_pipeline.tex    # mat -> pkl, cleaning, indexing pitfalls
│   ├── 05_eda.tex         # figures + correlation analysis
│   ├── 06_discussion.tex
│   └── 07_plan.tex
├── figures/              # vector PDFs + stats.json (generated)
├── scripts/make_figures.py
└── notes.md              # scratch area: dump ideas/results here before writing them up
```

## Overleaf
1. Zip this folder (or use `report_overleaf.zip` in the repo root).
2. Overleaf → **New Project → Upload Project** → select the zip.
3. Compiler: pdfLaTeX (default). Main document: `main.tex`.

Red `[TODO: ...]` markers show placeholders (department, roll no., guide, timeline).

## Regenerating figures
From the repo root (needs the cleaned dataset, see the root README):
```
python3 report/scripts/make_figures.py            # EDA figures (2018-02-20 batch)
python3 report/scripts/make_cleaning_figures.py   # data-cleaning figures
```
Re-upload `figures/` to Overleaf afterwards.
