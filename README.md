# Do Review Comments Matter? Review Language and Acceptance at ICLR

This repository contains the manuscript, figure-generation scripts, derived data products, and supporting project documents for the ICLR peer-review language study.

## Public repository URL

`https://github.com/BKZhao/ICRL-Review-Language`

## Chinese guide

For a Chinese walkthrough of the paper, see:

`PAPER_GUIDE_ZH.md`

## Repository structure

- `DOC/`
  - Course planning materials, archived raw input files, and earlier exploratory assets.
- `paper/`
  - `main.tex`: manuscript source.
  - `references.bib`: cleaned bibliography.
  - `scripts/`: data-processing, appendix-generation, and figure-redesign scripts.
  - `data/derived/`: analysis-ready tables and numeric outputs used by the paper.
  - `figures/`: manuscript and appendix figures.

## Data source

The study dataset comes from OpenReview, using the ICLR group page and the OpenReview API for conference years 2018--2023.

OpenReview ICLR page:

`https://openreview.net/group?id=ICLR.cc`

## Build

From the repository root:

```bash
bash paper/scripts/build.sh
```

This rebuilds the derived outputs, regenerates the figures, and compiles the manuscript PDF.

## Suggested data-availability sentence

Project repository URL:

`https://github.com/BKZhao/ICRL-Review-Language`

Suggested sentence:

`Data, code, figures, and derived outputs for this study are available at https://github.com/BKZhao/ICRL-Review-Language. The underlying review records come from OpenReview, using the ICLR group page and the OpenReview API.`
