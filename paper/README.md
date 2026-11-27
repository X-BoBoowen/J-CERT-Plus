# Manuscript

`main.tex` and `main.pdf` contain *J-CERT+: Certified Full-Catalog Intervention Selection in Signed Graphs*, by Bowen Xue, Tianyi Chen, and Yuze Xia, Hangzhou Normal University, China. This is a manuscript; publication DOI and page fields remain template placeholders.

Build with a complete TeX Live installation supporting the supplied ACM/PVLDB template, including `latexmk`, `libertine`, `inconsolata`, `newtx`, `algorithms`, `algorithmicx`, and `kastrup`:

```bash
bash build.sh
```

The template and bibliography style are unmodified. Dependency information is in `review/TEX_DEPENDENCY_PROVENANCE.json`; no font files are distributed. The manuscript has 14 pages, with non-reference content ending on page 12, and 48 references. The three author-provided figures and captions are in `figures/` and `FIGURE_CAPTIONS.tex`. The plot-data inputs in `review/` support interpretation and regeneration of earlier artwork; the plotting scripts do not recreate the edited PNG artwork. Build from `main.tex`; no scientific experiment is needed for compilation.
