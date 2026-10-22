# figures

The figures as published are in the paper and its supplementary material. This directory
holds only this note.

The panel renders can be regenerated from the aggregate results in `../results`, without
MIMIC-IV access. From the repository root, stage the results into a working root first:

    python src/stage_results.py work
    export CBCEYE_ROOT="$PWD/work"
    python src/build_landscape_plotly.py   # Figure 1d
    python src/build_plotly_figs.py        # Figure 1e, Figure 2a-c
    python src/build_fig_eye_gain.py       # Figure 1f
    python src/build_fig2d_flowmap.py      # Figure 2d
    python src/build_fig3_final.py         # Figure 3
    python src/build_nhanes_figure.py      # Figure 4
    python src/build_fig5_territory.py     # Figure 5
    python src/territory_permutation.py    # Figure 5 permutation p-values

Renders are written to `work/`, `work/figures/` and `work/final_manuscript/figures/`.
Figure 1a-c is BioRender artwork, and the published composites set these renders beside it,
so the composites themselves are assembled by hand rather than by code. Figure 2e is fitted
on the patient-level tables. Panel renders use Plotly with the Kaleido static image backend,
pinned in `../requirements.txt`.
