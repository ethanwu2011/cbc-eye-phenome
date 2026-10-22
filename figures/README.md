# figures

This directory is empty on purpose. It is the output location that the figure scripts in
`../src` write to, so it exists in the repository only so those scripts have somewhere to
write on a fresh clone.

The figures as published are in the paper and its supplementary material. They are not
duplicated here, because they are composites assembled from the panel renders and would
drift out of step with the manuscript every time a panel is revised.

To regenerate them from the aggregate results in `../results`:

    python src/build_landscape_plotly.py
    python src/build_fig1b_organ.py
    python src/build_fig3_final.py
    python src/build_fig5_territory.py
    python src/build_fig_eye_gain.py
    python src/build_fig_triangulation.py
    python src/build_nhanes_figure.py
    python src/compose_figures.py     # assembles the multi-panel composites

Panel renders use Plotly with the Kaleido static image backend, pinned in
`../requirements.txt`. `compose_figures.py` trims each panel to its ink before scaling, so
panels of different aspect ratio sit flush in the composite.
