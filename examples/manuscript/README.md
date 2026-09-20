# Manuscript reproduction

This workflow runs the manuscript experiments in the software repository,
keeps raw runs under the project `analysis/` directory, checks the resulting
compact tables against the paper repository, and builds the paper from its
reviewed TeX and CSV sources. It follows the HydroCS project layout:

```text
<project-dir>/
├── vic-mf6/                 # this repository
├── vic-mf6-paper/           # paper repository
└── analysis/
    └── vic-mf6-manuscript/  # generated runs and comparison report
```

From a shell, copy and run:

```bash
mkdir -p ~/projects/vic-mf6-manuscript
cd ~/projects/vic-mf6-manuscript
git clone --recurse-submodules --branch manuscript \
  https://github.com/mabdazzam/vic-mf6.git vic-mf6
git clone --branch manuscript \
  https://github.com/mabdazzam/vic-mf6-paper.git vic-mf6-paper

python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r vic-mf6-paper/scripts/requirements-figures.txt

cd vic-mf6
VICMF6_VERSION=manuscript-2026-09-19 ./bundle/scripts/build-image.sh vic-mf6:manuscript
./bundle/scripts/run-manuscript.sh --workers 2
python3 examples/manuscript/scripts/compare-manuscript-tables.py

cd ../vic-mf6-paper
python3 scripts/create-manuscript-figures.py
make -C manuscript
make -C manuscript supplement graphical-abstract
```

The default results directory is
`~/projects/vic-mf6-manuscript/analysis/vic-mf6-manuscript/`. It contains
`execution.csv`, stage logs, raw model outputs, `tables/`, and
`table-comparison.csv`. The complete run must finish all stages and the table
comparison must report zero failures. The paper build uses the committed CSV
tables after this check; it does not run models during LaTeX compilation.

From the `vic-mf6` checkout, inspect the host results with:

```bash
find ../analysis/vic-mf6-manuscript -mindepth 1 -maxdepth 1 -type d -printf '%f/\n' | sort
cat ../analysis/vic-mf6-manuscript/execution.csv
```

`/results/manuscript` is only the corresponding path inside Docker. The runner
prints the host output directory at the start and after completion.

For a shorter software check, run this from `vic-mf6`:

```bash
./bundle/scripts/run-manuscript.sh --stages unit,acceptance,reference
```

The output directory must be empty. A second run can use the next numbered
analysis directory, for example `analysis/vic-mf6-manuscript-2`.

The software pins VIC and MODFLOW 6 in `bundle/components.lock` and records
the Python environment in each run. The paper tracks only reviewed TeX/PGF,
compact CSV plotting data, and figure scripts; raw NetCDF, NPZ, JSON, model
outputs, logs, PDFs, and local archives stay under `analysis/` or outside the
project repositories.
