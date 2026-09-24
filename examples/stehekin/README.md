# Stehekin end-to-end verification

This example is the smallest case that exercises the complete VIC-MF6
software stack. It builds a connected synthetic MODFLOW 6 model, constructs
the overlap table from the VIC and MODFLOW 6 grids, runs two-way coupling, and
checks the resulting water-transfer and groundwater-budget diagnostics.

The groundwater model is a numerical fixture. It is not a calibrated model of
the Stehekin basin.

## Requirements

Complete the repository installation and smoke tests in the root
[README](../../README.md). The acceptance run also needs:

- a modified VIC 5 Image Driver executable;
- a MODFLOW 6 shared library with XMI and API6 support;
- a Stehekin VIC global file whose DOMAIN, PARAMETERS, and FORCING paths are
  readable; and
- one MPI implementation compatible with Python, VIC, and MODFLOW 6.

The example builds the MODFLOW 6 input deck itself. It does not need a MODFLOW
6 executable.

## Configure local paths

From the repository root:

~~~bash
cp examples/stehekin/config.example.yml \
   examples/stehekin/config.local.yml
${EDITOR:-vi} examples/stehekin/config.local.yml
~~~

Replace the three CHANGE_ME values:

~~~yaml
mf6:
  library: /absolute/path/to/libmf6.so

vic:
  global_file: /absolute/path/to/Stehekin_image_test.global.txt
  executable: /absolute/path/to/vic_image.exe
~~~

The local configuration is ignored by Git.

## Run complete acceptance

~~~bash
./scripts/acceptance_stehekin.sh \
    examples/stehekin/config.local.yml
~~~

The wrapper performs these steps in order:

1. clears only the example's prior generated run products;
2. builds the 12-cell connected MODFLOW 6 DISU fixture;
3. rebuilds the conservative VIC-MODFLOW 6 overlap table;
4. builds the native MPI child-disconnect helper;
5. runs configuration inspection and preflight;
6. launches one controller and one MODFLOW 6 worker;
7. checks ten daily coupling windows and both exchange directions; and
8. writes the postprocessing tables, acceptance record, report, and figures.

A successful run ends with the Stehekin checker and postprocessor reporting
PASS or **[OK]**. Review:

~~~text
examples/stehekin/run/diagnostics/run_summary.json
examples/stehekin/run/postprocessing/acceptance_summary.txt
examples/stehekin/run/postprocessing/report.md
~~~

## Generated-file boundary

The acceptance workflow may replace only these example-owned products:

~~~text
examples/stehekin/exchange_table.csv
examples/stehekin/exchange_table.summary.json
examples/stehekin/exchange_table.summary.txt
examples/stehekin/run/
~~~

These paths are ignored by Git. The workflow reads but does not modify the
configured VIC global file, VIC executable, or MODFLOW 6 shared library.

## Run individual stages

The individual commands are useful when diagnosing a failure:

~~~bash
python examples/stehekin/create_mf6.py

./scripts/build_stehekin_exchange_table.sh \
    examples/stehekin/config.local.yml

./native/build_disconnect.sh

./vicmf6 inspect -c examples/stehekin/config.local.yml
./vicmf6 preflight -c examples/stehekin/config.local.yml

mpirun -np 2 ./vicmf6 run \
    -c examples/stehekin/config.local.yml

python scripts/check_stehekin_result.py \
    examples/stehekin/run/diagnostics

./vicmf6 post all \
    -c examples/stehekin/config.local.yml
~~~

Re-run the model builder before retrying coupling if MODFLOW 6 advanced and the
coupled run did not finish.

## Fixture design

The generated model is a one-layer, 3-by-4 DISU grid with 12 connected cells
and a deliberately nonmatching VIC-MODFLOW 6 mapping. It uses a local vertical
datum with the VIC soil-base and MODFLOW 6 top at -1 m. Alternating initial
heads create an inspectable lateral-flow response.

See [Coupling design](../../docs/coupling-design.md) for the sign, volume,
head-transform, mapping, and MPI contracts. Passing this fixture verifies those
software contracts for this case; it does not establish basin calibration.
