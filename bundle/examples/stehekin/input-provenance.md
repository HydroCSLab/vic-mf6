# Stehekin input provenance

The three NetCDF files are byte-for-byte copies from
`UW-Hydro/VIC_sample_data` commit
`997fc6bdc423cba72dac7895098d636172179967`. The bundle uses lowercase names
that follow the HydroCS technical-file convention:

| Bundled file | Source path |
| --- | --- |
| `input/domain_stehekin_20151028.nc` | `image/Stehekin/parameters/domain.stehekin.20151028.nc` |
| `input/stehekin_parameters_20160327.nc` | `image/Stehekin/parameters/Stehekin_test_params_20160327.nc` |
| `input/stehekin_forcings_10_days_1949.nc` | `image/Stehekin/forcings/Stehekin_image_test.forcings_10days.1949.nc` |

`input/checksums.sha256` records the upstream byte content. The bundle checks
these hashes before every image build. `stehekin.global.txt` is a small bundle
configuration derived from the sample global file: it uses relative input
paths and declares the two groundwater-coupling output variables.
