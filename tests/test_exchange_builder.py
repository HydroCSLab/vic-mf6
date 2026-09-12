"""Pure-unit tests for exchange-table preprocessing helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vicmf6.preprocess.exchange_builder import (
    _coordinate_edges,
    _mf6_discretization,
    parse_vic_global,
)


def test_coordinate_edges_preserve_regular_centers():
    centers = np.asarray([48.1875, 48.3125, 48.4375, 48.5625])
    edges = _coordinate_edges(centers, "latitude")
    np.testing.assert_allclose(edges, [48.125, 48.25, 48.375, 48.5, 48.625])


def test_parse_vic_global_discovers_domain_and_variable_names(tmp_path: Path):
    domain = tmp_path / "domain.nc"
    domain.touch()
    parameters = tmp_path / "params.nc"
    parameters.touch()
    global_file = tmp_path / "vic.global.txt"
    global_file.write_text(
        "\n".join(
            [
                "DOMAIN domain.nc",
                "DOMAIN_TYPE LAT latitude",
                "DOMAIN_TYPE LON longitude",
                "DOMAIN_TYPE MASK active",
                "DOMAIN_TYPE AREA cell_area",
                "PARAMETERS params.nc",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_vic_global(global_file)
    assert parsed["DOMAIN"] == domain.resolve()
    assert parsed["PARAMETERS"] == parameters.resolve()
    assert parsed["DOMAIN_TYPE"]["LAT"] == "latitude"
    assert parsed["DOMAIN_TYPE"]["AREA"] == "cell_area"


def test_mf6_discretization_does_not_misclassify_disu_as_dis():
    class FakeDisu:
        pass

    class FakeGwf:
        name = "STEHEKIN"

        def __init__(self):
            self.disu = FakeDisu()

        def get_package(self, name):
            return self.disu if name.lower() == "disu" else None

        @property
        def dis(self):
            # Mimic FloPy's ambiguous convenience accessor that can expose
            # the DISU package through ``gwf.dis``.
            return self.disu

    assert _mf6_discretization(FakeGwf()) == "DISU"
