"""Check physical volume conservation at each aggregation boundary.

Use gross positive, gross negative, and net equality across geometric mapping.
Use net equality when opposite overlap signs combine on the same MF6 node."""

from __future__ import annotations

from collections.abc import Iterable

from ..errors import ConservationError
from .records import SignedVolume


def assert_signed_volume_close(
    expected: SignedVolume,
    actual: SignedVolume,
    *,
    absolute_tolerance_m3: float,
    relative_tolerance: float,
    label: str,
) -> None:
    """compare signed volumes without allowing positive and negative errors to cancel."""

    checks = (
        ("positive", expected.positive_m3, actual.positive_m3),
        ("negative", expected.negative_m3, actual.negative_m3),
        ("net", expected.net_m3, actual.net_m3),
    )
    failures: list[str] = []
    for name, expected_value, actual_value in checks:
        error = abs(actual_value - expected_value)
        allowed = absolute_tolerance_m3 + relative_tolerance * max(
            abs(expected_value), abs(actual_value)
        )
        if error > allowed:
            failures.append(
                f"{name}: expected={expected_value:.17g} actual={actual_value:.17g} error={error:.6e} allowed={allowed:.6e}"
            )
    if failures:
        raise ConservationError(
            f"{label} failed signed conservation: " + "; ".join(failures)
        )


def assert_net_volume_close(
    expected: SignedVolume,
    actual: SignedVolume,
    *,
    absolute_tolerance_m3: float,
    relative_tolerance: float,
    label: str,
) -> None:
    """compare only net volume when aggregation can cancel opposite signs."""

    error = abs(actual.net_m3 - expected.net_m3)
    allowed = absolute_tolerance_m3 + relative_tolerance * max(
        abs(expected.net_m3), abs(actual.net_m3)
    )
    if error > allowed:
        raise ConservationError(
            f"{label} failed net conservation: "
            f"expected={expected.net_m3:.17g} actual={actual.net_m3:.17g} "
            f"error={error:.6e} allowed={allowed:.6e}"
        )


def combine_signed_volumes(volumes: Iterable[SignedVolume]) -> SignedVolume:
    items = list(volumes)
    return SignedVolume(
        positive_m3=float(sum(item.positive_m3 for item in items)),
        negative_m3=float(sum(item.negative_m3 for item in items)),
        net_m3=float(sum(item.net_m3 for item in items)),
    )
