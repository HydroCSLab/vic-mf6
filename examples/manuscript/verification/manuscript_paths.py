"""Runtime paths for the portable original numerical experiments."""
import os
from pathlib import Path

RESULT_ROOT = Path(os.environ["VICMF6_VERIFICATION_OUTPUT"]).resolve()
SAMPLE_ROOT = Path(os.environ["VICMF6_VERIFICATION_SAMPLE"]).resolve()
VIC_ROOT = Path(os.environ["VICMF6_VERIFICATION_VIC"]).resolve()
MF6_LIBRARY = Path(os.environ["LIBMF6"]).resolve()


def source_provenance(command):
    """Describe installed source without requiring a Git checkout in Docker."""
    revisions = Path(os.environ["VICMF6_COMPONENT_REVISIONS"]).read_text()
    if command == ["git", "rev-parse", "HEAD"]:
        return next(line.split("|")[1] for line in revisions.splitlines()
                    if line.startswith("vic|")) + "\n"
    return ("Installed source, staged for this experiment; no Git metadata.\n"
            "See source-hashes.csv and component-revisions.txt in the suite root.\n")
