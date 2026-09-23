import gzip
from pathlib import Path


def open_audit_text(path, mode):
    destination = Path(path)
    if "w" in mode:
        destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".gz":
        return gzip.open(str(destination), mode + "t", encoding="utf-8")
    return destination.open(mode, encoding="utf-8")
