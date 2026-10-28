"""Official download integrity and chemical identity verification."""

from pathlib import Path
import hashlib
import pytest
from reverse_dose.downloads import fetch_file, select_pubchem_cid, verify_downloads
from reverse_dose.utils import atomic_json


def test_official_downloads_refuse_overwrites_and_tampering(tmp_path: Path) -> None:
    item = fetch_file(tmp_path, "record.txt", "https://wiki.nci.nih.gov/test", payload=b"original")
    assert item["sha256"] == hashlib.sha256(b"original").hexdigest()
    with pytest.raises(ValueError, match="immutable"):
        fetch_file(tmp_path, "record.txt", "https://wiki.nci.nih.gov/test", payload=b"changed")
    atomic_json(
        tmp_path / "official_downloads.json",
        {"schema_version": "official-downloads-v1", "files": [item]},
    )
    verify_downloads(tmp_path)
    (tmp_path / "record.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        verify_downloads(tmp_path)


def test_ambiguous_chemistry_only_accepts_identical_inorganic_salt_parent(tmp_path: Path) -> None:
    def record(cid: int, smiles: str) -> None:
        atomic_json(
            tmp_path / f"candidate_{cid}.json",
            {"PropertyTable": {"Properties": [{"CID": cid, "SMILES": smiles}]}},
        )

    record(1, "CCN.Cl.Cl")
    record(2, "CCN")
    assert select_pubchem_cid(tmp_path, [1, 2]) == 2
    record(1, "CCO.Cl.Cl")
    with pytest.raises(ValueError, match="not unique"):
        select_pubchem_cid(tmp_path, [1, 2])
