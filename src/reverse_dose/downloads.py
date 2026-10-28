"""Explicit acquisition of publisher releases and official reference records.

Training never downloads or modifies raw data. This separate command writes
immutable files and a URL/checksum inventory; existing files are verified.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
from zipfile import ZipFile

from .layout import DatasetPaths
from .utils import atomic_json, read_json, sha256_file

LOGGER = logging.getLogger(__name__)
NCI_BASE = "https://wiki.nci.nih.gov/download/attachments/338237347/"
ONEIL_ARTICLES = (22503133, 22503127, 22503130, 22503139)
ONEIL_DOI = "10.1158/1535-7163.MCT-15-0843"
OFFICIAL_HOSTS = {
    "wiki.nci.nih.gov",
    "api.figshare.com",
    "ndownloader.figshare.com",
    "pubchem.ncbi.nlm.nih.gov",
}


def _request(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    if (
        urllib.parse.urlparse(url).scheme != "https"
        or urllib.parse.urlparse(url).hostname not in OFFICIAL_HOSTS
    ):
        raise ValueError(f"Not an approved official source URL: {url}")
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "DuoDose/0.1 research data acquisition"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt + 1 == retries:
                raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt + 1 == retries:
                raise
        time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Could not download {url}")


def fetch_file(
    raw: Path, relative: str, url: str, *, md5: str | None = None, payload: bytes | None = None
) -> dict[str, Any]:
    """Install an official file atomically; refuse to replace existing bytes."""
    path = raw / relative
    if not path.resolve().is_relative_to(raw.resolve()):
        raise ValueError("Download path must stay inside the selected raw directory")
    if not path.exists():
        data = payload if payload is not None else _request(url)
        if md5 is not None and hashlib.md5(data).hexdigest() != md5:
            raise ValueError(f"Publisher checksum mismatch: {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".download-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            # Hard-link installation fails if another process installed the path.
            os.link(name, path)
        finally:
            Path(name).unlink(missing_ok=True)
        LOGGER.info("Downloaded official source %s", path)
    if md5 is not None and hashlib.md5(path.read_bytes()).hexdigest() != md5:
        raise ValueError(f"Existing file differs from publisher checksum: {path}")
    if payload is not None and path.read_bytes() != payload:
        raise ValueError(f"Existing immutable reference differs from requested record: {path}")
    return {
        "path": relative,
        "url": url,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "publisher_md5": md5,
    }


def verify_downloads(raw: Path) -> dict[str, Any]:
    """Verify every acquired file before its metadata or chemistry is consumed."""
    manifest = read_json(raw / "official_downloads.json")
    if manifest.get("schema_version") != "official-downloads-v1":
        raise ValueError("Unsupported official download manifest")
    for item in manifest["files"]:
        path = raw / item["path"]
        if (
            not path.resolve().is_relative_to(raw.resolve())
            or not path.is_file()
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"Official source integrity mismatch: {path}")
    return manifest


def select_pubchem_cid(directory: Path, identifiers: list[int]) -> int:
    """Disambiguate only identical organic structures with inorganic counterions.

    No stereochemistry, protonation, tautomer or organic-fragment inference is
    allowed. There must be exactly one isolated-parent CID among the exact-name
    hits. The original multi-hit response and every candidate are archived.
    """
    if len(identifiers) == 1:
        return identifiers[0]
    from rdkit import Chem
    from .preprocess import _pubchem_property

    parents, isolated = set(), []
    for cid in identifiers:
        smiles, _ = _pubchem_property((directory / f"candidate_{cid}.json").read_bytes(), cid)
        molecule = Chem.MolFromSmiles(smiles)
        fragments = Chem.GetMolFrags(molecule, asMols=True)
        organic = [m for m in fragments if any(atom.GetAtomicNum() == 6 for atom in m.GetAtoms())]
        if len(organic) != 1:
            raise ValueError("Ambiguous PubChem name: multiple organic components")
        for fragment in fragments:
            if not any(atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms()):
                if fragment.GetNumAtoms() != 1 or fragment.GetAtomWithIdx(0).GetAtomicNum() not in {
                    3,
                    9,
                    11,
                    12,
                    17,
                    19,
                    20,
                    35,
                    53,
                }:
                    raise ValueError("Ambiguous PubChem name: unrecognized counterion")
        parents.add(Chem.MolToSmiles(organic[0], canonical=True, isomericSmiles=True))
        if len(fragments) == 1:
            isolated.append(cid)
    if len(parents) != 1 or len(isolated) != 1:
        raise ValueError(
            "Ambiguous PubChem name: exact organic stereochemical parent is not unique"
        )
    return isolated[0]


def _pubchem(raw: Path, name: str, files: list[dict[str, Any]]) -> None:
    """Archive official exact-name resolution, followed by the exact CID record.

    Names must occur verbatim in returned synonyms. Ambiguous name hits are
    errors; a missing record is explicit and never replaced by a similar drug.
    """
    directory = "pubchem/" + hashlib.sha256(name.encode()).hexdigest()[:16]
    identity_url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        + urllib.parse.quote(name, safe="")
        + "/cids/JSON"
    )
    local = raw / directory / "identity.json"
    try:
        identity = local.read_bytes() if local.exists() else _request(identity_url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            LOGGER.warning("No official PubChem exact-name record for %s", name)
            return
        raise
    identifiers = json.loads(identity)["IdentifierList"]["CID"]
    files.append(fetch_file(raw, directory + "/identity.json", identity_url, payload=identity))
    if len(identifiers) != 1:
        for candidate in identifiers:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{candidate}/property/IsomericSMILES/JSON"
            files.append(fetch_file(raw, directory + f"/candidate_{candidate}.json", url))
            time.sleep(0.25)
    cid = select_pubchem_cid(raw / directory, identifiers)
    for kind, suffix in (
        ("synonyms", "/synonyms/JSON"),
        ("property", "/property/IsomericSMILES/JSON"),
    ):
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}" + suffix
        files.append(fetch_file(raw, directory + f"/{kind}.json", url))
        time.sleep(0.25)
    synonyms = read_json(raw / directory / "synonyms.json")["InformationList"]["Information"]

    def normalize(value: str) -> str:
        return "".join(c for c in value.casefold() if c.isalnum())

    if (
        len(synonyms) != 1
        or synonyms[0]["CID"] != cid
        or normalize(name) not in {normalize(s) for s in synonyms[0]["Synonym"]}
    ):
        raise ValueError(f"PubChem CID {cid} does not verify the exact released name {name}")


def fetch_official(dataset_root: Path, dataset_name: str, source: str) -> dict[str, Any]:
    """Acquire only files from the original publisher and NIH reference services."""
    paths = DatasetPaths(dataset_root, dataset_name)
    raw = paths.raw
    raw.mkdir(parents=True, exist_ok=True)
    previous = verify_downloads(raw) if (raw / "official_downloads.json").exists() else None
    if previous is not None:
        if previous["source"] != source:
            raise ValueError("Official download inventory belongs to another dataset source")
        LOGGER.info("Verified %d official files; reusing offline inventory", len(previous["files"]))
        return previous
    files: list[dict[str, Any]] = []
    if source == "oneil":
        for article in ONEIL_ARTICLES:
            url = f"https://api.figshare.com/v2/articles/{article}/versions/1"
            local = raw / f"publisher/{article}.json"
            payload = local.read_bytes() if local.exists() else _request(url)
            metadata = json.loads(payload)
            if not any(r.get("identifier") == ONEIL_DOI for r in metadata["related_materials"]):
                raise ValueError(
                    "Publisher supplement does not reference the original O'Neil paper"
                )
            files.append(fetch_file(raw, f"publisher/{article}.json", url, payload=payload))
            for entry in metadata["files"]:
                files.append(
                    fetch_file(raw, entry["name"], entry["download_url"], md5=entry["computed_md5"])
                )
        # The publisher's single-agent file is small and supplies the complete
        # released compound labels without rescanning the large combination file.
        from openpyxl import load_workbook

        metadata = read_json(raw / "publisher/22503127.json")
        with (raw / metadata["files"][0]["name"]).open("rb") as handle:
            workbook = load_workbook(handle, read_only=True, data_only=True)
            rows = iter(workbook.active.values)
            header = next(rows)
            position = header.index("drug_name")
            drugs = sorted({str(row[position]) for row in rows if row[position] is not None})
            workbook.close()
        for name in drugs:
            _pubchem(raw, name, files)
        documentation = "https://doi.org/" + ONEIL_DOI
    elif source == "nci_almanac":
        for name in (
            "ComboCompoundSet.sdf",
            "ComboCompoundNames_small.txt",
            "ALMANAC_DataFields.txt",
        ):
            files.append(fetch_file(raw, name, NCI_BASE + name))
        if not (raw / "ComboDrugGrowth_Nov2017.csv").exists():
            archive = "ComboDrugGrowth_Nov2017.zip"
            files.append(fetch_file(raw, archive, NCI_BASE + archive))
            with ZipFile(raw / archive) as z:
                members = [m for m in z.namelist() if Path(m).name == "ComboDrugGrowth_Nov2017.csv"]
                if len(members) != 1:
                    raise ValueError("Unexpected NCI response archive contents")
                files.append(
                    fetch_file(
                        raw,
                        "ComboDrugGrowth_Nov2017.csv",
                        NCI_BASE + archive,
                        payload=z.read(members[0]),
                    )
                )
        # The NCI SDF's cisplatin representation fails strict RDKit valence
        # checks. Resolve its exact NCI-released name through NIH, with evidence.
        _pubchem(raw, "Cisplatin", files)
        documentation = "https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/338237347/NCI-ALMANAC"
    else:
        raise ValueError(
            "fetch-official supports oneil and nci_almanac; DREAM inputs are managed separately"
        )
    manifest = {
        "schema_version": "official-downloads-v1",
        "source": source,
        "documentation": documentation,
        "files": sorted(files, key=lambda x: x["path"]),
    }
    atomic_json(raw / "official_downloads.json", manifest)
    return manifest
