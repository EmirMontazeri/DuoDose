"""Evidence-preserving cell identities and chemistry for independent screens."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from .downloads import select_pubchem_cid, verify_downloads
from .preprocess import _pubchem_property, normalize_cell_name
from .screen_readers import one_file


def map_screen_cells(
    labels: list[str], reference_raw: Path, aliases: dict[str, dict[str, str]], policy: str
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any], list[Path]]:
    """Match released names to CMP without merging derivatives or inventing RNA-seq."""
    directory = reference_raw / "cell_model_passports"
    model_path = one_file(directory, "model_list_*.csv")
    availability_path = one_file(directory, "model_dataset_availability_*.csv")
    models = pd.read_csv(model_path, dtype=str).fillna("")
    availability = pd.read_csv(availability_path, dtype=str).fillna("")
    if (
        not {"model_id", "model_name"} <= set(models)
        or models.model_id.eq("").any()
        or models.model_id.duplicated().any()
    ):
        raise ValueError("CMP model catalog needs unique IDs and released model names")
    if availability.model_id.duplicated().any():
        raise ValueError("CMP availability catalog has duplicated model IDs")
    available = pd.Series(False, index=availability.index)
    for column in ("RNASeq Broad Cell Lines", "RNASeq Sanger Cell Lines"):
        if (
            column not in availability
            or not availability[column].isin(["True", "False", "true", "false"]).all()
        ):
            raise ValueError(f"Invalid CMP cell-line RNA-seq availability column {column}")
        available |= availability[column].str.lower().eq("true")
    rna_models = set(availability.loc[available, "model_id"])
    indexes: list[dict[str, set[str]]] = [defaultdict(set) for _ in range(3)]
    for row in models.to_dict("records"):
        for level, names in enumerate(
            (
                [row["model_name"]],
                row.get("synonyms", "").split(";"),
                [row.get("CCLE_ID", "").split("_", 1)[0]],
            )
        ):
            for name in names:
                normalized = normalize_cell_name(name)
                if normalized:
                    indexes[level][normalized].add(row["model_id"])
    lookup = models.set_index("model_id")
    alias_lookup = {}
    for label, spec in aliases.items():
        if (
            set(spec) != {"cell_id", "evidence"}
            or not spec["evidence"].strip()
            or spec["cell_id"] not in lookup.index
        ):
            raise ValueError(f"Alias {label} requires a known CMP model_id and evidence")
        key = normalize_cell_name(label)
        if key in alias_lookup and alias_lookup[key] != spec:
            raise ValueError(f"Conflicting configured aliases for {label}")
        alias_lookup[key] = spec
    records, names, mappings, excluded = {}, {}, {}, {}
    for label in sorted(set(labels)):
        key = normalize_cell_name(label)
        candidates, method = set(), "unresolved"
        for index, level in zip(indexes, ("model_name", "synonyms", "CCLE_ID_prefix"), strict=True):
            if key in index:
                candidates, method = index[key], level
                break
        if key in alias_lookup:
            chosen = alias_lookup[key]["cell_id"]
            if candidates and chosen not in candidates:
                raise ValueError(f"Configured alias {label} conflicts with released CMP metadata")
            candidates, method = {chosen}, "configured_alias"
        if len(candidates) != 1:
            reason = "ambiguous_cell_label" if candidates else "unresolved_cell_label"
            excluded[label] = {"reason": reason, "candidates": sorted(candidates)}
        else:
            model = next(iter(candidates))
            if model not in rna_models:
                excluded[label] = {"reason": "missing_cell_expression", "model_id": model}
            else:
                names[label] = model
                mappings[label] = {
                    "model_id": model,
                    "model_name": lookup.loc[model, "model_name"],
                    "method": method,
                    "evidence": (
                        alias_lookup[key]["evidence"]
                        if method == "configured_alias"
                        else f"CMP {method}: exact normalized name; no derivative substitution"
                    ),
                }
                records[model] = {
                    "cell_id": model,
                    "model_id": model,
                    "cosmic_id": lookup.loc[model].get("COSMIC_ID", ""),
                    "model_name": lookup.loc[model, "model_name"],
                    "mapping_method": "released_CMP_name_or_evidenced_alias",
                }
        if label in excluded and policy == "error":
            raise ValueError(f"Cell {label}: {excluded[label]}")
    if not records:
        raise ValueError(
            "No response cells have both unambiguous CMP identity and released cell-line RNA-seq"
        )
    cells = pd.DataFrame([records[k] for k in sorted(records)])
    return (
        cells,
        names,
        {"mappings": mappings, "excluded_labels": excluded, "aliases": aliases},
        [model_path, availability_path],
    )


def _pubchem_structure(raw: Path, name: str) -> tuple[str | None, list[Path]]:
    """Read the archived exact-name/CID/synonym chain, never query during preprocessing."""
    directory = raw / "pubchem" / hashlib.sha256(name.encode()).hexdigest()[:16]
    if not directory.exists():
        return None, []
    files = [directory / f"{part}.json" for part in ("identity", "synonyms", "property")]
    if not all(p.is_file() for p in files):
        raise ValueError(f"Incomplete official chemistry reference for {name}")
    identifiers = json.loads(files[0].read_text())["IdentifierList"]["CID"]
    cid = select_pubchem_cid(directory, identifiers)
    if len(identifiers) != 1:
        files.extend(directory / f"candidate_{candidate}.json" for candidate in identifiers)
    synonyms = json.loads(files[1].read_text())["InformationList"]["Information"]
    if (
        len(synonyms) != 1
        or synonyms[0]["CID"] != cid
        or normalize_cell_name(name) not in {normalize_cell_name(n) for n in synonyms[0]["Synonym"]}
    ):
        raise ValueError(f"Archived PubChem synonyms do not establish {name}")
    smiles, _ = _pubchem_property(files[2].read_bytes(), cid)
    return smiles, files


def screen_chemistry(
    raw: Path, source: str, drugs: list[str], bits: int, radius: int, resolve_pubchem: bool
) -> tuple[pd.DataFrame, dict[str, str], np.ndarray, np.ndarray, list[Path]]:
    """Use NCI structures or archived NIH records with explicit alias evidence."""
    inventory = verify_downloads(raw)
    if inventory["source"] != source:
        raise ValueError("Official reference inventory belongs to a different screen")
    consumed = [raw / "official_downloads.json"]
    # Pin the entire download inventory, including publisher checksums/legends.
    consumed.extend(raw / entry["path"] for entry in inventory["files"])
    structures: dict[str, str] = {}
    drug_names: dict[str, str] = {}
    mapping = {name: name for name in drugs}
    if source == "nci_almanac":
        sdf = raw / "ComboCompoundSet.sdf"
        consumed.append(sdf)
        with rdBase.BlockLogs():
            for molecule in Chem.SDMolSupplier(str(sdf), removeHs=False):
                if molecule is not None:
                    key = "NSC" + molecule.GetProp("NSC")
                    structures[key] = Chem.MolToSmiles(
                        molecule, canonical=True, isomericSmiles=True
                    )
        name_path = raw / "ComboCompoundNames_small.txt"
        consumed.append(name_path)
        names: dict[str, list[str]] = defaultdict(list)
        for line in name_path.read_text().splitlines():
            identifier, label = line.split("\t", 1)
            names["NSC" + str(int(identifier))].append(label.strip())
        for name in drugs:
            drug_names[name] = ";".join(names.get(name, []))
            if name in structures:
                continue
            # A source identifier absent from the SDF can be harmonized only
            # when the original NCI exact-name list points to one represented NSC.
            alternatives = [
                other
                for other in structures
                if set(names.get(name, [])) & set(names.get(other, []))
            ]
            if len(alternatives) == 1:
                mapping[name] = alternatives[0]
            elif resolve_pubchem and name in names:
                for label in names[name]:
                    smiles, paths = _pubchem_structure(raw, label)
                    consumed.extend(paths)
                    if smiles is not None:
                        structures[name] = smiles
                        break
    else:
        for name in drugs:
            drug_names[name] = name
            if resolve_pubchem:
                smiles, paths = _pubchem_structure(raw, name)
                consumed.extend(paths)
                if smiles is not None:
                    structures[name] = smiles
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    records, identifiers, vectors = [], [], []
    for drug in sorted(set(mapping.values())):
        smiles = structures.get(drug)
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        supported = mol is not None and mol.GetNumAtoms() > 0
        records.append(
            {
                "drug_id": drug,
                "released_drug_name": drug_names.get(drug, drug),
                "supported": supported,
                "canonical_isomeric_smiles": smiles,
                "structure_status": (
                    "official_reference" if supported else "no_validated_official_structure"
                ),
                "aliases": json.dumps(
                    sorted(n for n, canonical in mapping.items() if canonical == drug)
                ),
                "fingerprint_bits": bits,
                "fingerprint_radius": radius,
            }
        )
        if supported:
            vector = np.zeros(bits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), vector)
            identifiers.append(drug)
            vectors.append(vector)
    if not vectors:
        raise ValueError("No verified official structures; run fetch-official for this dataset")
    return (
        pd.DataFrame(records),
        mapping,
        np.asarray(identifiers),
        np.stack(vectors),
        sorted(set(consumed)),
    )
