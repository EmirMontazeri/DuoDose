#!/usr/bin/env python3

"""
Dose-response dataset audit.

READ-ONLY for dataset/raw/.

Validates:
  1. Dataset directory structure and critical files
  2. AZ-DREAM corrected matrix counts
  3. DREAM cell metadata
  4. DREAM -> Cell Model Passports mapping by COSMIC ID
  5. Cell Model Passports RNA-seq availability for DREAM cells
  6. Drug chemistry / structure coverage
  7. Representative corrected 2D dose-response matrices
  8. Representative raw combination matrices
  9. Dose-response cell-label consistency

Writes:
    dataset/manifests/audit_report.json

Usage:
    uv run python audit_dataset.py

Strict mode:
    uv run python audit_dataset.py --strict

Normal expected result with the current official source files:
    PASS_WITH_WARNINGS

Known source-level chemistry exceptions:
  Biologics without conventional small-molecule SMILES:
    ADAM17   -> MEDI3622
    CD19     -> MT-103 / Blinatumomab
    PDGFR_Ab -> MEDI-575
    TNFSF10  -> TRAIL

  Unresolved challenge agents:
    ATR_3
    PIP5K1
    SLC16A4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

EXPECTED_DREAM_CELLS = 85
EXPECTED_DREAM_MATRICES = 11608
EXPECTED_OI_MATRICES = 20489
EXPECTED_DREAM_DRUGS = 118

CHEM_COL = "SMILES or PubChem ID"

SAMPLE_MATRIX_COUNT = 12


# -----------------------------------------------------------------------------
# Known source-level drug exceptions
# -----------------------------------------------------------------------------

KNOWN_BIOLOGIC_NO_SMILES = {
    "ADAM17": "MEDI3622; monoclonal antibody",
    "CD19": "MT-103 / Blinatumomab; bispecific antibody",
    "PDGFR_Ab": "MEDI-575; monoclonal antibody",
    "TNFSF10": "TRAIL; protein ligand",
}

KNOWN_UNRESOLVED_AGENT_STRUCTURE_GAPS = {
    "ATR_3",
    "PIP5K1",
    "SLC16A4",
}

KNOWN_STRUCTURE_GAPS = set(KNOWN_BIOLOGIC_NO_SMILES) | KNOWN_UNRESOLVED_AGENT_STRUCTURE_GAPS


# -----------------------------------------------------------------------------
# Global audit state
# -----------------------------------------------------------------------------

CHECKS: list[dict] = []


# Suppress harmless source-workbook style warnings.
warnings.filterwarnings(
    "ignore",
    message="Workbook contains no default style.*",
)


# =============================================================================
# Helpers
# =============================================================================


def heading(text: str) -> None:
    print("\n" + "=" * 88)
    print(text)
    print("=" * 88)


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]

    x = float(n)

    for unit in units:
        if x < 1024:
            return f"{x:.2f} {unit}"
        x /= 1024

    return f"{x:.2f} EB"


def record(
    name: str,
    status: str,
    message: str,
    details: dict | None = None,
) -> None:
    """
    Add one audit result and print it.
    """

    status = status.upper()

    if status not in {"PASS", "WARN", "FAIL"}:
        raise ValueError(f"Invalid audit status: {status}")

    result = {
        "name": name,
        "status": status,
        "message": message,
        "details": details or {},
    }

    CHECKS.append(result)

    print(
        f"[{status}]".ljust(8),
        f"{name}: {message}",
    )


def normalize_cosmic(value) -> str | None:
    """
    Convert COSMIC identifiers to stable string keys.

    Handles values such as:
        924100
        924100.0
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    value = re.sub(
        r"\.0$",
        "",
        value,
    )

    if not value:
        return None

    if value.lower() in {
        "nan",
        "none",
        "null",
    }:
        return None

    return value


def normalize_name(value) -> str:
    """
    Loose identifier normalization for diagnostic comparisons.

    Examples:
        NCI-H226 -> NCIH226
        BT-474   -> BT474
        MDA-MB-231 -> MDAMB231
    """

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(value).upper(),
    )


def clean_structure(value) -> str | None:
    """
    Treat blank or whitespace-only chemistry entries as missing.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    if value.lower() in {
        "nan",
        "none",
        "null",
    }:
        return None

    return value


def systematic_sample(
    paths: list[Path],
    n: int,
) -> list[Path]:
    """
    Deterministic sample spread across sorted paths.
    """

    paths = sorted(paths)

    if len(paths) <= n:
        return paths

    if n <= 1:
        return [paths[0]]

    indexes = [round(i * (len(paths) - 1) / (n - 1)) for i in range(n)]

    return [paths[i] for i in indexes]


# =============================================================================
# Dose-response matrix parser
# =============================================================================


def parse_dose_matrix(
    df: pd.DataFrame,
) -> tuple[list[str], dict]:
    """
    Validate an AZ-DREAM matrix.

    Expected layout resembles:

                    Agent 2 doses
             0   d1   d2   d3 ...
       0     response...
    Agent1
       d1    response...
       d2    response...
       ...

       (=Agent 1)

       Agent1 ...
       Agent2 ...
       Unit1  ...
       Unit2  ...
       Title  ...

    Returns:
        errors: list[str]
        info:   dict
    """

    errors: list[str] = []

    if df.empty:
        return ["matrix is empty"], {}

    if df.shape[0] < 5 or df.shape[1] < 4:
        return [f"matrix is unexpectedly small: {df.shape}"], {}

    first_col = df.iloc[:, 0].astype(str).str.strip().str.lower().tolist()

    first_row = df.iloc[0, :].astype(str).str.strip().str.lower().tolist()

    try:
        row_marker = first_col.index("(=agent 1)")
    except ValueError:
        row_marker = None

    try:
        col_marker = first_row.index("(=agent 2)")
    except ValueError:
        col_marker = None

    if row_marker is None:
        errors.append("missing '(=Agent 1)' marker")

    if col_marker is None:
        errors.append("missing '(=Agent 2)' marker")

    if errors:
        return errors, {}

    if row_marker <= 1:
        errors.append("no usable Agent 1 concentration rows")

    if col_marker <= 1:
        errors.append("no usable Agent 2 concentration columns")

    if errors:
        return errors, {}

    # -------------------------------------------------------------------------
    # Concentration axes
    # -------------------------------------------------------------------------

    dose1 = pd.to_numeric(
        df.iloc[
            1:row_marker,
            0,
        ],
        errors="coerce",
    )

    dose2 = pd.to_numeric(
        df.iloc[
            0,
            1:col_marker,
        ],
        errors="coerce",
    )

    if dose1.isna().any():
        errors.append("Agent 1 concentration axis contains non-numeric values")

    if dose2.isna().any():
        errors.append("Agent 2 concentration axis contains non-numeric values")

    if len(dose1) < 2:
        errors.append("Agent 1 concentration axis contains fewer than 2 points")

    if len(dose2) < 2:
        errors.append("Agent 2 concentration axis contains fewer than 2 points")

    # -------------------------------------------------------------------------
    # Response grid
    # -------------------------------------------------------------------------

    responses = df.iloc[
        1:row_marker,
        1:col_marker,
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    numeric_count = int(responses.notna().sum().sum())

    response_count = int(responses.size)

    numeric_fraction = numeric_count / max(response_count, 1)

    if numeric_fraction < 0.95:
        errors.append(f"only {numeric_fraction:.1%} " "of response grid is numeric")

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    metadata = {}

    for i in range(
        row_marker + 1,
        df.shape[0],
    ):
        key = df.iat[i, 0]

        value = df.iat[i, 1] if df.shape[1] > 1 else None

        if pd.isna(key):
            continue

        key = str(key).strip()

        if key:
            metadata[key] = value

    required_metadata = [
        "Agent1",
        "Agent2",
        "Unit1",
        "Unit2",
        "Title",
    ]

    for field in required_metadata:
        if field not in metadata:
            errors.append(f"missing metadata field {field}")

    info = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "dose1_points": int(len(dose1)),
        "dose2_points": int(len(dose2)),
        "response_points": response_count,
        "numeric_response_fraction": float(numeric_fraction),
        "agent1": str(metadata.get("Agent1", "")),
        "agent2": str(metadata.get("Agent2", "")),
        "unit1": str(metadata.get("Unit1", "")),
        "unit2": str(metadata.get("Unit2", "")),
        "cell": str(metadata.get("Title", "")),
    }

    return errors, info


# =============================================================================
# Main audit
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=("Audit the dose-response dataset."))

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "Project root containing dataset/. " "Defaults to the directory containing this script."
        ),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=("Return exit code 2 when warnings exist."),
    )

    args = parser.parse_args()

    root = args.root.resolve()

    dataset = root / "dataset"
    raw = dataset / "raw"

    az = raw / "az_dream"

    dream_molecular = raw / "az_dream_molecular"

    cmp_dir = raw / "cell_model_passports"

    print("Dose-response Dataset Audit")

    print(
        "Project root :",
        root,
    )

    print(
        "Dataset root :",
        dataset,
    )

    print(
        "Timestamp    :",
        datetime.now(timezone.utc).isoformat(),
    )

    # =========================================================================
    # 1. Dataset structure
    # =========================================================================

    heading("1. DATASET STRUCTURE")

    required_paths = {
        "dataset.md": dataset / "dataset.md",
        "processed/cell_expression": dataset / "processed" / "cell_expression",
        "processed/drug_features": dataset / "processed" / "drug_features",
        "processed/dose_response": dataset / "processed" / "dose_response",
        "splits": dataset / "splits",
        "manifests": dataset / "manifests",
        "AZ-DREAM": az,
        "corrected DREAM matrices": az / "DREAM_Challenge_matrices_corrected",
        "corrected OI matrices": az / "DREAM_OI_matrices_corrected",
        "drug information": az / "Drug_info_release.csv",
        "supplemental chemistry": az
        / ("Extra DrugChemistryData to release " "for Nat Comms FINAL RELEASE_200219.xlsx"),
        "DREAM cell metadata": dream_molecular / "cell_info.csv",
        "DREAM expression": dream_molecular / "gex.csv",
        "DREAM molecular archive": dream_molecular / "Sanger_molecular_data.zip",
        "CMP model metadata": cmp_dir / "model_list_20260814.csv",
        "CMP gene identifiers": cmp_dir / "gene_identifiers_20241212.csv",
        "CMP RNA-seq": cmp_dir / "rnaseq_all_20260323.zip",
    }

    missing_paths = []

    for label, path in required_paths.items():

        if path.exists():

            if path.is_file():
                suffix = f" [{human_size(path.stat().st_size)}]"
            else:
                suffix = ""

            print(f"OK      {label}{suffix}")

        else:

            print(f"MISSING {label}")

            missing_paths.append(label)

    if missing_paths:

        record(
            "Required files",
            "FAIL",
            (f"{len(missing_paths)} " "required path(s) are missing."),
            {
                "missing": missing_paths,
            },
        )

    else:

        record(
            "Required files",
            "PASS",
            ("All critical " "source paths are present."),
        )

    # Cannot continue meaningfully without source roots.
    if not (az.exists() and dream_molecular.exists() and cmp_dir.exists()):
        return finish(
            dataset,
            args.strict,
        )

    # =========================================================================
    # 2. AZ-DREAM dose-response corpus
    # =========================================================================

    heading("2. AZ-DREAM DOSE-RESPONSE CORPUS")

    dream_matrix_dir = az / "DREAM_Challenge_matrices_corrected"

    oi_matrix_dir = az / "DREAM_OI_matrices_corrected"

    dream_matrices = sorted(dream_matrix_dir.glob("*.xlsx"))

    oi_matrices = sorted(oi_matrix_dir.glob("*.xlsx"))

    if len(dream_matrices) >= EXPECTED_DREAM_MATRICES:

        record(
            "Corrected DREAM matrices",
            "PASS",
            (f"{len(dream_matrices):,} found."),
        )

    else:

        record(
            "Corrected DREAM matrices",
            "FAIL",
            (
                f"{len(dream_matrices):,} found; "
                f"expected at least "
                f"{EXPECTED_DREAM_MATRICES:,}."
            ),
        )

    if len(oi_matrices) >= EXPECTED_OI_MATRICES:

        record(
            "Corrected OI matrices",
            "PASS",
            (f"{len(oi_matrices):,} found."),
        )

    else:

        record(
            "Corrected OI matrices",
            "FAIL",
            (f"{len(oi_matrices):,} found; " f"expected at least " f"{EXPECTED_OI_MATRICES:,}."),
        )

    # -------------------------------------------------------------------------
    # Parse corrected filenames
    # -------------------------------------------------------------------------

    used_drugs: set[str] = set()

    matrix_cell_counts = Counter()

    malformed_filenames = []

    for path in dream_matrices:

        parts = path.stem.split("&")

        if len(parts) < 4:

            malformed_filenames.append(path.name)

            continue

        drug_a = parts[0]
        drug_b = parts[1]
        cell = parts[2]

        used_drugs.add(drug_a)

        used_drugs.add(drug_b)

        matrix_cell_counts[cell] += 1

    print(
        "Unique drugs in corrected DREAM matrices:",
        len(used_drugs),
    )

    print(
        "Unique cell labels in corrected DREAM matrices:",
        len(matrix_cell_counts),
    )

    if len(used_drugs) == EXPECTED_DREAM_DRUGS:

        record(
            "DREAM drug population",
            "PASS",
            (f"{EXPECTED_DREAM_DRUGS} " "unique drugs found."),
        )

    else:

        record(
            "DREAM drug population",
            "WARN",
            (f"{len(used_drugs)} unique drugs found; " f"expected {EXPECTED_DREAM_DRUGS}."),
        )

    if malformed_filenames:

        record(
            "Matrix filename format",
            "WARN",
            (
                f"{len(malformed_filenames)} "
                "corrected matrix filename(s) "
                "do not follow the expected pattern."
            ),
            {
                "examples": malformed_filenames[:20],
            },
        )

    else:

        record(
            "Matrix filename format",
            "PASS",
            ("All corrected filenames follow " "the expected pattern."),
        )

    # =========================================================================
    # 3. DREAM cell metadata
    # =========================================================================

    heading("3. DREAM CELL METADATA")

    dream = None

    cell_info_path = dream_molecular / "cell_info.csv"

    try:

        dream = pd.read_csv(cell_info_path)

        required_cell_columns = {
            "Sanger.Name",
            "CCLE.Name",
            "Alternative.Name",
            "COSMIC",
        }

        missing_columns = required_cell_columns - set(dream.columns)

        if missing_columns:

            record(
                "DREAM cell metadata",
                "FAIL",
                ("Missing required columns: " f"{sorted(missing_columns)}"),
            )

        elif len(dream) != EXPECTED_DREAM_CELLS:

            record(
                "DREAM cell metadata",
                "WARN",
                (f"{len(dream)} cell lines found; " f"expected " f"{EXPECTED_DREAM_CELLS}."),
            )

        else:

            record(
                "DREAM cell metadata",
                "PASS",
                (f"{len(dream)} cell lines found " "with required identifiers."),
            )

    except Exception as e:

        record(
            "DREAM cell metadata",
            "FAIL",
            (f"Could not read cell_info.csv: {e}"),
        )

    # -------------------------------------------------------------------------
    # Original DREAM expression reference
    # -------------------------------------------------------------------------

    gex_path = dream_molecular / "gex.csv"

    try:

        gex = pd.read_csv(
            gex_path,
            low_memory=False,
        )

        gene_count = len(gex)

        cell_count = max(
            len(gex.columns) - 1,
            0,
        )

        print(f"DREAM gex.csv: " f"{gene_count:,} genes x " f"{cell_count} cell columns")

        if gene_count >= 10_000 and cell_count >= 80:

            record(
                "Original DREAM expression",
                "PASS",
                (f"{gene_count:,} genes x " f"{cell_count} cells."),
            )

        else:

            record(
                "Original DREAM expression",
                "WARN",
                (f"Unexpected expression shape: " f"{gex.shape}."),
            )

    except Exception as e:

        record(
            "Original DREAM expression",
            "WARN",
            (f"Could not inspect gex.csv: {e}"),
        )

    # -------------------------------------------------------------------------
    # Molecular archive integrity
    # -------------------------------------------------------------------------

    molecular_zip = dream_molecular / "Sanger_molecular_data.zip"

    try:

        with zipfile.ZipFile(molecular_zip) as z:

            members = z.namelist()

        expected_archive_files = [
            "cell_info.csv",
            "gex.csv",
            "mutations.csv",
            "cnv_gene.csv",
        ]

        absent = [
            name
            for name in expected_archive_files
            if not any(member.endswith(name) for member in members)
        ]

        if absent:

            record(
                "DREAM molecular archive",
                "WARN",
                ("Archive is readable but " "some expected entries were not found."),
                {
                    "missing": absent,
                },
            )

        else:

            record(
                "DREAM molecular archive",
                "PASS",
                (f"Archive is readable " f"with {len(members)} entries."),
            )

    except Exception as e:

        record(
            "DREAM molecular archive",
            "FAIL",
            (f"Could not open archive: {e}"),
        )

    # =========================================================================
    # 4. DREAM -> Cell Model Passports mapping
    # =========================================================================

    heading("4. DREAM -> CELL MODEL PASSPORT MAPPING")

    target_model_ids: set[str] = set()

    mapped = None

    model_path = cmp_dir / "model_list_20260814.csv"

    try:

        models = pd.read_csv(
            model_path,
            low_memory=False,
        )

        required_model_columns = {
            "model_id",
            "model_name",
            "COSMIC_ID",
        }

        missing_model_columns = required_model_columns - set(models.columns)

        if missing_model_columns:
            raise RuntimeError(("CMP model metadata lacks: " f"{sorted(missing_model_columns)}"))

        if dream is None:
            raise RuntimeError("DREAM cell metadata unavailable")

        dream_map = dream.copy()

        dream_map["COSMIC_KEY"] = dream_map["COSMIC"].map(normalize_cosmic)

        model_map = models[
            [
                "model_id",
                "model_name",
                "COSMIC_ID",
            ]
        ].copy()

        model_map["COSMIC_KEY"] = model_map["COSMIC_ID"].map(normalize_cosmic)

        model_map = model_map[model_map["COSMIC_KEY"].notna()]

        # ---------------------------------------------------------------------
        # Verify that DREAM COSMIC IDs do not map to multiple CMP model IDs.
        # ---------------------------------------------------------------------

        multiplicity = model_map.groupby("COSMIC_KEY")["model_id"].nunique()

        dream_cosmic_keys = set(dream_map["COSMIC_KEY"].dropna())

        ambiguous = sorted(
            key
            for key in dream_cosmic_keys
            if multiplicity.get(
                key,
                0,
            )
            > 1
        )

        if ambiguous:

            record(
                "DREAM-CMP COSMIC uniqueness",
                "FAIL",
                (f"{len(ambiguous)} DREAM " "COSMIC identifier(s) map " "to multiple CMP models."),
                {
                    "ambiguous_cosmic_ids": ambiguous,
                },
            )

        else:

            record(
                "DREAM-CMP COSMIC uniqueness",
                "PASS",
                ("DREAM COSMIC IDs map uniquely."),
            )

        # One lookup row per COSMIC identifier.
        model_map_unique = model_map.sort_values("model_id").drop_duplicates("COSMIC_KEY")

        mapped = dream_map.merge(
            model_map_unique,
            on="COSMIC_KEY",
            how="left",
        )

        matched_count = int(mapped["model_id"].notna().sum())

        if matched_count == len(dream) == EXPECTED_DREAM_CELLS:

            record(
                "DREAM-CMP cell mapping",
                "PASS",
                (f"{matched_count}/" f"{len(dream)} DREAM cells " "mapped by COSMIC ID."),
            )

        else:

            missing_cells = mapped.loc[
                mapped["model_id"].isna(),
                [
                    "Sanger.Name",
                    "COSMIC",
                ],
            ]

            record(
                "DREAM-CMP cell mapping",
                "FAIL",
                (f"{matched_count}/" f"{len(dream)} DREAM cells mapped."),
                {
                    "missing": missing_cells.to_dict(orient="records"),
                },
            )

        target_model_ids = set(mapped["model_id"].dropna().astype(str))

    except Exception as e:

        record(
            "DREAM-CMP cell mapping",
            "FAIL",
            (f"Could not perform mapping: {e}"),
        )

    # =========================================================================
    # 5. Cell Model Passports RNA-seq coverage
    # =========================================================================

    heading("5. CELL MODEL PASSPORT RNA-SEQ COVERAGE")

    rnaseq_zip = cmp_dir / "rnaseq_all_20260323.zip"

    if not target_model_ids:

        record(
            "RNA-seq DREAM-cell coverage",
            "FAIL",
            ("No mapped DREAM model IDs " "were available."),
        )

    else:

        try:

            with zipfile.ZipFile(rnaseq_zip) as z:

                csv_members = [name for name in z.namelist() if name.lower().endswith(".csv")]

                if not csv_members:
                    raise RuntimeError(("RNA-seq ZIP contains " "no CSV file."))

                member = next(
                    (name for name in csv_members if "rnaseq_all" in name.lower()),
                    csv_members[0],
                )

                zip_info = z.getinfo(member)

                print(
                    "RNA-seq archive member:",
                    member,
                )

                print(
                    "Uncompressed size:",
                    human_size(zip_info.file_size),
                )

                # -------------------------------------------------------------
                # Schema check
                # -------------------------------------------------------------

                with z.open(member) as f:

                    header = pd.read_csv(
                        f,
                        nrows=0,
                    )

                required_rna_columns = {
                    "model_id",
                    "gene_symbol",
                    "rsem_tpm",
                }

                missing_rna_columns = required_rna_columns - set(header.columns)

                if missing_rna_columns:

                    record(
                        "RNA-seq schema",
                        "FAIL",
                        ("Missing required columns: " f"{sorted(missing_rna_columns)}"),
                    )

                else:

                    record(
                        "RNA-seq schema",
                        "PASS",
                        ("model_id, gene_symbol and " "rsem_tpm present."),
                    )

                    present_ids = set()
                    usable_ids = set()

                    with z.open(member) as f:

                        reader = pd.read_csv(
                            f,
                            usecols=[
                                "model_id",
                                "rsem_tpm",
                            ],
                            chunksize=750_000,
                            low_memory=False,
                        )

                        for chunk in reader:

                            ids = chunk["model_id"].astype(str)

                            mask = ids.isin(target_model_ids)

                            if not mask.any():
                                continue

                            subset = chunk.loc[
                                mask,
                                [
                                    "model_id",
                                    "rsem_tpm",
                                ],
                            ].copy()

                            present_ids.update(subset["model_id"].dropna().astype(str))

                            numeric_tpm = pd.to_numeric(
                                subset["rsem_tpm"],
                                errors="coerce",
                            )

                            usable_ids.update(
                                subset.loc[
                                    numeric_tpm.notna(),
                                    "model_id",
                                ]
                                .dropna()
                                .astype(str)
                            )

                            # Stop once every target cell
                            # has at least one usable TPM value.
                            if usable_ids >= target_model_ids:
                                break

            print(
                "Mapped DREAM IDs appearing in RNA-seq:",
                (f"{len(present_ids)}/" f"{len(target_model_ids)}"),
            )

            print(
                "Mapped DREAM IDs with usable RSEM TPM:",
                (f"{len(usable_ids)}/" f"{len(target_model_ids)}"),
            )

            missing_usable = sorted(target_model_ids - usable_ids)

            if missing_usable:

                record(
                    "RNA-seq DREAM-cell coverage",
                    "FAIL",
                    (f"{len(missing_usable)} " "mapped DREAM cell(s) lack " "usable RSEM TPM."),
                    {
                        "missing_model_ids": missing_usable,
                    },
                )

            else:

                record(
                    "RNA-seq DREAM-cell coverage",
                    "PASS",
                    (
                        f"{len(usable_ids)}/"
                        f"{len(target_model_ids)} "
                        "mapped DREAM cells have "
                        "usable RSEM TPM."
                    ),
                )

        except Exception as e:

            record(
                "RNA-seq DREAM-cell coverage",
                "FAIL",
                ("Could not inspect RNA-seq " f"archive: {e}"),
            )

    # =========================================================================
    # 6. Drug chemistry coverage
    # =========================================================================

    heading("6. DRUG CHEMISTRY COVERAGE")

    primary_path = az / "Drug_info_release.csv"

    supplemental_path = az / (
        "Extra DrugChemistryData to release " "for Nat Comms FINAL RELEASE_200219.xlsx"
    )

    try:

        primary = pd.read_csv(
            primary_path,
            keep_default_na=False,
        )

        supplement = pd.read_excel(
            supplemental_path,
            sheet_name="Drug_Chemistry_Data",
            keep_default_na=False,
        )

        required_chemistry_columns = {
            "ChallengeName",
            CHEM_COL,
        }

        primary_missing_columns = required_chemistry_columns - set(primary.columns)

        supplement_missing_columns = required_chemistry_columns - set(supplement.columns)

        if primary_missing_columns:
            raise RuntimeError(
                ("Primary chemistry table lacks: " f"{sorted(primary_missing_columns)}")
            )

        if supplement_missing_columns:
            raise RuntimeError(
                ("Supplemental chemistry table lacks: " f"{sorted(supplement_missing_columns)}")
            )

        # ---------------------------------------------------------------------
        # Check duplicate challenge identifiers
        # ---------------------------------------------------------------------

        primary_duplicates = sorted(
            primary.loc[
                primary["ChallengeName"].duplicated(keep=False),
                "ChallengeName",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        supplement_duplicates = sorted(
            supplement.loc[
                supplement["ChallengeName"].duplicated(keep=False),
                "ChallengeName",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        if primary_duplicates:

            record(
                "Primary chemistry identifiers",
                "WARN",
                (f"{len(primary_duplicates)} " "duplicated ChallengeName " "identifier(s) found."),
                {
                    "duplicates": primary_duplicates,
                },
            )

        if supplement_duplicates:

            record(
                "Supplemental chemistry identifiers",
                "WARN",
                (
                    f"{len(supplement_duplicates)} "
                    "duplicated ChallengeName "
                    "identifier(s) found."
                ),
                {
                    "duplicates": supplement_duplicates,
                },
            )

        # ---------------------------------------------------------------------
        # Merge the two official releases by ChallengeName
        # ---------------------------------------------------------------------

        chem = primary[
            [
                "ChallengeName",
                CHEM_COL,
            ]
        ].merge(
            supplement[
                [
                    "ChallengeName",
                    CHEM_COL,
                ]
            ],
            on="ChallengeName",
            how="outer",
            suffixes=(
                "_primary",
                "_supplement",
            ),
        )

        chem["structure_primary"] = chem[f"{CHEM_COL}_primary"].map(clean_structure)

        chem["structure_supplement"] = chem[f"{CHEM_COL}_supplement"].map(clean_structure)

        # Prefer primary; otherwise use supplemental.
        chem["structure"] = chem["structure_primary"].combine_first(chem["structure_supplement"])

        unique_chemistry_names = set(chem["ChallengeName"].dropna().astype(str))

        names_with_structure = set(
            chem.loc[
                chem["structure"].notna(),
                "ChallengeName",
            ]
            .dropna()
            .astype(str)
        )

        names_without_structure = sorted(unique_chemistry_names - names_with_structure)

        print(
            "Unique chemistry entries:",
            len(unique_chemistry_names),
        )

        print(
            "Entries with directly usable " "SMILES/PubChem identifier:",
            len(names_with_structure),
        )

        print(
            "Entries without directly usable " "SMILES/PubChem identifier:",
            len(names_without_structure),
        )

        if names_without_structure:

            print(
                "Missing:",
                ", ".join(names_without_structure),
            )

        if len(unique_chemistry_names) >= 119:

            record(
                "Chemistry release population",
                "PASS",
                (f"{len(unique_chemistry_names)} " "unique challenge compounds/agents found."),
            )

        else:

            record(
                "Chemistry release population",
                "WARN",
                (f"{len(unique_chemistry_names)} " "unique challenge entries found."),
            )

        # ---------------------------------------------------------------------
        # Restrict to agents actually used in DREAM corrected matrices.
        # ---------------------------------------------------------------------

        missing_used = sorted(used_drugs - names_with_structure)

        known_biologic_missing = sorted(set(missing_used) & set(KNOWN_BIOLOGIC_NO_SMILES))

        known_unresolved_missing = sorted(set(missing_used) & KNOWN_UNRESOLVED_AGENT_STRUCTURE_GAPS)

        unexpected_missing = sorted(set(missing_used) - KNOWN_STRUCTURE_GAPS)

        print()

        print(
            "Used DREAM drugs/agents:",
            len(used_drugs),
        )

        print(
            "Used agents with directly usable chemistry:",
            (len(used_drugs) - len(missing_used)),
        )

        print(
            "Used agents without directly usable chemistry:",
            len(missing_used),
        )

        if known_biologic_missing:

            print("\nKnown biologics without " "small-molecule SMILES:")

            for name in known_biologic_missing:

                print(f"  {name}: " f"{KNOWN_BIOLOGIC_NO_SMILES[name]}")

        if known_unresolved_missing:

            print("\nKnown unresolved " "challenge agents:")

            for name in known_unresolved_missing:

                print(f"  {name}")

        if unexpected_missing:

            print("\nUnexpected missing " "chemistry entries:")

            for name in unexpected_missing:
                print(f"  {name}")

            record(
                "Used-drug chemistry coverage",
                "FAIL",
                (f"{len(unexpected_missing)} " "unexpected used agent(s) " "lack chemistry."),
                {
                    "unexpected_missing": unexpected_missing,
                    "known_biologics_without_smiles": {
                        name: KNOWN_BIOLOGIC_NO_SMILES[name] for name in known_biologic_missing
                    },
                    "known_unresolved_agents": known_unresolved_missing,
                },
            )

        elif missing_used:

            record(
                "Used-drug chemistry coverage",
                "WARN",
                (
                    f"{len(used_drugs) - len(missing_used)}/"
                    f"{len(used_drugs)} used DREAM agents "
                    "have directly usable "
                    "SMILES/PubChem identifiers; "
                    f"{len(missing_used)} known "
                    "source-level exception(s) remain."
                ),
                {
                    "biologics_without_small_molecule_smiles": {
                        name: KNOWN_BIOLOGIC_NO_SMILES[name] for name in known_biologic_missing
                    },
                    "unresolved_challenge_agents": known_unresolved_missing,
                },
            )

        else:

            record(
                "Used-drug chemistry coverage",
                "PASS",
                (f"All {len(used_drugs)} " "used DREAM agents have " "directly usable chemistry."),
            )

    except Exception as e:

        record(
            "Drug chemistry",
            "FAIL",
            (f"Could not audit " f"chemistry files: {e}"),
        )

    # =========================================================================
    # 7. Corrected matrix validation
    # =========================================================================

    heading("7. CORRECTED DOSE-MATRIX VALIDATION")

    corrected_sample = systematic_sample(
        dream_matrices,
        SAMPLE_MATRIX_COUNT,
    )

    structural_failures = []
    metadata_discrepancies = []

    for path in corrected_sample:

        try:

            df = pd.read_excel(
                path,
                sheet_name=0,
                header=None,
            )

            errors, info = parse_dose_matrix(df)

            if errors:

                structural_failures.append(
                    {
                        "file": path.name,
                        "errors": errors,
                        "info": info,
                    }
                )

                continue

            # -----------------------------------------------------------------
            # Metadata-vs-filename checks are warnings,
            # not structural failures.
            # -----------------------------------------------------------------

            parts = path.stem.split("&")

            if len(parts) >= 4:

                filename_drugs = {
                    normalize_name(parts[0]),
                    normalize_name(parts[1]),
                }

                metadata_drugs = {
                    normalize_name(info["agent1"]),
                    normalize_name(info["agent2"]),
                }

                if filename_drugs != metadata_drugs:

                    metadata_discrepancies.append(
                        {
                            "file": path.name,
                            "type": "drug-label mismatch",
                            "filename_drugs": sorted(filename_drugs),
                            "metadata_drugs": sorted(metadata_drugs),
                        }
                    )

                if normalize_name(parts[2]) != normalize_name(info["cell"]):

                    metadata_discrepancies.append(
                        {
                            "file": path.name,
                            "type": "cell-label mismatch",
                            "filename_cell": parts[2],
                            "matrix_title": info["cell"],
                        }
                    )

        except Exception as e:

            structural_failures.append(
                {
                    "file": path.name,
                    "errors": [str(e)],
                }
            )

    if structural_failures:

        record(
            "Corrected matrix parsing",
            "FAIL",
            (
                f"{len(structural_failures)}/"
                f"{len(corrected_sample)} "
                "sampled corrected matrices "
                "failed structural validation."
            ),
            {
                "failures": structural_failures,
            },
        )

    else:

        record(
            "Corrected matrix parsing",
            "PASS",
            (
                f"All {len(corrected_sample)} "
                "sampled matrices contain "
                "valid 2D dose-response grids."
            ),
        )

    if metadata_discrepancies:

        record(
            "Corrected matrix metadata labels",
            "WARN",
            (
                f"{len(metadata_discrepancies)} "
                "sampled metadata-label "
                "discrepancy/discrepancies found."
            ),
            {
                "discrepancies": metadata_discrepancies,
            },
        )

    else:

        record(
            "Corrected matrix metadata labels",
            "PASS",
            ("Sampled matrix metadata " "matches filenames."),
        )

    # =========================================================================
    # 8. Raw combination matrices
    # =========================================================================

    heading("8. RAW COMBINATION-MATRIX VALIDATION")

    raw_root = az / "Drug_synergy_data" / "Raw_Data_csv"

    raw_combination_files = []

    if raw_root.exists():

        for path in raw_root.rglob("*.csv"):

            relative = path.relative_to(raw_root)

            directory_parts = [part.lower() for part in relative.parts[:-1]]

            # Important:
            # inspect directory names ONLY.
            #
            # Do not search the entire absolute path,
            # because the project itself contains the word
            # "combination".
            is_combination_directory = any("combination" in part for part in directory_parts)

            is_monotherapy_directory = any("monotherapy" in part for part in directory_parts)

            if is_combination_directory and not is_monotherapy_directory:

                raw_combination_files.append(path)

    raw_combination_files = sorted(raw_combination_files)

    print(
        "Raw combination CSV files:",
        len(raw_combination_files),
    )

    if not raw_combination_files:

        record(
            "Raw combination files",
            "WARN",
            ("No extracted raw combination " "CSV files were found."),
        )

    else:

        record(
            "Raw combination files",
            "PASS",
            (f"{len(raw_combination_files):,} " "raw combination CSV files found."),
        )

        raw_sample = systematic_sample(
            raw_combination_files,
            SAMPLE_MATRIX_COUNT,
        )

        raw_failures = []

        for path in raw_sample:

            try:

                df = pd.read_csv(
                    path,
                    header=None,
                    engine="python",
                )

                errors, info = parse_dose_matrix(df)

                if errors:

                    raw_failures.append(
                        {
                            "file": str(path.relative_to(raw_root)),
                            "errors": errors,
                            "info": info,
                        }
                    )

            except Exception as e:

                raw_failures.append(
                    {
                        "file": str(path.relative_to(raw_root)),
                        "errors": [str(e)],
                    }
                )

        if raw_failures:

            record(
                "Raw matrix parsing",
                "FAIL",
                (
                    f"{len(raw_failures)}/"
                    f"{len(raw_sample)} "
                    "sampled raw combination "
                    "matrices failed validation."
                ),
                {
                    "failures": raw_failures,
                },
            )

        else:

            record(
                "Raw matrix parsing",
                "PASS",
                (
                    f"All {len(raw_sample)} "
                    "sampled raw combination "
                    "matrices contain valid "
                    "2D dose-response grids."
                ),
            )

    # =========================================================================
    # 9. Dose-response cell-label consistency
    # =========================================================================

    heading("9. DOSE-RESPONSE CELL-LABEL CONSISTENCY")

    if dream is None:

        record(
            "Dose-response cell labels",
            "FAIL",
            ("DREAM cell metadata unavailable."),
        )

    else:

        known_cell_names = set()

        for column in [
            "Sanger.Name",
            "Alternative.Name",
            "CCLE.Name",
        ]:

            if column not in dream.columns:
                continue

            known_cell_names.update(normalize_name(value) for value in dream[column].dropna())

        unmatched_cells = {
            cell: count
            for cell, count in sorted(matrix_cell_counts.items())
            if normalize_name(cell) not in known_cell_names
        }

        if unmatched_cells:

            print("Unmatched response cell labels:")

            for (
                cell,
                count,
            ) in unmatched_cells.items():

                print(f"  {cell}: " f"{count:,} matrix file(s)")

            record(
                "Dose-response cell labels",
                "WARN",
                (
                    f"{len(unmatched_cells)} "
                    "response cell label(s) require "
                    "explicit alias resolution."
                ),
                {
                    "unmatched": unmatched_cells,
                },
            )

        else:

            record(
                "Dose-response cell labels",
                "PASS",
                ("All response-matrix cell labels " "match DREAM metadata."),
            )

    # =========================================================================
    # Finish
    # =========================================================================

    return finish(
        dataset,
        args.strict,
    )


# =============================================================================
# Final report
# =============================================================================


def finish(
    dataset: Path,
    strict: bool,
) -> int:

    heading("FINAL AUDIT SUMMARY")

    pass_count = sum(item["status"] == "PASS" for item in CHECKS)

    warning_count = sum(item["status"] == "WARN" for item in CHECKS)

    fail_count = sum(item["status"] == "FAIL" for item in CHECKS)

    for check in CHECKS:

        print(f"{check['status']:4s}  " f"{check['name']}")

    print()

    print(
        "PASS :",
        pass_count,
    )

    print(
        "WARN :",
        warning_count,
    )

    print(
        "FAIL :",
        fail_count,
    )

    if fail_count:

        overall_status = "FAIL"

    elif warning_count:

        overall_status = "PASS_WITH_WARNINGS"

    else:

        overall_status = "PASS"

    print()

    print("=" * 88)

    print(
        "OVERALL DATASET STATUS:",
        overall_status,
    )

    print("=" * 88)

    if fail_count == 0 and warning_count > 0:

        print(
            "\nDataset acquisition and core structure "
            "are complete. Warnings describe known "
            "source-level limitations or identifier "
            "aliases that must be handled explicitly "
            "during preprocessing."
        )

    # -------------------------------------------------------------------------
    # Machine-readable manifest
    # -------------------------------------------------------------------------

    manifest_dir = dataset / "manifests"

    try:

        manifest_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_path = manifest_dir / "audit_report.json"

        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "overall_status": overall_status,
            "pass_count": pass_count,
            "warning_count": warning_count,
            "fail_count": fail_count,
            "expected_dataset_properties": {
                "dream_cells": EXPECTED_DREAM_CELLS,
                "dream_corrected_matrices": EXPECTED_DREAM_MATRICES,
                "oi_corrected_matrices": EXPECTED_OI_MATRICES,
                "dream_drugs": EXPECTED_DREAM_DRUGS,
            },
            "known_accepted_source_limitations": {
                "biologics_without_small_molecule_smiles": KNOWN_BIOLOGIC_NO_SMILES,
                "unresolved_challenge_agent_structures": sorted(
                    KNOWN_UNRESOLVED_AGENT_STRUCTURE_GAPS
                ),
            },
            "checks": CHECKS,
        }

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                report,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        print(
            "\nMachine-readable report:",
            report_path,
        )

    except Exception as e:

        print("\nWARNING: could not write " f"audit_report.json: {e}")

    # -------------------------------------------------------------------------
    # Exit status
    # -------------------------------------------------------------------------

    if fail_count:
        return 1

    if strict and warning_count:
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
