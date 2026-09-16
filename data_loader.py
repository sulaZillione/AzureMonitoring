"""Load and normalize one or more Azure usage CSV/XLSX exports."""

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from data_helpers import add_resource_identity


REQUIRED_COLUMNS = {
    "SubscriptionName",
    "Date",
    "ResourceGuid",
    "ServiceName",
    "ServiceType",
    "ServiceRegion",
    "ServiceResource",
    "Quantity",
    "Cost",
}


def _source_name(source) -> str:
    name = getattr(source, "name", None)
    if name:
        return Path(str(name)).name
    if isinstance(source, (str, Path)):
        return Path(source).name
    return "Uploaded data"


def _rewind(source) -> None:
    if hasattr(source, "seek"):
        source.seek(0)


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = frame.columns.astype(str).str.strip()
    duplicates = frame.columns[frame.columns.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"duplicate column names after trimming: {', '.join(duplicates)}")
    return frame


def normalize_usage_dates(values: pd.Series) -> tuple[pd.Series, int]:
    """Parse usage dates and repair Excel's partial day/month conversion pattern."""
    native_mask = values.map(lambda value: isinstance(value, (date, datetime, pd.Timestamp)))
    text_mask = values.notna() & ~native_mask
    text_values = values.loc[text_mask].astype(str).str.strip()
    date_parts = text_values.str.extract(
        r"^(?P<first>\d{1,2})[/-](?P<second>\d{1,2})[/-](?P<year>\d{4})(?:\s.*)?$"
    ).apply(pd.to_numeric, errors="coerce")
    month_first_evidence = (
        date_parts["first"].between(1, 12) & date_parts["second"].gt(12)
    ).any()
    day_first_evidence = (
        date_parts["first"].gt(12) & date_parts["second"].between(1, 12)
    ).any()

    native_dates = pd.to_datetime(values.loc[native_mask], errors="coerce")
    partial_excel_conversion = (
        native_mask.any()
        and text_mask.any()
        and month_first_evidence
        and not day_first_evidence
        and native_dates.notna().all()
        and native_dates.dt.day.le(12).all()
    )

    if not partial_excel_conversion:
        return pd.to_datetime(values, errors="coerce", format="mixed"), 0

    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    parsed.loc[text_mask] = pd.to_datetime(
        text_values,
        errors="coerce",
        format="mixed",
        dayfirst=False,
    )
    corrected_native = native_dates.map(
        lambda value: value.replace(month=value.day, day=value.month)
    )
    parsed.loc[native_mask] = corrected_native
    correction_count = int(
        (native_dates.dt.month.to_numpy() != native_dates.dt.day.to_numpy()).sum()
    )
    return parsed, correction_count


def _read_tables(source, display_name: str) -> list[tuple[str, pd.DataFrame]]:
    suffix = Path(display_name).suffix.lower()
    try:
        _rewind(source)
        if suffix == ".csv":
            return [("CSV", _clean_columns(pd.read_csv(source)))]
        if suffix == ".xlsx":
            worksheets = pd.read_excel(source, sheet_name=None, engine="openpyxl")
            matching = []
            for sheet_name, frame in worksheets.items():
                cleaned = _clean_columns(frame)
                if REQUIRED_COLUMNS.issubset(cleaned.columns):
                    matching.append((str(sheet_name), cleaned))
            if matching:
                return matching
            raise ValueError("no worksheet contains all required Azure usage columns")
        raise ValueError("only .csv and .xlsx files are supported")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not be read ({exc})") from exc


def _normalize_table(
    frame: pd.DataFrame,
    source_file: str,
    source_sheet: str,
) -> tuple[pd.DataFrame, dict, list[str], int]:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("contains headers but no usage rows")

    parsed_dates, date_corrections = normalize_usage_dates(frame["Date"])
    parsed_cost = pd.to_numeric(frame["Cost"], errors="coerce")
    parsed_quantity = pd.to_numeric(frame["Quantity"], errors="coerce")
    problems = []
    if parsed_dates.isna().any():
        problems.append(f"{parsed_dates.isna().sum():,} invalid Date values")
    if parsed_cost.isna().any():
        problems.append(f"{parsed_cost.isna().sum():,} invalid Cost values")
    if parsed_quantity.isna().any():
        problems.append(f"{parsed_quantity.isna().sum():,} invalid Quantity values")
    if problems:
        raise ValueError("could not be safely analysed: " + "; ".join(problems))

    frame = frame.copy()
    frame["Date"] = parsed_dates
    frame["Cost"] = parsed_cost
    frame["Quantity"] = parsed_quantity
    for column in [
        "SubscriptionName",
        "ServiceName",
        "ServiceType",
        "ServiceRegion",
        "ServiceResource",
        "ResourceGuid",
    ]:
        frame[column] = frame[column].fillna("Unknown").astype(str)

    original_columns = list(frame.columns)
    frame, identity_metadata = add_resource_identity(frame)
    frame["AnalysisSourceFile"] = source_file
    frame["AnalysisSourceSheet"] = source_sheet
    return frame, identity_metadata, original_columns, date_corrections


def load_usage_sources(sources) -> tuple[pd.DataFrame, dict]:
    """Combine CSV/XLSX sources while retaining row-level source provenance."""
    if not isinstance(sources, (list, tuple)):
        sources = (sources,)
    if not sources:
        raise ValueError("Select at least one Azure usage file.")

    frames = []
    summaries = []
    original_columns = []
    identity_sources = set()
    group_sources = set()
    resource_id_sources = set()
    has_resource_identity = False
    has_resource_group = False
    total_date_corrections = 0
    filename_occurrences: dict[str, int] = {}

    for source in sources:
        filename = _source_name(source)
        filename_occurrences[filename] = filename_occurrences.get(filename, 0) + 1
        occurrence = filename_occurrences[filename]
        source_label = filename if occurrence == 1 else f"{filename} ({occurrence})"

        try:
            tables = _read_tables(source, filename)
        except ValueError as exc:
            raise ValueError(f"{source_label}: {exc}") from exc

        for sheet_name, raw_frame in tables:
            try:
                frame, identity_metadata, table_columns, date_corrections = _normalize_table(
                    raw_frame,
                    source_label,
                    sheet_name,
                )
            except ValueError as exc:
                location = source_label if sheet_name == "CSV" else f"{source_label} / {sheet_name}"
                raise ValueError(f"{location}: {exc}") from exc

            frames.append(frame)
            for column in table_columns:
                if column not in original_columns:
                    original_columns.append(column)
            if identity_metadata["identity_source"]:
                identity_sources.add(identity_metadata["identity_source"])
            if identity_metadata["resource_group_source"]:
                group_sources.add(identity_metadata["resource_group_source"])
            if identity_metadata["resource_id_column"]:
                resource_id_sources.add(identity_metadata["resource_id_column"])
            has_resource_identity = (
                has_resource_identity or identity_metadata["has_resource_identity"]
            )
            has_resource_group = has_resource_group or identity_metadata["has_resource_group"]
            total_date_corrections += date_corrections

            summaries.append(
                {
                    "Source file": source_label,
                    "Worksheet": sheet_name,
                    "Rows": len(frame),
                    "Subscriptions": frame["SubscriptionName"].nunique(),
                    "Cost": frame["Cost"].sum(),
                    "First usage": frame["Date"].min(),
                    "Last usage": frame["Date"].max(),
                    "Excel date corrections": date_corrections,
                }
            )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    source_columns = ["AnalysisSourceFile", "AnalysisSourceSheet"]
    download_columns = original_columns + [
        column for column in source_columns if column not in original_columns
    ]
    metadata = {
        "original_columns": original_columns,
        "download_columns": download_columns,
        "identity_label": "Resource name",
        "identity_source": ", ".join(sorted(identity_sources)) or None,
        "has_resource_identity": has_resource_identity,
        "resource_id_column": ", ".join(sorted(resource_id_sources)) or None,
        "has_resource_group": has_resource_group,
        "resource_group_source": ", ".join(sorted(group_sources)) or None,
        "source_file_count": len(sources),
        "source_table_count": len(summaries),
        "source_summaries": summaries,
        "date_corrections": total_date_corrections,
    }
    return combined, metadata
