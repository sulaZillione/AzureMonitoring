"""CSV resource-identity normalization for Azure usage exports."""

import pandas as pd


RESOURCE_NAME_COLUMNS = [
    "ResourceName",
    "InstanceName",
    "ResourceDisplayName",
    "VMName",
    "VirtualMachineName",
]
RESOURCE_ID_COLUMNS = ["ResourceId", "ResourceID", "InstanceId", "ResourceUri", "ResourceURI"]
RESOURCE_GROUP_COLUMNS = ["ResourceGroup", "ResourceGroupName"]


def find_column(columns, candidates):
    lookup = {column.lower(): column for column in columns}
    return next((lookup[name.lower()] for name in candidates if name.lower() in lookup), None)


def resource_name_from_id(resource_ids: pd.Series) -> pd.Series:
    return resource_ids.fillna("").astype(str).str.rstrip("/").str.split("/").str[-1].replace("", "Unknown")


def resource_group_from_id(resource_ids: pd.Series) -> pd.Series:
    def extract(value):
        parts = str(value).strip("/").split("/")
        lowered = [part.lower() for part in parts]
        if "resourcegroups" not in lowered:
            return "Unknown"
        group_index = lowered.index("resourcegroups") + 1
        return parts[group_index] if group_index < len(parts) and parts[group_index] else "Unknown"

    return resource_ids.fillna("").map(extract)


def add_resource_identity(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add normalized resource name, group, and ID columns without inventing identity."""
    df = df.copy()
    resource_name_column = find_column(df.columns, RESOURCE_NAME_COLUMNS)
    resource_id_column = find_column(df.columns, RESOURCE_ID_COLUMNS)
    resource_group_column = find_column(df.columns, RESOURCE_GROUP_COLUMNS)

    if resource_name_column:
        explicit_names = df[resource_name_column].fillna("").astype(str).str.strip()
        if resource_id_column:
            parsed_names = resource_name_from_id(df[resource_id_column])
            df["AnalysisResourceName"] = explicit_names.where(explicit_names.ne(""), parsed_names)
        else:
            df["AnalysisResourceName"] = explicit_names.replace("", "Unknown")
        identity_source = resource_name_column
    elif resource_id_column:
        df["AnalysisResourceName"] = resource_name_from_id(df[resource_id_column])
        identity_source = resource_id_column
    else:
        df["AnalysisResourceName"] = "Not provided"
        identity_source = None

    has_resource_identity = df["AnalysisResourceName"].ne("Unknown").any() and identity_source is not None

    if resource_group_column:
        explicit_groups = df[resource_group_column].fillna("").astype(str).str.strip()
        if resource_id_column:
            parsed_groups = resource_group_from_id(df[resource_id_column])
            df["AnalysisResourceGroup"] = explicit_groups.where(explicit_groups.ne(""), parsed_groups)
        else:
            df["AnalysisResourceGroup"] = explicit_groups.replace("", "Unknown")
        resource_group_source = resource_group_column
    elif resource_id_column:
        df["AnalysisResourceGroup"] = resource_group_from_id(df[resource_id_column])
        resource_group_source = resource_id_column
    else:
        df["AnalysisResourceGroup"] = "Not provided"
        resource_group_source = None

    has_resource_group = (
        df["AnalysisResourceGroup"].ne("Unknown").any()
        and resource_group_source is not None
    )
    if resource_id_column:
        df["AnalysisResourceId"] = df[resource_id_column].fillna("Unknown").astype(str)
    else:
        df["AnalysisResourceId"] = "Not provided by this export"

    metadata = {
        "identity_label": "Resource name",
        "identity_source": identity_source,
        "has_resource_identity": has_resource_identity,
        "resource_id_column": resource_id_column,
        "has_resource_group": has_resource_group,
        "resource_group_source": resource_group_source,
    }
    return df, metadata
