import io
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from data_loader import load_usage_sources


class NamedBytesIO(io.BytesIO):
    def __init__(self, content: bytes, name: str):
        super().__init__(content)
        self.name = name


class DataLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project_dir = Path(__file__).resolve().parents[1]
        cls.sample = pd.read_csv(project_dir / "AzureUsage 1.csv").head(4)

    def csv_source(self, frame: pd.DataFrame, name: str) -> NamedBytesIO:
        return NamedBytesIO(frame.to_csv(index=False).encode("utf-8"), name)

    def test_combines_multiple_csv_files_and_retains_provenance(self):
        first = self.sample.iloc[:2].copy()
        second = self.sample.iloc[2:].copy()
        second["SubscriptionName"] = "Second sponsorship"

        combined, metadata = load_usage_sources(
            [self.csv_source(first, "pie.csv"), self.csv_source(second, "mwt.csv")]
        )

        self.assertEqual(len(combined), 4)
        self.assertEqual(metadata["source_file_count"], 2)
        self.assertEqual(set(combined["AnalysisSourceFile"]), {"pie.csv", "mwt.csv"})
        self.assertEqual(
            set(combined["SubscriptionName"]),
            {first.iloc[0]["SubscriptionName"], "Second sponsorship"},
        )
        self.assertIn("AnalysisSourceFile", metadata["download_columns"])

    def test_excel_loads_only_worksheets_with_usage_schema(self):
        workbook = Workbook()
        instructions = workbook.active
        instructions.title = "Instructions"
        instructions.append(["Notes"])
        instructions.append(["Read me"])
        for sheet_name, frame in {
            "PIE": self.sample.iloc[:2],
            "MWT": self.sample.iloc[2:],
        }.items():
            sheet = workbook.create_sheet(sheet_name)
            sheet.append(frame.columns.tolist())
            for values in frame.itertuples(index=False, name=None):
                sheet.append(
                    [value.item() if hasattr(value, "item") else value for value in values]
                )
        content = io.BytesIO()
        workbook.save(content)
        source = NamedBytesIO(content.getvalue(), "sponsorships.xlsx")

        combined, metadata = load_usage_sources([source])

        self.assertEqual(len(combined), 4)
        self.assertEqual(metadata["source_table_count"], 2)
        self.assertEqual(set(combined["AnalysisSourceSheet"]), {"PIE", "MWT"})

    def test_invalid_file_identifies_the_source(self):
        invalid = self.csv_source(pd.DataFrame({"Date": ["2026-01-01"]}), "broken.csv")
        with self.assertRaisesRegex(ValueError, "broken.csv: missing required columns"):
            load_usage_sources([invalid])

    def test_repairs_partial_excel_month_day_conversion(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Usage"
        sheet.append(self.sample.columns.tolist())
        rows = self.sample.iloc[:3].copy()
        dates = ["8/20/2025", datetime(2025, 10, 9), datetime(2026, 12, 8)]
        for values, usage_date in zip(rows.itertuples(index=False, name=None), dates):
            output = [value.item() if hasattr(value, "item") else value for value in values]
            output[rows.columns.get_loc("Date")] = usage_date
            sheet.append(output)
        content = io.BytesIO()
        workbook.save(content)
        source = NamedBytesIO(content.getvalue(), "mixed-dates.xlsx")

        combined, metadata = load_usage_sources([source])

        self.assertEqual(
            combined["Date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2025-08-20", "2025-09-10", "2026-08-12"],
        )
        self.assertEqual(metadata["date_corrections"], 2)


if __name__ == "__main__":
    unittest.main()
