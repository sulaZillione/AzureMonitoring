import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


class DashboardTests(unittest.TestCase):
    def setUp(self):
        project_dir = Path(__file__).resolve().parents[1]
        self.project_dir = project_dir
        self.app = AppTest.from_file(project_dir / "app.py", default_timeout=60).run()

    def test_default_dashboard_and_meter_catalog_render(self):
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(
            [tab.label for tab in self.app.tabs],
            [
                "Overview",
                "Cost drivers",
                "Resource drill-through",
                "Meter catalog",
                "Details & data quality",
            ],
        )
        meter_tables = [frame.value for frame in self.app.dataframe if "ResourceGuid" in frame.value.columns]
        self.assertEqual(len(meter_tables), 1)
        self.assertEqual(len(meter_tables[0]), 52)
        detail_table = next(
            frame.value for frame in self.app.dataframe if "Billing meter GUID" in frame.value.columns
        )
        self.assertAlmostEqual(meter_tables[0]["FilteredCost"].sum(), detail_table["Cost"].sum(), places=6)
        self.assertEqual(len(self.app.get("plotly_chart")), 10)

    def test_monthly_comparison_uses_equal_elapsed_days(self):
        filters = dict(self.app.session_state["applied_filters"])
        filters["granularity"] = "Monthly"
        self.app.session_state["applied_filters"] = filters
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        comparison = next(metric for metric in self.app.metric if metric.label.startswith("Spend"))
        self.assertIn("Sep 01", comparison.label)
        self.assertIn("Sep 09", comparison.label)

    def test_full_meter_drill_path_renders(self):
        data = pd.read_csv(self.project_dir / "AzureUsage 1.csv")
        data["Date"] = pd.to_datetime(data["Date"])
        row = data.sort_values("Date").iloc[-1]
        path = [
            row["SubscriptionName"],
            row["ServiceName"],
            row["ServiceType"],
            row["ServiceRegion"],
            row["ServiceResource"],
            row["ResourceGuid"],
        ]
        self.app.session_state["drill_path"] = path
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.session_state["drill_path"]), 6)
        self.assertTrue(any(header.value.startswith("Billing meter") for header in self.app.subheader))

    def test_enrichment_button_merges_pricing_fields(self):
        meter_id = pd.read_csv(self.project_dir / "AzureUsage 1.csv").iloc[0]["ResourceGuid"]
        enrichment = [
            {
                "ResourceGuid": meter_id,
                "PricingStatus": "Matched",
                "MeterName": "Test meter",
                "ProductName": "Test product",
                "AzureSku": "Test_SKU",
                "ServiceFamily": "Compute",
                "UnitOfMeasure": "1 Hour",
                "PriceType": "Consumption",
                "RetailPriceUSD": 0.1,
                "RetailRegion": "centralindia",
                "EffectiveStartDate": "2026-01-01T00:00:00Z",
            }
        ]
        with patch("pricing_client.enrich_meter_records", return_value=enrichment):
            app = AppTest.from_file(self.project_dir / "app.py", default_timeout=60).run()
            button = next(item for item in app.button if item.label == "Enrich meter details")
            button.click()
            app.run()
        self.assertEqual(len(app.exception), 0)
        meter_table = next(frame.value for frame in app.dataframe if "ResourceGuid" in frame.value.columns)
        self.assertIn("MeterName", meter_table.columns)
        self.assertEqual(meter_table.loc[meter_table["ResourceGuid"] == meter_id, "MeterName"].iloc[0], "Test meter")


if __name__ == "__main__":
    unittest.main()
