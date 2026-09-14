import unittest

import pandas as pd

from data_helpers import add_resource_identity, resource_group_from_id


RESOURCE_ID = (
    "/subscriptions/sub-1/resourceGroups/parsed-rg/providers/"
    "Microsoft.Compute/virtualMachines/vm-production-01"
)


class DataHelperTests(unittest.TestCase):
    def test_resource_group_and_name_are_parsed_from_arm_id(self):
        normalized, metadata = add_resource_identity(pd.DataFrame({"ResourceId": [RESOURCE_ID]}))
        self.assertEqual(normalized.loc[0, "AnalysisResourceGroup"], "parsed-rg")
        self.assertEqual(normalized.loc[0, "AnalysisResourceName"], "vm-production-01")
        self.assertTrue(metadata["has_resource_group"])
        self.assertTrue(metadata["has_resource_identity"])

    def test_explicit_resource_fields_take_precedence(self):
        source = pd.DataFrame(
            {
                "ResourceId": [RESOURCE_ID, RESOURCE_ID],
                "ResourceGroupName": ["explicit-rg", ""],
                "InstanceName": ["explicit-vm", ""],
            }
        )
        normalized, metadata = add_resource_identity(source)
        self.assertEqual(normalized["AnalysisResourceGroup"].tolist(), ["explicit-rg", "parsed-rg"])
        self.assertEqual(normalized["AnalysisResourceName"].tolist(), ["explicit-vm", "vm-production-01"])
        self.assertEqual(metadata["resource_group_source"], "ResourceGroupName")
        self.assertEqual(metadata["identity_source"], "InstanceName")

    def test_absent_identity_is_marked_not_provided(self):
        normalized, metadata = add_resource_identity(pd.DataFrame({"Cost": [1.0]}))
        self.assertEqual(normalized.loc[0, "AnalysisResourceGroup"], "Not provided")
        self.assertEqual(normalized.loc[0, "AnalysisResourceName"], "Not provided")
        self.assertFalse(metadata["has_resource_group"])
        self.assertFalse(metadata["has_resource_identity"])

    def test_malformed_arm_id_does_not_raise(self):
        groups = resource_group_from_id(pd.Series(["not-an-arm-id", None]))
        self.assertEqual(groups.tolist(), ["Unknown", "Unknown"])


if __name__ == "__main__":
    unittest.main()
