import unittest
from unittest.mock import Mock

import requests

from pricing_client import (
    PricingLookupError,
    clear_price_cache,
    enrich_meter_records,
    fetch_meter_items,
    lookup_meter,
    lookup_meter_cached,
    select_best_price,
)


METER_ID = "bf433558-fab7-4b7e-ab4a-07e1099377a8"
CONTEXT = {
    "ResourceGuid": METER_ID,
    "ServiceName": "Virtual Machines",
    "ServiceType": "BS Series",
    "ServiceRegion": "IN Central",
    "ServiceResource": "B2ms",
}


class FakeResponse:
    def __init__(self, payload=None, json_error=None):
        self.payload = payload
        self.json_error = json_error

    def raise_for_status(self):
        return None

    def json(self):
        if self.json_error:
            raise self.json_error
        return self.payload


class PricingClientTests(unittest.TestCase):
    def setUp(self):
        clear_price_cache()

    def test_select_best_price_prefers_consumption_and_matching_sku(self):
        items = [
            {
                "type": "Reservation",
                "skuName": "B2ms",
                "serviceName": "Virtual Machines",
                "effectiveStartDate": "2026-01-01T00:00:00Z",
            },
            {
                "type": "Consumption",
                "skuName": "B2s",
                "serviceName": "Virtual Machines",
                "effectiveStartDate": "2026-02-01T00:00:00Z",
            },
            {
                "type": "Consumption",
                "skuName": "B2ms",
                "serviceName": "Virtual Machines",
                "location": "IN Central",
                "effectiveStartDate": "2025-01-01T00:00:00Z",
            },
        ]
        selected = select_best_price(items, CONTEXT)
        self.assertEqual(selected["skuName"], "B2ms")
        self.assertEqual(selected["type"], "Consumption")

    def test_lookup_meter_normalizes_api_result(self):
        get = Mock(
            return_value=FakeResponse(
                {
                    "Items": [
                        {
                            "meterId": METER_ID,
                            "meterName": "B2ms",
                            "productName": "Virtual Machines BS Series",
                            "armSkuName": "Standard_B2ms",
                            "serviceFamily": "Compute",
                            "unitOfMeasure": "1 Hour",
                            "type": "Consumption",
                            "retailPrice": 0.09,
                            "armRegionName": "centralindia",
                            "effectiveStartDate": "2025-01-01T00:00:00Z",
                            "isPrimaryMeterRegion": True,
                        }
                    ]
                }
            )
        )
        result = lookup_meter(CONTEXT, get=get)
        self.assertEqual(result["PricingStatus"], "Matched")
        self.assertEqual(result["AzureSku"], "Standard_B2ms")
        self.assertEqual(result["UnitOfMeasure"], "1 Hour")
        self.assertEqual(result["RetailPriceUSD"], 0.09)

    def test_lookup_meter_returns_not_found_for_empty_items(self):
        result = lookup_meter(CONTEXT, get=Mock(return_value=FakeResponse({"Items": []})))
        self.assertEqual(result["PricingStatus"], "Not found")
        self.assertEqual(result["PricingMatchCount"], 0)

    def test_timeout_becomes_unavailable_without_raising(self):
        get = Mock(side_effect=requests.Timeout("timed out"))
        result = lookup_meter(CONTEXT, get=get)
        self.assertEqual(result["PricingStatus"], "Unavailable")
        self.assertIn("timed out", result["PricingError"])

    def test_malformed_response_is_rejected(self):
        get = Mock(return_value=FakeResponse({"unexpected": []}))
        with self.assertRaises(PricingLookupError):
            fetch_meter_items(METER_ID, get=get)

    def test_invalid_guid_is_rejected_without_network_call(self):
        get = Mock()
        with self.assertRaises(PricingLookupError):
            fetch_meter_items("not-a-guid", get=get)
        get.assert_not_called()

    def test_cache_prevents_repeat_calls(self):
        get = Mock(return_value=FakeResponse({"Items": []}))
        first = lookup_meter_cached(CONTEXT, get=get)
        second = lookup_meter_cached(CONTEXT, get=get)
        self.assertEqual(first, second)
        self.assertEqual(get.call_count, 1)

    def test_concurrent_enrichment_deduplicates_meter_ids(self):
        calls = []

        def lookup(record):
            calls.append(record["ResourceGuid"])
            return {"ResourceGuid": record["ResourceGuid"], "PricingStatus": "Matched"}

        results = enrich_meter_records([CONTEXT, CONTEXT.copy()], lookup=lookup, max_workers=2)
        self.assertEqual(len(results), 1)
        self.assertEqual(calls, [METER_ID])


if __name__ == "__main__":
    unittest.main()
