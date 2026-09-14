"""Azure Retail Prices API client used to enrich legacy billing meter GUIDs."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import Lock
from time import monotonic
from typing import Callable
from uuid import UUID

import requests


RETAIL_PRICES_URL = "https://prices.azure.com/api/retail/prices"
CACHE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_TIMEOUT_SECONDS = 8
MAX_WORKERS = 6

_cache: dict[tuple, tuple[float, dict]] = {}
_cache_lock = Lock()


class PricingLookupError(RuntimeError):
    """Raised when Azure pricing data cannot be safely interpreted."""


def _text(value) -> str:
    return "" if value is None else str(value).strip()


def _normalized(value) -> str:
    return "".join(character for character in _text(value).lower() if character.isalnum())


def _is_guid(value: str) -> bool:
    try:
        UUID(value)
    except (ValueError, TypeError, AttributeError):
        return False
    return True


def fetch_meter_items(
    meter_id: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    get: Callable = requests.get,
) -> list[dict]:
    """Return all current public retail-price records for one billing meter."""
    if not _is_guid(meter_id):
        raise PricingLookupError("Invalid meter GUID")

    try:
        response = get(
            RETAIL_PRICES_URL,
            params={"$filter": f"meterId eq '{meter_id}'"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise PricingLookupError(f"Pricing API request failed: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise PricingLookupError("Pricing API returned invalid JSON") from exc

    if not isinstance(payload, dict):
        raise PricingLookupError("Pricing API returned an unexpected response")
    items = payload.get("Items", payload.get("items"))
    if not isinstance(items, list):
        raise PricingLookupError("Pricing API response did not contain an Items list")
    return [item for item in items if isinstance(item, dict)]


def _match_score(item: dict, context: dict) -> tuple[int, str]:
    """Rank exact-meter results against the service, SKU, and region in the CSV."""
    score = 0
    price_type = _normalized(item.get("type"))
    if price_type == "consumption":
        score += 40
    elif price_type == "devtestconsumption":
        score += 25
    elif price_type == "reservation":
        score -= 20

    if item.get("isPrimaryMeterRegion") is True:
        score += 5

    target_sku = _normalized(context.get("ServiceResource"))
    api_skus = [_normalized(item.get("skuName")), _normalized(item.get("armSkuName"))]
    if target_sku and any(target_sku == sku or target_sku in sku or sku in target_sku for sku in api_skus if sku):
        score += 25

    target_service = _normalized(context.get("ServiceName"))
    if target_service and target_service == _normalized(item.get("serviceName")):
        score += 15

    target_region = _normalized(context.get("ServiceRegion"))
    api_regions = [_normalized(item.get("location")), _normalized(item.get("armRegionName"))]
    if target_region and target_region in api_regions:
        score += 10

    effective_date = _text(item.get("effectiveStartDate"))
    return score, effective_date


def select_best_price(items: list[dict], context: dict) -> dict | None:
    """Select one reproducible benchmark record from possible pricing matches."""
    if not items:
        return None
    return max(items, key=lambda item: _match_score(item, context))


def _base_result(meter_id: str) -> dict:
    return {
        "ResourceGuid": meter_id,
        "PricingStatus": "Not found",
        "MeterName": None,
        "ProductName": None,
        "AzureSku": None,
        "ServiceFamily": None,
        "UnitOfMeasure": None,
        "PriceType": None,
        "RetailPriceUSD": None,
        "RetailRegion": None,
        "EffectiveStartDate": None,
        "PricingMatchCount": 0,
        "PricingError": None,
        "PricingRetrievedAt": datetime.now(timezone.utc).isoformat(),
    }


def lookup_meter(context: dict, timeout: int = DEFAULT_TIMEOUT_SECONDS, get: Callable = requests.get) -> dict:
    """Fetch and normalize one meter's best public retail-price benchmark."""
    meter_id = _text(context.get("ResourceGuid"))
    result = _base_result(meter_id)
    try:
        items = fetch_meter_items(meter_id, timeout=timeout, get=get)
        result["PricingMatchCount"] = len(items)
        selected = select_best_price(items, context)
        if selected is None:
            return result
        result.update(
            {
                "PricingStatus": "Matched",
                "MeterName": selected.get("meterName"),
                "ProductName": selected.get("productName"),
                "AzureSku": selected.get("armSkuName") or selected.get("skuName"),
                "ServiceFamily": selected.get("serviceFamily"),
                "UnitOfMeasure": selected.get("unitOfMeasure"),
                "PriceType": selected.get("type"),
                "RetailPriceUSD": selected.get("retailPrice"),
                "RetailRegion": selected.get("armRegionName") or selected.get("location"),
                "EffectiveStartDate": selected.get("effectiveStartDate"),
            }
        )
    except PricingLookupError as exc:
        result["PricingStatus"] = "Unavailable"
        result["PricingError"] = str(exc)
    return result


def _cache_key(context: dict) -> tuple:
    return (
        _text(context.get("ResourceGuid")),
        _text(context.get("ServiceName")),
        _text(context.get("ServiceType")),
        _text(context.get("ServiceRegion")),
        _text(context.get("ServiceResource")),
    )


def clear_price_cache() -> None:
    """Clear in-process enrichment results. Primarily useful for tests."""
    with _cache_lock:
        _cache.clear()


def lookup_meter_cached(
    context: dict,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    get: Callable = requests.get,
    ttl_seconds: int = CACHE_TTL_SECONDS,
    clock: Callable = monotonic,
) -> dict:
    """Look up a meter and cache the result, including failures, for the TTL."""
    key = _cache_key(context)
    now = clock()
    with _cache_lock:
        cached = _cache.get(key)
        if cached and now - cached[0] < ttl_seconds:
            return cached[1].copy()

    result = lookup_meter(context, timeout=timeout, get=get)
    with _cache_lock:
        _cache[key] = (now, result.copy())
    return result


def enrich_meter_records(
    records: list[dict],
    lookup: Callable[[dict], dict] = lookup_meter_cached,
    max_workers: int = MAX_WORKERS,
) -> list[dict]:
    """Enrich unique meters concurrently while retaining deterministic order."""
    unique_records = {}
    for record in records:
        meter_id = _text(record.get("ResourceGuid"))
        unique_records.setdefault(meter_id, record)

    results = {}
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(unique_records) or 1))) as executor:
        futures = {
            executor.submit(lookup, record): meter_id
            for meter_id, record in unique_records.items()
        }
        for future in as_completed(futures):
            meter_id = futures[future]
            try:
                results[meter_id] = future.result()
            except Exception as exc:  # Keep the dashboard usable if one worker fails unexpectedly.
                result = _base_result(meter_id)
                result["PricingStatus"] = "Unavailable"
                result["PricingError"] = f"Unexpected lookup failure: {exc}"
                results[meter_id] = result

    return [results[meter_id] for meter_id in unique_records]
