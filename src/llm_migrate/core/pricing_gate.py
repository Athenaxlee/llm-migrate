"""Unit-aware registry-contradiction gate for adapted pricing values (V1.6.0-c).

The field run accepted an adapted config that priced the target at
0.0022/0.011 per THOUSAND tokens, justified by a "documented regional
premium", while the run's own registry facts said $2/$10 per MILLION with
no premium. A naive numeric comparison would either always or never fire
(the values differ by 1000x in scale and 1.1x in substance), so every
comparison here is unit-aware:

- values are read as `Decimal` (from their string form, never binary floats);
- a key that names its unit (`per_1k`, `per_million`, `per_token`, ...) is
  read in that unit only; otherwise the three standard scales are tried;
- a value equal to the TARGET's registry price under some scale is
  consistent; equal to the SOURCE's price is `stale` (source pricing left in
  the adapted config); within 0.5x-2x of the target price under the closest
  scale is a `contradiction` reporting the implied factor; anything else is
  `unit_unrecognized` — never silently passed.

Only values the scanner's PRICING couplings already identified are checked,
and only in profiles that govern this migration. A contradiction is cleared
when the change that set the value cites registry-recorded or plan-carried
evidence for the departure; an unknown-evidence citation cannot clear it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from llm_migrate.core.models import ModelPricing, PricePerMillionTokens

Scale = Literal["per_token", "per_1k", "per_1m"]
Side = Literal["input", "output", "cached_input"]
Verdict = Literal["consistent", "stale", "contradiction", "unit_unrecognized"]

# Multiplier that turns a value in each scale into a per-1M-token price.
_TO_PER_MILLION: dict[Scale, Decimal] = {
    "per_token": Decimal(1_000_000),
    "per_1k": Decimal(1000),
    "per_1m": Decimal(1),
}
_SCALE_LABELS: dict[Scale, str] = {
    "per_token": "per token",
    "per_1k": "per 1K tokens",
    "per_1m": "per 1M tokens",
}
_CONTRADICTION_BAND = (Decimal("0.5"), Decimal(2))


@dataclass(frozen=True)
class PricingCheck:
    """The verdict for one adapted pricing value."""

    key_path: str
    raw_value: str
    verdict: Verdict
    side: Side | None = None
    scale: Scale | None = None
    per_million: Decimal | None = None
    target_price: Decimal | None = None
    source_price: Decimal | None = None
    factor: Decimal | None = None
    expiry_note: str | None = None

    @property
    def message(self) -> str:
        where = f"{self.key_path} = {self.raw_value}"
        unit = _SCALE_LABELS[self.scale] if self.scale else ""
        side = self.side.replace("_", " ") if self.side else "token"
        target = f"${_money(self.target_price)} per 1M" if self.target_price is not None else "?"
        if self.verdict == "contradiction":
            text = (
                f"{where} reads as ${_money(self.per_million)} per 1M tokens ({unit}), but the "
                f"registry records the target's {side} price as {target}: an implied factor of "
                f"{_factor(self.factor)}x. Cite registry-recorded or plan-carried evidence for "
                "the departure in the change's annotation, or correct the value"
            )
        elif self.verdict == "stale":
            text = (
                f"{where} equals the SOURCE model's registry {side} price "
                f"(${_money(self.source_price)} per 1M, read {unit}); the adapted configuration "
                f"still carries source pricing (the target's is {target})"
            )
        else:
            text = (
                f"{where} matches no standard pricing scale (per token / per 1K / per 1M) "
                f"within 0.5x-2x of the target's registry {side} price ({target}); state the "
                "unit in the key or correct the value"
            )
        return text + (f". {self.expiry_note}" if self.expiry_note else "")


def _money(value: Decimal | None) -> str:
    if value is None:
        return "?"
    return format(value.normalize(), "f")


def _factor(value: Decimal | None) -> str:
    if value is None:
        return "?"
    return format(value.quantize(Decimal("0.001")).normalize(), "f")


def to_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return number if number.is_finite() and number > 0 else None


def key_scale(key: str) -> Scale | None:
    """The unit a pricing key names explicitly, if any."""
    lowered = key.casefold().replace("-", "_")
    if any(marker in lowered for marker in ("1m", "per_m", "million", "per_1000000")):
        return "per_1m"
    if any(marker in lowered for marker in ("1k", "per_k", "thousand", "_k_", "per_1000")):
        return "per_1k"
    if any(marker in lowered for marker in ("per_token", "per_tok")):
        return "per_token"
    return None


def key_sides(key_path: tuple[str, ...]) -> list[Side]:
    """Which price a pricing value is (input/output/cached), from its key path."""
    lowered = " ".join(key_path).casefold()
    if "cache" in lowered:
        return ["cached_input"]
    if any(marker in lowered for marker in ("output", "completion", "response")):
        return ["output"]
    if any(marker in lowered for marker in ("input", "prompt", "request")):
        return ["input"]
    return ["input", "output"]


def _prices(pricing: ModelPricing | None) -> dict[Side, PricePerMillionTokens]:
    if pricing is None:
        return {}
    items: dict[Side, PricePerMillionTokens | None] = {
        "input": pricing.input,
        "output": pricing.output,
        "cached_input": pricing.cached_input,
    }
    return {side: price for side, price in items.items() if price is not None}


def classify_price(
    key_path: tuple[str, ...],
    value: Any,
    *,
    target: ModelPricing | None,
    source: ModelPricing | None,
    as_of: date,
) -> PricingCheck | None:
    """Classify one adapted pricing value; None when it cannot be compared."""
    number = to_decimal(value)
    target_prices = _prices(target)
    if number is None or not target_prices:
        return None
    source_prices = _prices(source)
    sides = [side for side in key_sides(key_path) if side in target_prices]
    if not sides:
        return None
    named = key_scale(key_path[-1]) if key_path else None
    scales: list[Scale] = [named] if named else ["per_token", "per_1k", "per_1m"]
    label = ".".join(key_path)
    raw = str(value)

    def expiry(side: Side) -> str | None:
        price = target_prices[side]
        if price.valid_until is not None and price.valid_until < as_of:
            return (
                f"Note: the registry records this target price as valid until "
                f"{price.valid_until.isoformat()}"
                + (f" ({price.notes})" if price.notes else "")
                + "; refresh the registry evidence before relying on the comparison"
            )
        return None

    for side in sides:
        for scale in scales:
            per_million = number * _TO_PER_MILLION[scale]
            if per_million == target_prices[side].amount:
                return PricingCheck(
                    label, raw, "consistent", side, scale, per_million, target_prices[side].amount
                )
    for side in sides:
        for scale in scales:
            per_million = number * _TO_PER_MILLION[scale]
            if side in source_prices and per_million == source_prices[side].amount:
                return PricingCheck(
                    label,
                    raw,
                    "stale",
                    side,
                    scale,
                    per_million,
                    target_prices[side].amount,
                    source_prices[side].amount,
                    expiry_note=expiry(side),
                )
    best: tuple[Decimal, Side, Scale, Decimal] | None = None
    for side in sides:
        for scale in scales:
            per_million = number * _TO_PER_MILLION[scale]
            ratio = per_million / target_prices[side].amount
            distance = abs(ratio.ln()) if ratio > 0 else Decimal("Infinity")
            if best is None or distance < best[0]:
                best = (distance, side, scale, ratio)
    assert best is not None
    _, side, scale, ratio = best
    low, high = _CONTRADICTION_BAND
    if low <= ratio <= high:
        return PricingCheck(
            label,
            raw,
            "contradiction",
            side,
            scale,
            number * _TO_PER_MILLION[scale],
            target_prices[side].amount,
            source_prices[side].amount if side in source_prices else None,
            factor=ratio,
            expiry_note=expiry(side),
        )
    return PricingCheck(
        label,
        raw,
        "unit_unrecognized",
        side,
        None,
        None,
        target_prices[side].amount,
        expiry_note=expiry(side),
    )
