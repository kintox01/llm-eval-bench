# USD per 1M tokens (input, output), checked 2026-09-27
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "anthropic:claude-haiku-4-5": (1.00, 5.00),
    "anthropic:claude-sonnet-5": (2.00, 10.00),
    "anthropic:claude-opus-5": (5.00, 25.00),
    "openai:gpt-5.6-luna": (0.20, 1.20),
    "openai:gpt-5.6-terra": (2.00, 12.00),
    "typesafe:*": (0.042, 0.0),
}


def estimate_cost(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    prices = PRICES_PER_MTOK.get(model)
    if prices is None and model.startswith("typesafe:"):
        prices = PRICES_PER_MTOK["typesafe:*"]
    if prices is None or input_tokens is None:
        return None
    input_price, output_price = prices
    return (input_tokens * input_price + (output_tokens or 0) * output_price) / 1_000_000
