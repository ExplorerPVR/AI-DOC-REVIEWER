"""Model token pricing and cost calculation for supported LLM providers."""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Fallback baseline pricing for unlisted models ($ per 1M tokens)
DEFAULT_BASELINE_INPUT_PRICE: float = 0.10
DEFAULT_BASELINE_OUTPUT_PRICE: float = 0.40


@dataclass(frozen=True)
class ModelPrice:
    """Pricing per million tokens for input and output."""

    input_price_per_million: float
    output_price_per_million: float


# Official paid list prices per 1,000,000 tokens (USD)
# Free calls still carry a price on paper based on published provider rates.
MODEL_PRICING: dict[str, ModelPrice] = {
    # Google Gemini Models (source: ai.google.dev/pricing)
    "gemini-flash-latest": ModelPrice(
        input_price_per_million=0.10,
        output_price_per_million=0.40,
    ),
    "gemini-2.5-flash": ModelPrice(
        input_price_per_million=0.10,
        output_price_per_million=0.40,
    ),
    "gemini-2.0-flash": ModelPrice(
        input_price_per_million=0.10,
        output_price_per_million=0.40,
    ),
    "gemini-1.5-flash": ModelPrice(
        input_price_per_million=0.075,
        output_price_per_million=0.30,
    ),
    "gemini-1.5-pro": ModelPrice(
        input_price_per_million=1.25,
        output_price_per_million=5.00,
    ),
    # Groq Cloud Models (source: groq.com/pricing)
    "qwen/qwen3.8-27b": ModelPrice(
        input_price_per_million=0.20,
        output_price_per_million=0.20,
    ),
    "llama-3.3-70b-versatile": ModelPrice(
        input_price_per_million=0.59,
        output_price_per_million=0.79,
    ),
    "llama-3.1-8b-instant": ModelPrice(
        input_price_per_million=0.05,
        output_price_per_million=0.08,
    ),
    # Local / Open-Weights Models (e.g., Ollama)
    "llama3": ModelPrice(
        input_price_per_million=0.0,
        output_price_per_million=0.0,
    ),
    "mistral": ModelPrice(
        input_price_per_million=0.0,
        output_price_per_million=0.0,
    ),
}

# Default models per provider (can be overridden via environment variable)
DEFAULT_MODELS: dict[str, str] = {
    "gemini": os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
    "groq": os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
    "ollama": os.getenv("OLLAMA_MODEL", "llama3"),
}


def get_model_price(model_name: str) -> ModelPrice:
    """Retrieve pricing structure for a given model, defaulting to standard rate."""
    normalized_name = model_name.lower().strip()
    if normalized_name in MODEL_PRICING:
        return MODEL_PRICING[normalized_name]

    # Match version tags (e.g., 'gemini-1.5-flash-latest' -> 'gemini-1.5-flash')
    for key, price in MODEL_PRICING.items():
        if key in normalized_name:
            return price

    # Warn when pricing is unconfigured and fall back to baseline price
    logger.warning(
        "No pricing configured for model '%s'. Falling back to baseline price "
        "($%.2f in / $%.2f out per million tokens).",
        model_name,
        DEFAULT_BASELINE_INPUT_PRICE,
        DEFAULT_BASELINE_OUTPUT_PRICE,
    )
    return ModelPrice(
        input_price_per_million=DEFAULT_BASELINE_INPUT_PRICE,
        output_price_per_million=DEFAULT_BASELINE_OUTPUT_PRICE,
    )


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate the estimated USD cost of an LLM call at paid list price.

    Parameters
    ----------
    model_name : str
        The specific model identifier used for generation.
    input_tokens : int
        Number of prompt / input tokens.
    output_tokens : int
        Number of completion / generated tokens.

    Returns
    -------
    float
        Calculated cost in USD rounded to 8 decimal places.
    """
    price = get_model_price(model_name)
    input_cost = (max(0, input_tokens) / 1_000_000.0) * price.input_price_per_million
    output_cost = (
        max(0, output_tokens) / 1_000_000.0
    ) * price.output_price_per_million
    total_cost = input_cost + output_cost
    return round(total_cost, 8)
