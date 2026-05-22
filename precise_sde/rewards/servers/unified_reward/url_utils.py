"""Helpers for normalizing UnifiedReward API endpoints."""


def normalize_unifiedreward_base_url(base_url: str) -> str:
    """Accept either a server root URL or a `/v1` API URL and return the server root."""
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized
