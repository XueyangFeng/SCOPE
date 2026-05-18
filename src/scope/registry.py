from __future__ import annotations

from scope.webarena.adapter import WebArenaAdapter
from scope.webshop.adapter import WebShopAdapter
from scope.webvoyager.adapter import WebVoyagerAdapter


def get_adapter(name: str):
    key = name.strip().lower()
    adapters = {
        "webarena": WebArenaAdapter(),
        "webshop": WebShopAdapter(),
        "webvoyager": WebVoyagerAdapter(),
    }
    if key not in adapters:
        raise ValueError(f"Unknown dataset adapter: {name}")
    return adapters[key]
