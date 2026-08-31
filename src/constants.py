# SPDX-License-Identifier: MIT
"""Game constants for Kaggriculture simulation and agent policy."""

BOARD_SIZE = 10
HALF_BOARD = BOARD_SIZE // 2
TURNS_PER_DAY = 24
TOTAL_DAYS = 30
TOTAL_TURNS = TURNS_PER_DAY * TOTAL_DAYS  # 720
SHED_CAPACITY = 100
MAX_MARKET_ORDERS_PER_TURN = 10

# Inner-corner tiles around the shed (NW, NE, SW, SE)
SHED_TILES = [
    (HALF_BOARD - 1, HALF_BOARD - 1),
    (HALF_BOARD, HALF_BOARD - 1),
    (HALF_BOARD - 1, HALF_BOARD),
    (HALF_BOARD, HALF_BOARD),
]

CROPS = {
    "WHEAT": {"seed": 10, "base_price": 25, "first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT": {"seed": 20, "base_price": 35, "first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO": {"seed": 50, "base_price": 60, "first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "base_price": 120, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON": {"seed": 80, "base_price": 250, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP", "first_yield_day": 4, "interval": 1, "max_yield": 4, "product": "EGG"},
    "COW": {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_yield": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_yield": 6, "product": "WOOL"},
}

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
LAND_PRICES = [1000, 2000, 4000]
