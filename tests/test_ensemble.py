# SPDX-License-Identifier: MIT
"""Tests for StrategyEnsemble meta-controller (Task-27)."""

import pytest
from unittest.mock import MagicMock, patch
from src.ensemble import (
    StrategyEnsemble,
    _get_game_phase,
    _weighted_vote,
    PHASE_WEIGHTS,
    VALID_WORKER_ACTIONS,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal observation builder
# ---------------------------------------------------------------------------

def _make_obs(day=5, hour=6, step=None, money=5000, farmer=(4, 4),
              tiles=None, shed=None, seeds=None, prices=None,
              unlocked_quadrants=None, market_inventory=None):
    """Build a minimal Kaggriculture observation dict."""
    if step is None:
        step = day * 24 + hour
    if tiles is None:
        tiles = [[None] * 10 for _ in range(10)]
        # Mark shed center tiles
        tiles[4][4] = {"kind": "SHED"}
        tiles[4][5] = {"kind": "SHED"}
        tiles[5][4] = {"kind": "SHED"}
        tiles[5][5] = {"kind": "SHED"}
    if shed is None:
        shed = {}
    if seeds is None:
        seeds = {"CARROT": 6, "WHEAT": 6}
    if prices is None:
        prices = {"CARROT": 35, "TOMATO": 60, "WHEAT": 25}
    if unlocked_quadrants is None:
        unlocked_quadrants = ["NW"]

    return {
        "step": step,
        "day": day,
        "hour": hour,
        "player": 0,
        "farms": [
            {
                "money": money,
                "farmer": list(farmer),
                "hands": [],
                "tiles": tiles,
                "unlocked_quadrants": unlocked_quadrants,
                "hires_today": 0,
            },
            {
                "money": 3000,
                "farmer": [5, 5],
                "hands": [],
                "tiles": tiles,
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
        ],
        "private": {"shed": shed, "seeds": seeds},
        "market": {
            "prices": prices,
            "inventory": market_inventory or {"WHEAT": 10000},
        },
        "town": {"unlocked_shops": []},
    }


# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------

class TestGamePhase:
    def test_early_phase(self):
        assert _get_game_phase(0) == "early"
        assert _get_game_phase(8) == "early"

    def test_mid_phase(self):
        assert _get_game_phase(9) == "mid"
        assert _get_game_phase(20) == "mid"

    def test_late_phase(self):
        assert _get_game_phase(21) == "late"
        assert _get_game_phase(29) == "late"


# ---------------------------------------------------------------------------
# Weighted voting
# ---------------------------------------------------------------------------

class TestWeightedVote:
    def test_single_policy_dominant(self):
        candidates = {"ppo": {"NORTH": 0.9, "SOUTH": 0.1}}
        weights = {"ppo": 1.0}
        best, scores = _weighted_vote(candidates, weights)
        assert best == "NORTH"
        assert scores["NORTH"] > scores["SOUTH"]

    def test_multi_policy_tiebreak(self):
        candidates = {
            "bc": {"NORTH": 0.8, "SOUTH": 0.2},
            "ppo": {"SOUTH": 0.7, "NORTH": 0.3},
        }
        weights = {"bc": 0.6, "ppo": 0.4}
        best, scores = _weighted_vote(candidates, weights)
        # bc votes NORTH (0.6*0.8=0.48), ppo votes SOUTH (0.4*0.7=0.28)
        # NORTH total = 0.48 + 0.4*0.3 = 0.60
        # SOUTH total = 0.6*0.2 + 0.28 = 0.40
        assert best == "NORTH"

    def test_empty_candidates_return_pass(self):
        best, scores = _weighted_vote({}, {"bc": 0.5})
        assert best == "PASS"

    def test_invalid_actions_filtered(self):
        candidates = {"ppo": {"INVALID_ACT": 1.0, "NORTH": 0.5}}
        weights = {"ppo": 1.0}
        best, scores = _weighted_vote(candidates, weights)
        assert best == "NORTH"
        assert "INVALID_ACT" not in scores


# ---------------------------------------------------------------------------
# Ensemble initialization & lazy loading
# ---------------------------------------------------------------------------

class TestEnsembleInit:
    def test_default_construction(self):
        ensemble = StrategyEnsemble()
        assert ensemble.safety_layer is not None
        assert ensemble.strategy_planner is not None
        assert ensemble.market_optimizer is not None
        assert not ensemble._policies_initialized

    def test_policies_lazy_init(self):
        ensemble = StrategyEnsemble({"use_mcts": False, "use_ml_policy": False})
        assert not ensemble._policies_initialized
        ensemble._init_policies()
        assert ensemble._policies_initialized

    def test_double_init_is_idempotent(self):
        ensemble = StrategyEnsemble({"use_mcts": False})
        ensemble._init_policies()
        # Second call should be a no-op
        ensemble._init_policies()
        assert ensemble._policies_initialized


# ---------------------------------------------------------------------------
# Safety override
# ---------------------------------------------------------------------------

class TestSafetyOverride:
    def test_safety_jobs_override_ensemble(self):
        """When safety jobs exist, ensemble voting is skipped (safety has veto)."""
        obs = _make_obs(day=5, hour=6)
        # Plant an unwatered crop (emergency) at (0, 0)
        obs["farms"][0]["tiles"][0][0] = {
            "kind": "PLANT",
            "crop": "CARROT",
            "watered_today": False,
            "consecutive_unwatered": 2,
            "planted_day": 3,
            "yield_units": 0,
        }

        ensemble = StrategyEnsemble({"use_mcts": False, "use_ml_policy": False})
        # Mock out ML policies so ensemble would recommend NORTH if called
        ensemble._policies_initialized = True

        result = ensemble.act(obs)
        # The farmer should be working toward the emergency watering, not doing an ensemble action
        farmer_op = result["farmer"]
        assert farmer_op != ["PASS"], "Safety should assign a job, not PASS"

    def test_no_safety_allows_ensemble(self):
        """When no safety jobs, ensemble voting is exercised."""
        obs = _make_obs(day=5, hour=6)

        ensemble = StrategyEnsemble({"use_mcts": False})
        ensemble._policies_initialized = True
        # Inject a mock BC policy
        mock_bc = MagicMock()
        mock_bc.is_loaded = True
        mock_bc.predict_farmer_action_probs.return_value = {"NORTH": 0.9, "SOUTH": 0.1}
        ensemble._bc_policy = mock_bc

        result = ensemble.act(obs)
        # Should have produced a valid action dict
        assert "farmer" in result
        assert "hands" in result
        assert "market" in result


# ---------------------------------------------------------------------------
# Act pipeline produces valid output
# ---------------------------------------------------------------------------

class TestActPipeline:
    def test_act_returns_valid_structure(self):
        obs = _make_obs()
        ensemble = StrategyEnsemble({"use_mcts": False})
        ensemble._policies_initialized = True  # skip actual model loading

        result = ensemble.act(obs)
        assert isinstance(result, dict)
        assert "farmer" in result
        assert "hands" in result
        assert "market" in result
        assert isinstance(result["farmer"], list)
        assert isinstance(result["hands"], list)
        assert isinstance(result["market"], list)

    def test_act_farmer_action_is_valid(self):
        obs = _make_obs()
        ensemble = StrategyEnsemble({"use_mcts": False})
        ensemble._policies_initialized = True

        result = ensemble.act(obs)
        farmer_action = result["farmer"][0] if result["farmer"] else "PASS"
        assert farmer_action in VALID_WORKER_ACTIONS or farmer_action in ("PASS",)

    def test_market_orders_capped_at_10(self):
        obs = _make_obs(shed={"CARROT": 200, "TOMATO": 100, "WHEAT": 200, "MELON": 50, "STRAWBERRY": 50})
        ensemble = StrategyEnsemble({"use_mcts": False})
        ensemble._policies_initialized = True

        result = ensemble.act(obs)
        assert len(result["market"]) <= 10


# ---------------------------------------------------------------------------
# Advise debugging output
# ---------------------------------------------------------------------------

class TestAdvise:
    def test_advise_returns_diagnostic_keys(self):
        obs = _make_obs(day=15)
        ensemble = StrategyEnsemble({"use_mcts": False})
        ensemble._policies_initialized = True

        advice = ensemble.advise(obs)
        assert "game_phase" in advice
        assert advice["game_phase"] == "mid"
        assert "phase_weights" in advice
        assert "ensemble_scores" in advice
        assert "recommended_action" in advice
        assert "opponent_archetype" in advice
        assert "policies_loaded" in advice

    def test_advise_phase_weights_match_day(self):
        for day, expected_phase in [(3, "early"), (15, "mid"), (25, "late")]:
            obs = _make_obs(day=day)
            ensemble = StrategyEnsemble({"use_mcts": False})
            ensemble._policies_initialized = True

            advice = ensemble.advise(obs)
            assert advice["game_phase"] == expected_phase
            assert advice["phase_weights"] == PHASE_WEIGHTS[expected_phase]


# ---------------------------------------------------------------------------
# Opponent integration via market optimizer
# ---------------------------------------------------------------------------

class TestOpponentIntegration:
    def test_feed5_first_on_turn_zero(self):
        """Opponent tracker should inject Feed5-first order at step 0."""
        obs = _make_obs(day=0, hour=0, step=0, money=5000,
                        seeds={"CARROT": 6}, shed={})
        ensemble = StrategyEnsemble({"use_mcts": False})
        ensemble._policies_initialized = True

        result = ensemble.act(obs)
        market = result["market"]
        # Should contain a BUY_SEED WHEAT order from Feed5-first counter
        feed5 = [o for o in market if o[0] == "BUY_SEED" and o[1] == "WHEAT"]
        assert len(feed5) >= 1, f"Feed5-first order missing; market = {market}"
