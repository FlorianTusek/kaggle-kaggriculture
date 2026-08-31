# Kaggriculture Competition Roadmap

> **Competition**: [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)
> **Deadline**: Final submission September 30, 2026 · Final leaderboard October 1–15, 2026
> **Repo**: `D:\DunderCode\kaggle-kaggriculture`

---

## 1. Problem Statement

Kaggriculture is a **simulation-based code competition** where you build an autonomous AI agent that manages a virtual farm. The goal is to **maximize total profit (bank balance)** by the end of a 30-day season (720 discrete turns, 24 turns/day).

This is a **two-player, turn-based** simulation run via `kaggle_environments`. Agents play head-to-head episodes against other submitted bots, and ranking is determined by an **Elo-style ladder**.

### Key Constraints
- **5 submissions per day**
- **No internet** in submission notebooks
- Submission is a **Python notebook** containing an `agent(observation, configuration)` function
- Each turn allows multiple actions; max **10 market orders per turn**

---

## 2. Core Game Mechanics

### 2.1 Farm Operations
| System | Details |
|--------|---------|
| **Crops** | Plant → water → (optional fertilize) → harvest. Missing a watering cycle causes crop damage/loss. |
| **Livestock** | Buy animals (cows, sheep, chickens) → feed consistently (wheat) → collect produce (milk, wool, eggs). Missed feedings reduce output or kill animals. |
| **Land** | Start with a base plot. Purchase adjacent land quadrants to expand capacity. |
| **Labor** | Hire farm hands to scale operations. Cost follows a **Fibonacci-based curve** — early hires are cheap, later ones expensive. |

### 2.2 Market & Economy
- **Dynamic pricing**: Prices react to your sales volume and simulated town demand.
- **Price decay**: Oversupplying a good tanks its price — timing and diversification matter.
- **Max 10 market orders per turn**: Forces batching and prioritization.

### 2.3 The Penalty Cascade
The environment is **punishingly unforgiving**:
- A single missed watering or feeding can trigger a cascade of losses.
- Bankruptcy is a real outcome — poor resource management compounds rapidly.
- This makes **reliability** more important than cleverness in early iterations.

---

## 3. Evaluation Metric

| Metric | Description |
|--------|-------------|
| **Primary** | Total bank balance at end of 720 turns |
| **Ranking** | Elo-style ladder from head-to-head episodes |
| **Games** | Agents play many episodes against opponents of similar skill |

---

## 4. Strategic Approaches (Ordered by Priority)

### Phase 1: Deterministic Heuristic Baseline (Week 1)
**Why first**: Community consensus is that robust rule-based agents **outperform naive RL** in this competition due to the long horizon (720 turns), delayed rewards, and punishing mechanics.

**Tasks**:
1. **Study the environment API**: Understand `observation` and `configuration` schemas. Identify all available actions.
2. **Build a "safe loop" agent**: Hard-code reliable sequences for:
   - Planting wheat early (cheap, feeds animals)
   - Watering every crop every cycle (never miss)
   - Feeding all animals every cycle (never miss)
   - Harvesting at optimal time
   - Selling goods when prices are favorable
3. **Implement resource management**:
   - Track cash flow and avoid overspending on land/labor
   - Fibonacci-aware hiring strategy (hire early when cheap)
4. **Submit and benchmark** against the leaderboard

### Phase 2: Market Optimization (Week 2)
**Tasks**:
1. **Implement price tracking**: Monitor market prices across turns to detect trends.
2. **Build a sell-timing heuristic**: Avoid dumping all goods at once (price decay). Spread sales across turns.
3. **Demand-responsive planting**: Adjust crop mix based on observed price signals.
4. **Diversification strategy**: Balance crop types and animal products to avoid single-commodity risk.

### Phase 3: Behavioral Cloning & Imitation Learning (Week 2–3)
**Why**: Public replay datasets exist. Top agents' strategies can be extracted.

**Tasks**:
1. **Download episode replays** from Kaggle datasets section.
2. **Parse replay data** into (state, action) pairs.
3. **Train a behavioral cloning model** (supervised learning on expert actions).
4. **Use as a warm-start** for the heuristic agent — let it handle edge cases the rules miss.

### Phase 4: Reinforcement Learning (Week 3–4, if time permits)
**Why last**: RL struggles with the 720-turn horizon and delayed rewards. Only pursue if the heuristic baseline plateaus.

**Tasks**:
1. **Define reward shaping**: Don't rely on end-of-game profit alone. Add intermediate rewards for:
   - Successful harvests
   - Animal produce collection
   - Cash flow milestones
2. **Train with self-play** or against the heuristic baseline.
3. **Combine with behavioral cloning** (pre-train policy, then fine-tune with RL).

### Phase 5: Advanced Search & Meta-Game (Week 4+)
**Why**: Once PPO and Behavioral Cloning plateau, the highest echelon of Kaggle agents rely on search algorithms (like MCTS) augmented by neural networks, and opponent modeling.

**Tasks**:
1. **Monte Carlo Tree Search (MCTS) Hybrid**:
   - Implement MCTS for the agent's decision-making process.
   - Use the Phase 4 PPO policy network as a prior to prune the MCTS search tree, avoiding exhaustive search.
   - Use a trained value network (instead of full random rollouts) to evaluate leaf nodes quickly.
2. **Opponent Modeling & Meta-Agents**:
   - Identify opponent archetypes (e.g., aggressive expander, market dumper).
   - **Wheat Denial Tactics**: Be aware of opponents buying out early wheat (14-19 units) to starve your livestock. Implement the **"Feed5-first" strategy** (buying 5 wheat at step 0) as a meta-counter to secure early feed.
   - Train a meta-layer that monitors the opponent's strategy during the game and switches between sub-policies to counter them.
3. **Strategy Ensembling**:
   - Combine multiple policies (e.g., heuristic safety layer + MCTS for market + PPO for expansion) rather than relying on a single monolithic model.
   - Average MCTS visit counts across different models to make robust decisions.
4. **Data Augmentation & Local Validation**:
   - Build a highly stable local cross-validation environment (Elo-style simulator) to test against a battery of diverse agents.
   - Augment training data by swapping roles or mirroring farm states to prevent overfitting.

---

## 5. Feature Engineering Ideas

### State Representation
- **Farm state**: Grid occupancy, crop growth stages, water status, animal health/feed status
- **Economic state**: Current cash, inventory of goods, market price history (rolling window)
- **Labor state**: Number of farm hands, cost of next hire
- **Turn context**: Current turn number, day number, time-of-day (morning/afternoon/night may affect mechanics)
- **Opponent signals**: If observable — opponent's farm size, market activity

### Derived Features
- **Cash flow rate**: ΔBalance over last N turns
- **Price momentum**: Moving average of price changes per commodity
- **ROI per crop/animal**: Expected revenue minus input costs (seeds, water, feed, labor)
- **Capacity utilization**: % of land in use, % of labor capacity utilized
- **Risk score**: How close to bankruptcy (low cash + high upcoming costs)

---

## 6. Modeling Strategies

| Approach | Pros | Cons | Priority |
|----------|------|------|----------|
| **Rule-based heuristic** | Reliable, interpretable, fast to iterate | Hard to optimize edge cases | ★★★ High |
| **Behavioral cloning** | Learns from top players, good baseline | Needs quality replay data, brittle | ★★ Medium |
| **Reinforcement Learning** | Can discover novel strategies | Slow to train, hard to debug, long horizon | ★ Low |
| **Hybrid (rules + ML)** | Best of both — rules for safety, ML for optimization | Complex to integrate | ★★ Medium |

### Recommended Architecture
```
┌─────────────────────────────────────┐
│           Agent Controller          │
├──────────┬──────────┬───────────────┤
│ Safety   │ Strategy │ Market        │
│ Layer    │ Planner  │ Optimizer     │
│          │          │               │
│ Never    │ Decides  │ Tracks prices │
│ miss     │ what to  │ and times     │
│ water/   │ plant,   │ sales for     │
│ feed     │ buy,     │ max profit    │
│ cycles   │ hire     │               │
└──────────┴──────────┴───────────────┘
```

**Safety Layer** runs first every turn (water all crops, feed all animals), then **Strategy Planner** decides investments (new crops, animals, land, labor), then **Market Optimizer** handles sell orders.

---

## 7. Technical Implementation Plan

### Repository Structure (extend existing template)
```
kaggle-kaggriculture/
├── src/
│   ├── agent.py          # Main agent(obs, config) entry point
│   ├── safety.py         # Water/feed/harvest safety loops
│   ├── strategy.py       # Planting, buying, hiring decisions
│   ├── market.py         # Price tracking and sell optimization
│   ├── state.py          # State parsing and feature extraction
│   ├── constants.py      # Game constants (Fibonacci costs, crop timings, etc.)
│   ├── replay_parser.py  # Parse episode replays for behavioral cloning
│   └── models.py         # ML models (behavioral cloning, RL policy)
├── notebooks/
│   ├── 01_eda.ipynb           # Explore environment API and game mechanics
│   ├── 02_baseline.ipynb      # Heuristic baseline development
│   ├── 03_replay_analysis.ipynb  # Analyze top player replays
│   └── 04_submission.ipynb    # Clean submission notebook
├── data/
│   └── replays/               # Downloaded episode replay data
└── tests/
    ├── test_safety.py         # Verify safety loops never miss
    └── test_strategy.py       # Verify investment logic
```

### Key Dependencies
- `kaggle_environments` — local simulation and testing
- `numpy`, `pandas` — data handling
- `torch` or `tensorflow` — if doing behavioral cloning / RL
- `pytest` — testing

---

## 8. Timeline

| Week | Dates | Focus | Deliverable |
|------|-------|-------|-------------|
| **1** | Sep 1–7 | Environment study + heuristic baseline | Working agent that survives 720 turns, first submission |
| **2** | Sep 8–14 | Market optimization + replay analysis | Improved agent with price-aware selling |
| **3** | Sep 15–21 | Behavioral cloning + hybrid approach | ML-augmented agent |
| **4** | Sep 22–30 | Polish, edge cases, final submissions | Best possible agent before deadline |

### Milestones
- [ ] **M1**: Environment API fully understood, constants documented
- [ ] **M2**: Heuristic agent survives full game without bankruptcy
- [ ] **M3**: First leaderboard submission + Elo baseline established
- [ ] **M4**: Market optimizer reduces price decay losses by ≥ 20%
- [ ] **M5**: Behavioral cloning model trained on ≥ 1000 replays
- [ ] **M6**: Final agent submitted before Sep 30 deadline

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Environment API changes mid-competition | High | Pin `kaggle_environments` version; monitor forums |
| Heuristic baseline plateaus early | Medium | Move to behavioral cloning sooner |
| Replay data quality is poor | Medium | Focus on top-100 player replays only |
| RL training too slow / unstable | Low | RL is Phase 4 — skip if no time |
| Bankruptcy cascades in edge cases | High | Safety layer runs first every turn, no exceptions |

---

## 10. Open Questions (for team discussion)

1. **Do we have Kaggle API credentials set up?** Needed to download competition data and replays.
2. **GPU access**: Do we have GPU compute for behavioral cloning / RL training?
3. **Submission workflow**: Who manages the 5/day submission budget?
4. **Competition forum monitoring**: Should Creed track discussion threads for meta-strategy shifts?
