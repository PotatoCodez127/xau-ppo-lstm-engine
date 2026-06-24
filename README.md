# XAU-PPO-LSTM-Engine (Project Aurelius)

Volatility-resilient Deep Reinforcement Learning (DRL) execution agent designed to optimize discrete action vectors within a simulated, multi-timeframe XAUUSD spot gold market.

---

## 🏛️ Architectural Overview

The engine treats spot market price execution as a continuous **Markov Decision Process (MDP)**. To combat non-stationarity and policy tracking divergence over highly volatile financial time-series, the system structures sequential observation spaces with an out-of-sample firewall layout.

### System Topology
* **Core Backbone:** Recurrent Proximal Policy Optimization (`RecurrentPPO`) utilizing an Actor-Critic multi-layer perceptron paired with an LSTM temporal feature extractor to capture non-linear, multi-timeframe macro dependencies.
* **Complete State Representation:** Observations ingest a rolling sequence window ($W=30$) containing synchronized market features combined with a real-time portfolio status tracking array (`position_state`, `bars_held`, `unrealized_pnl`) to achieve complete mathematical MDP conditions.
* **Feature Calibration Isolation:** Environment states are standardized online via running variance layers (`VecNormalize`). Scaling limits and rolling standard deviations are derived strictly from active processing partitions, blocking future lookahead data leakage.
```
xau-ppo-lstm-engine/
│
├── .github/workflows/       # Continuous Integration Pipeline
│   └── ci.yml               # Automated static analysis & clock-synced smoke tests
│
├── data/
│   ├── raw/                 # Raw tick/M1 data for XAUUSD and DXY indices
│   └── processed/           # Multi-timeframe engineered structural feature sets
│
├── environment/
│   └── trading_env.py       # Custom Gym/Gymnasium discrete execution environment
│
├── features/
│   ├── build_zones.py       # Wick-based support & resistance (S&R) calculators
│   ├── build_pivots.py      # Daily equilibrium and standard pivot point map
│   └── main_processor.py    # Multi-timeframe time-series alignment pipeline
│
├── models/
│   ├── lstm_ppo.py          # Session-based recurrent reinforcement training loop
│   ├── test_integration.py  # Mini-batch shape-validated CI smoke test
│   └── test_agent.py        # Out-of-sample forward backtester & evaluation engine
│
├── Dockerfile               # Multi-stage, CUDA-calibrated production deployment image
├── pyproject.toml           # Unified packaging configuration & deterministic linter truth
└── README.md                # System architectural storefront
```
---

## 🛠️ Infrastructure & Verification Guardrails

### 1. Deterministic Packaging & Tooling
The environment enforces modern Python packaging specifications (PEP 517 / PEP 621) via `pyproject.toml` with `setuptools.build_meta`. Code quality and formatting checks are governed by `Ruff` to handle quantitative naming constraints (e.g., whitelisting uppercase mathematical matrix terms like $X$ and $y$).

### 2. Multi-Stage Containerization Isolation
To preserve deployment mobility across local hardware and cloud scale clusters (e.g., Google Colab), the engine isolates wheel compilation inside a multi-stage `Dockerfile`:
* **Stage 1 (`builder`):** Compiles native C-extensions and heavy matrix wheels into an isolated cache layer.
* **Stage 2 (`runner`):** Copies compiled runtime wheels forward into a minimal runtime base, pinning the operating system clock to `UTC` to enforce temporal agreement with historical tick streams.

### 3. Automated CI Validation Gates
Every code push triggers a sequence inside GitHub Actions (`ci.yml`):
* Enforces strict compliance checks via `Ruff`.
* Synchronizes machine runner clocks explicitly to `UTC`.
* Installs dependencies deterministically and executes a truncated sequence (`test_integration.py`) to confirm shape stability and verify environment reward registration.

---

## 🚀 Quickstart & Verification Pipeline

### Local Native Installation
Set up an editable local installation mapping your flat-layout execution environment:
```bash
pip install --upgrade pip
pip install -e .
```
### Executing Local Validation Smoke Tests
Validate that tensor dimensions match environment observation bounds prior to full-scale training run cycles:
```bash
python models/test_integration.py
```

### Local Container Deployment
Construct and run the production-ready, CUDA-calibrated environment layer locally or remotely:
```bash
docker build -t xau-ppo-lstm-engine .
docker run --rm --gpus all xau-ppo-lstm-engine
```
---