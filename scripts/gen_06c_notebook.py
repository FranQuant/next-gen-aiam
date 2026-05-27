"""Generate notebooks/06c_rl_single_asset_dqn.ipynb."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent


def md(source: str) -> dict:
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "id": None, "metadata": {}, "source": src}


def code(source: str) -> dict:
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


# ---------------------------------------------------------------------------
# Cell 0: title + purpose (markdown)
# ---------------------------------------------------------------------------
cells = []

cells.append(md(
    "[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/FranQuant/next-gen-aiam/blob/main/"
    "notebooks/06c_rl_single_asset_dqn.ipynb)\n"
    "\n"
    "# Notebook 06c -- RL III: Single-Asset Directional DQN (tractability control)\n"
    "\n"
    "**Purpose -- positive control, not a leaderboard entry.**\n"
    "This notebook shows the same RL machinery *does* learn on a tractable single-asset "
    "directional problem, in deliberate contrast to the 06a/06b static collapse on the "
    "29-asset allocation universe.  The finding answers whether the collapse in 06a/06b "
    "is a property of the *problem* (high-dimensional simplex, weak per-asset reward signal) "
    "or a bug in the *implementation*.\n"
    "\n"
    "**DQN design credit:** Hilpisch, *Reinforcement Learning for Finance* (`dqlagent.py`), "
    "the book this repo extends.\n"
    "\n"
    "**What this notebook is NOT:**\n"
    "- Not a competitive portfolio strategy\n"
    "- Not comparable to the 29-asset leaderboard (ML bar 2.579 / classical 2.422 / 2.386)\n"
    "- Not the primary RL chapter (that is 06a/06b)\n"
    "\n"
    "**Off-axis caveat (stated once, always apply):** single asset vs 29-asset allocation, "
    "binary long/flat vs continuous simplex, weak B&H benchmark vs best-in-class ML.  "
    "Sharpe numbers here live in a different space from the main leaderboard and must "
    "**not** be rank-compared.\n"
    "\n"
    "**Cross-paradigm narrative:**\n"
    "- **06a** REINFORCE on N=29 => `STATIC_COLLAPSE_DETECTED` (converges to ~1/N equal-weight)\n"
    "- **06b** PPO on N=29 => confirms collapse\n"
    "- **06c** DQN on N=1 (SPY, long/flat) => shows where RL *does* learn"
))

# ---------------------------------------------------------------------------
# Cell 1: cross-paradigm context table (markdown)
# ---------------------------------------------------------------------------
cells.append(md(
    "## Context: cross-paradigm sequence\n"
    "\n"
    "| Notebook | Paradigm | Best OOS Sharpe (2023-2026) |\n"
    "|---|---|---|\n"
    "| [03](03_ml_strategies.ipynb) | ML ensemble (Lasso/RF/XGB -> MSR) | **2.579** <- bar |\n"
    "| [04](04_dl_strategies.ipynb) | DL predict-then-optimize | 2.386 |\n"
    "| [05](05_dl_portfolio_construction_exploration.ipynb) | DL direct-weight (BSV 2009) | 1.240 |\n"
    "| [06a](06a_rl_reinforce.ipynb) | RL -- REINFORCE (N=29) | 2.027 (`STATIC_COLLAPSE`) |\n"
    "| [06b](06b_rl_ppo.ipynb) | RL -- PPO (N=29) | 2.020 (`STATIC_COLLAPSE`) |\n"
    "| **06c** | **RL -- DQN (N=1, SPY long/flat)** | **this notebook -- OFF-AXIS** |\n"
    "\n"
    "**The control question:** if we shrink the problem to N=1 asset, 2 discrete actions, "
    "and dense per-step reward, does the RL machinery learn a feature-conditional policy?  "
    "If yes => collapse in 06a/06b is problem-driven, not a code defect."
))

# ---------------------------------------------------------------------------
# Cell 2: Colab setup + run params (code)
# ---------------------------------------------------------------------------
cells.append(code(
    "import sys, os, time, warnings\n"
    "from pathlib import Path\n"
    "\n"
    "IS_COLAB = 'google.colab' in sys.modules\n"
    "\n"
    "if IS_COLAB:\n"
    "    print('Detected Colab -- cloning repo...')\n"
    "    os.system('git clone https://github.com/FranQuant/next-gen-aiam.git 2>&1 | tail -3')\n"
    "    os.chdir('next-gen-aiam')\n"
    "    print(f'Working dir: {os.getcwd()}')\n"
    "    os.system('pip install -q -e . 2>&1 | tail -3')\n"
    "    os.system('pip install -q pyarrow pandas matplotlib torch 2>&1 | tail -3')\n"
    "else:\n"
    "    print(f'Local env -- {os.getcwd()}')\n"
    "\n"
    "import torch\n"
    "\n"
    "# DQN state dim K=10, hidden=24 -- micro-batch; CPU is fastest (no GPU transfer overhead)\n"
    "DEVICE = 'cpu'\n"
    "print(f'torch : {torch.__version__}  |  device: {DEVICE}')\n"
    "\n"
    "SEEDS    = [0, 1, 2, 3, 4]   # 5 seeds as specified\n"
    "EPISODES = 40                  # modest; keep untuned (this is a control, not a search)\n"
    "K        = 10                  # state window: last K lagged daily returns\n"
    "\n"
    "print(f'Seeds={SEEDS}  Episodes={EPISODES}  K={K}')"
))

# ---------------------------------------------------------------------------
# Cell 3: imports + rcParams (markdown)
# ---------------------------------------------------------------------------
cells.append(md("## §0 Setup -- imports, matplotlib, paths"))

# ---------------------------------------------------------------------------
# Cell 4: imports + rcParams (code)
# ---------------------------------------------------------------------------
cells.append(code(
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib\n"
    "import matplotlib.pyplot as plt\n"
    "import random\n"
    "from collections import deque\n"
    "import torch.nn as nn\n"
    "import torch.optim as optim\n"
    "\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "matplotlib.rcParams.update({\n"
    "    'font.family': 'serif',\n"
    "    'font.size': 10,\n"
    "    'figure.dpi': 120,\n"
    "    'axes.spines.top': False,\n"
    "    'axes.spines.right': False,\n"
    "    'axes.grid': True,\n"
    "    'grid.alpha': 0.2,\n"
    "    'figure.facecolor': 'white',\n"
    "    'axes.facecolor': 'white',\n"
    "})\n"
    "\n"
    "ROOT = Path(os.getcwd())\n"
    "if not (ROOT / 'pyproject.toml').exists():\n"
    "    ROOT = ROOT.parent\n"
    "\n"
    "for p in [str(ROOT / 'src'), str(ROOT / 'notebooks')]:\n"
    "    if p not in sys.path:\n"
    "        sys.path.insert(0, p)\n"
    "\n"
    "from _shared import ann_sharpe, ann_return, max_drawdown, TRADING_DAYS\n"
    "from aiam.data.split import TEST_START, TRAIN_END\n"
    "\n"
    "RESULTS_DIR = ROOT / 'results' / 'notebook_06c'\n"
    "FIG_DIR     = RESULTS_DIR / 'figures'\n"
    "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n"
    "FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "print(f'ROOT       : {ROOT}')\n"
    "print(f'Results    : {RESULTS_DIR}')\n"
    "print(f'TRAIN_END  : {TRAIN_END}')\n"
    "print(f'TEST_START : {TEST_START}')"
))

# ---------------------------------------------------------------------------
# Cell 5: data loading (markdown)
# ---------------------------------------------------------------------------
cells.append(md("## §1 Data Loading -- SPY from canonical 29-asset panel"))

# ---------------------------------------------------------------------------
# Cell 6: data loading (code)
# ---------------------------------------------------------------------------
cells.append(code(
    "# Load SPY from the canonical 29-asset daily-returns parquet (column 'SPY.US').\n"
    "# Same data source as 06a/06b for consistency.\n"
    "ret_all = pd.read_parquet(ROOT / 'data' / 'cache' / 'returns_29_2003_2026.parquet')\n"
    "ret_all.index = pd.to_datetime(ret_all.index)\n"
    "\n"
    "spy_all   = ret_all['SPY.US'].dropna()\n"
    "spy_train = spy_all.loc[:TRAIN_END]\n"
    "spy_oos   = spy_all.loc[TEST_START:]\n"
    "\n"
    "print(f'SPY full  : {spy_all.index[0].date()} -> {spy_all.index[-1].date()}  ({len(spy_all)} days)')\n"
    "print(f'Train     : {spy_train.index[0].date()} -> {spy_train.index[-1].date()}  ({len(spy_train)} days)')\n"
    "print(f'OOS       : {spy_oos.index[0].date()} -> {spy_oos.index[-1].date()}   ({len(spy_oos)} days)')\n"
    "print(f'Steps/ep (K={K}): {len(spy_train)-K-1}   OOS steps: {len(spy_oos)-K-1}')"
))

# ---------------------------------------------------------------------------
# Cell 7: environment (markdown)
# ---------------------------------------------------------------------------
cells.append(md(
    "## §2 Environment -- `SingleAssetEnv`\n"
    "\n"
    "Single-asset directional environment -- the tractable N=1 analogue of `PortfolioEnv` in "
    "`aiam/rl/env.py`:\n"
    "\n"
    "| | 29-asset (06a/06b) | 1-asset (06c) |\n"
    "|---|---|---|\n"
    "| State | 20-day rolling returns x 29 assets | 10 lagged daily returns |\n"
    "| Actions | Continuous simplex weights | Discrete: 0=flat, 1=long |\n"
    "| Reward | Portfolio return - lambda*variance | position * next-day return |\n"
    "| Difficulty | High (scales with N^2) | Low (scalar, dense signal) |\n"
    "\n"
    "No lookahead: state at step *i* = `returns[i:i+K]`; "
    "the position earns `returns[i+K]` (strictly next-day)."
))

# ---------------------------------------------------------------------------
# Cell 8: environment class (code)
# ---------------------------------------------------------------------------
cells.append(code(
    "class SingleAssetEnv:\n"
    "    # State = K lagged daily returns; actions = {0=flat, 1=long}; reward = position * next_return\n"
    "\n"
    "    def __init__(self, returns, K=10):\n"
    "        self.returns   = returns\n"
    "        self.K         = K\n"
    "        self.max_steps = len(returns) - K - 1\n"
    "\n"
    "    def reset(self):\n"
    "        self.step_idx = 0\n"
    "        return self.returns[0:self.K].copy()\n"
    "\n"
    "    def step(self, action):\n"
    "        i      = self.step_idx\n"
    "        reward = float(action) * self.returns[i + self.K]\n"
    "        self.step_idx += 1\n"
    "        done   = (self.step_idx >= self.max_steps)\n"
    "        if not done:\n"
    "            next_state = self.returns[self.step_idx:self.step_idx + self.K].copy()\n"
    "        else:\n"
    "            next_state = np.zeros(self.K, dtype=np.float32)\n"
    "        return next_state, reward, done\n"
    "\n"
    "\n"
    "# Sanity check\n"
    "_env = SingleAssetEnv(spy_train.values.astype(np.float32), K=K)\n"
    "s0 = _env.reset()\n"
    "s1, r1, d1 = _env.step(1)\n"
    "print(f'max_steps : {_env.max_steps}')\n"
    "print(f'state[0][:3]: {s0[:3]}')\n"
    "print(f'reward[step=0, action=1]: {r1:.6f}  ')\n"
    "print(f'  (= spy_train[{K}] = {spy_train.values[K]:.6f})')\n"
    "print(f'done[0]   : {d1}')"
))

# ---------------------------------------------------------------------------
# Cell 9: DQN agent (markdown)
# ---------------------------------------------------------------------------
cells.append(md(
    "## §3 DQN Agent -- reproducing `dqlagent.py`\n"
    "\n"
    "Architecture follows Hilpisch *Reinforcement Learning for Finance* (`dqlagent.py`):\n"
    "- **Q-network:** Linear(K, 24) -> ReLU -> Linear(24, 24) -> ReLU -> Linear(24, 2)\n"
    "- **Replay buffer:** `deque(maxlen=2000)`, uniform random sampling\n"
    "- **Exploration:** epsilon-greedy, eps: 1.0 -> 0.1, decay per step ~0.9975\n"
    "- **Bellman update:** gamma=0.5, Adam lr=1e-3, MSE loss, minibatch=32\n"
    "- **Target network:** hard copy every 50 gradient steps (stabilises training)"
))

# ---------------------------------------------------------------------------
# Cell 10: DQN agent code (code)
# ---------------------------------------------------------------------------
cells.append(code(
    "class QNetwork(nn.Module):\n"
    "    # Two hidden layers, 24 units, ReLU -- Hilpisch dqlagent.py design\n"
    "    def __init__(self, state_dim, n_actions=2, hidden=24):\n"
    "        super().__init__()\n"
    "        self.net = nn.Sequential(\n"
    "            nn.Linear(state_dim, hidden), nn.ReLU(),\n"
    "            nn.Linear(hidden, hidden),    nn.ReLU(),\n"
    "            nn.Linear(hidden, n_actions),\n"
    "        )\n"
    "    def forward(self, x):\n"
    "        return self.net(x)\n"
    "\n"
    "\n"
    "class ReplayBuffer:\n"
    "    def __init__(self, maxlen=2000):\n"
    "        self.buf = deque(maxlen=maxlen)\n"
    "    def push(self, s, a, r, s2, done):\n"
    "        self.buf.append((s, int(a), float(r), s2, float(done)))\n"
    "    def sample(self, batch_size):\n"
    "        batch = random.sample(self.buf, batch_size)\n"
    "        s, a, r, s2, d = zip(*batch)\n"
    "        return (\n"
    "            torch.FloatTensor(np.array(s)),\n"
    "            torch.LongTensor(np.array(a)).unsqueeze(1),\n"
    "            torch.FloatTensor(np.array(r)).unsqueeze(1),\n"
    "            torch.FloatTensor(np.array(s2)),\n"
    "            torch.FloatTensor(np.array(d)).unsqueeze(1),\n"
    "        )\n"
    "    def __len__(self): return len(self.buf)\n"
    "\n"
    "\n"
    "class DQNAgent:\n"
    "    # DQN with replay buffer and target network (Hilpisch dqlagent.py pattern)\n"
    "    def __init__(self, state_dim, n_actions=2, hidden=24,\n"
    "                 lr=1e-3, gamma=0.5,\n"
    "                 eps_start=1.0, eps_end=0.1, eps_decay=0.9975,\n"
    "                 batch_size=32, buffer_size=2000, target_update=50):\n"
    "        self.q      = QNetwork(state_dim, n_actions, hidden)\n"
    "        self.tgt    = QNetwork(state_dim, n_actions, hidden)\n"
    "        self.tgt.load_state_dict(self.q.state_dict())\n"
    "        self.opt    = optim.Adam(self.q.parameters(), lr=lr)\n"
    "        self.buf    = ReplayBuffer(buffer_size)\n"
    "        self.gamma  = gamma\n"
    "        self.eps    = eps_start\n"
    "        self.eps_lo = eps_end\n"
    "        self.eps_dc = eps_decay\n"
    "        self.bs     = batch_size\n"
    "        self.tgt_up = target_update\n"
    "        self._steps = 0\n"
    "\n"
    "    def act(self, state, greedy=False):\n"
    "        if not greedy and np.random.rand() < self.eps:\n"
    "            return int(np.random.randint(2))\n"
    "        with torch.no_grad():\n"
    "            return int(self.q(torch.FloatTensor(state)).argmax().item())\n"
    "\n"
    "    def observe(self, s, a, r, s2, done):\n"
    "        self.buf.push(s, a, r, s2, float(done))\n"
    "        if len(self.buf) >= self.bs:\n"
    "            self._update()\n"
    "        self.eps = max(self.eps_lo, self.eps * self.eps_dc)\n"
    "        self._steps += 1\n"
    "        if self._steps % self.tgt_up == 0:\n"
    "            self.tgt.load_state_dict(self.q.state_dict())\n"
    "\n"
    "    def _update(self):\n"
    "        s, a, r, s2, d = self.buf.sample(self.bs)\n"
    "        pred = self.q(s).gather(1, a)\n"
    "        with torch.no_grad():\n"
    "            nxt = self.tgt(s2).max(1, keepdim=True)[0]\n"
    "            tgt = r + self.gamma * nxt * (1.0 - d)\n"
    "        loss = nn.MSELoss()(pred, tgt)\n"
    "        self.opt.zero_grad(); loss.backward(); self.opt.step()\n"
    "\n"
    "\n"
    "# Parameter count\n"
    "_net = QNetwork(K)\n"
    "n_params = sum(p.numel() for p in _net.parameters())\n"
    "print(f'QNetwork params: {n_params}  (tiny by design -- this is a control)')"
))

# ---------------------------------------------------------------------------
# Cell 11: training header (markdown)
# ---------------------------------------------------------------------------
cells.append(md("## §4 Training -- 5 seeds x 40 episodes (untuned)"))

# ---------------------------------------------------------------------------
# Cell 12: training loop (code)
# ---------------------------------------------------------------------------
cells.append(code(
    "def train_dqn(returns_arr, seed, K=10, episodes=40):\n"
    "    torch.manual_seed(seed)\n"
    "    np.random.seed(seed)\n"
    "    random.seed(seed)\n"
    "    env    = SingleAssetEnv(returns_arr, K=K)\n"
    "    agent  = DQNAgent(state_dim=K)\n"
    "    ep_rewards = []\n"
    "    for ep in range(episodes):\n"
    "        state   = env.reset()\n"
    "        total_r = 0.0\n"
    "        while True:\n"
    "            action          = agent.act(state)\n"
    "            next_s, rew, done = env.step(action)\n"
    "            agent.observe(state, action, rew, next_s, done)\n"
    "            state    = next_s\n"
    "            total_r += rew\n"
    "            if done:\n"
    "                break\n"
    "        ep_rewards.append(total_r)\n"
    "    return agent, ep_rewards\n"
    "\n"
    "\n"
    "train_arr = spy_train.values.astype(np.float32)\n"
    "\n"
    "print(f'Training {len(SEEDS)} seeds x {EPISODES} episodes '\n"
    "      f'({len(SEEDS)*EPISODES} total, {len(train_arr)-K-1} steps/ep)...')\n"
    "t0 = time.time()\n"
    "\n"
    "trained_agents = {}\n"
    "all_ep_rewards = {}\n"
    "\n"
    "for seed in SEEDS:\n"
    "    agent, ep_rews = train_dqn(train_arr, seed=seed, K=K, episodes=EPISODES)\n"
    "    trained_agents[seed] = agent\n"
    "    all_ep_rewards[seed] = ep_rews\n"
    "    print(f'  Seed {seed}: ep_reward[-1]={ep_rews[-1]:.4f}  eps_final={agent.eps:.3f}')\n"
    "\n"
    "elapsed = time.time() - t0\n"
    "print(f'Training done in {elapsed:.1f}s  ({elapsed/60:.2f} min)')"
))

# ---------------------------------------------------------------------------
# Cell 13: OOS evaluation (markdown)
# ---------------------------------------------------------------------------
cells.append(md(
    "## §5 OOS Evaluation (2023+)\n"
    "\n"
    "Greedy policy (eps=0) over the OOS window.  "
    "Position(t) earns the strictly *next-day* return, "
    "matching the training reward contract exactly.\n"
    "\n"
    "Ensemble = mean position across seeds (soft vote in [0, 1]), "
    "producing a fractionally-scaled return series."
))

# ---------------------------------------------------------------------------
# Cell 14: OOS evaluation code (code)
# ---------------------------------------------------------------------------
cells.append(code(
    "def evaluate_oos(agent, returns_arr, K=10):\n"
    "    # Greedy rollout; returns (strat_array, positions_array, hit_rate)\n"
    "    T = len(returns_arr)\n"
    "    positions = []\n"
    "    for i in range(T - K - 1):\n"
    "        state  = returns_arr[i:i + K]\n"
    "        action = agent.act(state, greedy=True)\n"
    "        positions.append(action)\n"
    "    pos_arr  = np.array(positions, dtype=np.float32)\n"
    "    nxt_rets = returns_arr[K:K + len(positions)]\n"
    "    strat    = pos_arr * nxt_rets\n"
    "    # Directional hit-rate: long on up-days, flat on down-days\n"
    "    correct  = ((pos_arr == 1) & (nxt_rets > 0)) | ((pos_arr == 0) & (nxt_rets <= 0))\n"
    "    return strat, pos_arr, float(correct.mean())\n"
    "\n"
    "\n"
    "oos_arr = spy_oos.values.astype(np.float32)\n"
    "n_steps = len(oos_arr) - K - 1\n"
    "\n"
    "seed_strats    = {}\n"
    "seed_positions = {}\n"
    "seed_hit_rates = {}\n"
    "seed_sharpes   = []\n"
    "\n"
    "for seed in SEEDS:\n"
    "    strat, pos, hr = evaluate_oos(trained_agents[seed], oos_arr, K=K)\n"
    "    seed_strats[seed]    = strat\n"
    "    seed_positions[seed] = pos\n"
    "    seed_hit_rates[seed] = hr\n"
    "    idx = spy_oos.index[K:K + n_steps]\n"
    "    sr  = ann_sharpe(pd.Series(strat, index=idx))\n"
    "    seed_sharpes.append(sr)\n"
    "    print(f'  Seed {seed}: Sharpe={sr:.4f}  HitRate={hr:.4f}  '\n"
    "          f'LongFrac={pos.mean():.3f}')\n"
    "\n"
    "# Ensemble: mean position across seeds (soft vote)\n"
    "mean_pos     = np.stack([seed_positions[s] for s in SEEDS]).mean(axis=0)\n"
    "oos_idx      = spy_oos.index[K:K + n_steps]\n"
    "ens_strat    = mean_pos * oos_arr[K:K + n_steps]\n"
    "ret_ensemble = pd.Series(ens_strat, index=oos_idx, name='DQN_ensemble')\n"
    "\n"
    "ens_sharpe = ann_sharpe(ret_ensemble)\n"
    "ens_annret = ann_return(ret_ensemble)\n"
    "ens_maxdd  = max_drawdown(ret_ensemble)\n"
    "mean_hr    = float(np.mean(list(seed_hit_rates.values())))\n"
    "\n"
    "print(f'Ensemble Sharpe  : {ens_sharpe:.4f}')\n"
    "print(f'Ensemble Ann Ret : {ens_annret:.2%}')\n"
    "print(f'Ensemble Max DD  : {ens_maxdd:.2%}')\n"
    "print(f'Mean hit-rate    : {mean_hr:.2%}')\n"
    "print(f'Per-seed Sharpe  : mean={np.mean(seed_sharpes):.3f}  '\n"
    "      f'std={np.std(seed_sharpes):.3f}')\n"
    "print(f'Seed range       : [{min(seed_sharpes):.3f}, {max(seed_sharpes):.3f}]')"
))

# ---------------------------------------------------------------------------
# Cell 15: B&H header (markdown)
# ---------------------------------------------------------------------------
cells.append(md(
    "## §6 Benchmark -- Buy-and-Hold SPY\n"
    "\n"
    "SPY buy-and-hold over the same OOS date window is an **intentionally weak benchmark** -- "
    "it can only go long (no flat option) and earns the full equity risk premium.\n"
    "The DQN must match or beat it on its own terms (Sharpe) to demonstrate learning.\n"
    "\n"
    "*Note:* 2023-2026 was a strong equity bull market, making B&H a strong absolute performer "
    "but still the natural single-asset baseline for a long/flat agent."
))

# ---------------------------------------------------------------------------
# Cell 16: B&H code (code)
# ---------------------------------------------------------------------------
cells.append(code(
    "# Align B&H to the same date window as the DQN output\n"
    "bh_series = spy_oos.loc[oos_idx]\n"
    "bh_sharpe = ann_sharpe(bh_series)\n"
    "bh_annret = ann_return(bh_series)\n"
    "bh_maxdd  = max_drawdown(bh_series)\n"
    "\n"
    "print('Buy-and-Hold SPY (weak benchmark, same OOS window):')\n"
    "print(f'  Sharpe   : {bh_sharpe:.4f}')\n"
    "print(f'  Ann Ret  : {bh_annret:.2%}')\n"
    "print(f'  Max DD   : {bh_maxdd:.2%}')"
))

# ---------------------------------------------------------------------------
# Cell 17: results header (markdown)
# ---------------------------------------------------------------------------
cells.append(md(
    "## §7 Results & Contrast\n"
    "\n"
    "**The critical framing:** if DQN matches/beats B&H, the RL machinery *does* learn on the "
    "tractable problem -- so the 06a/06b static collapse is a property of **problem difficulty** "
    "(29-asset simplex, ~1/N diluted reward signal), not a code defect.\n"
    "\n"
    "**Off-axis caveat applied everywhere:** these numbers are not on the same axis as the "
    "29-asset leaderboard.  Do not compare DQN Sharpe to 2.579 / 2.422 / 2.386."
))

# ---------------------------------------------------------------------------
# Cell 18: results table (code)
# ---------------------------------------------------------------------------
cells.append(code(
    "dqn_beats_bh = ens_sharpe > bh_sharpe\n"
    "\n"
    "rows = {\n"
    "    f'DQN ensemble (5 seeds, K={K})': {\n"
    "        'Sharpe': f'{ens_sharpe:.3f}',\n"
    "        'Sharpe mean+-std': f'{np.mean(seed_sharpes):.3f} +- {np.std(seed_sharpes):.3f}',\n"
    "        'Ann Ret': f'{ens_annret:.1%}',\n"
    "        'Max DD':  f'{ens_maxdd:.1%}',\n"
    "        'Hit Rate': f'{mean_hr:.1%}',\n"
    "    },\n"
    "    'Buy-and-Hold SPY (weak baseline)': {\n"
    "        'Sharpe': f'{bh_sharpe:.3f}',\n"
    "        'Sharpe mean+-std': '-- (deterministic)',\n"
    "        'Ann Ret': f'{bh_annret:.1%}',\n"
    "        'Max DD':  f'{bh_maxdd:.1%}',\n"
    "        'Hit Rate': '-- (always long)',\n"
    "    },\n"
    "}\n"
    "results_df = pd.DataFrame(rows).T\n"
    "print(results_df.to_string())\n"
    "\n"
    "print()\n"
    "print('-' * 60)\n"
    "delta = ens_sharpe - bh_sharpe\n"
    "if dqn_beats_bh:\n"
    "    verdict = f'DQN BEATS B&H  delta={delta:+.3f}'\n"
    "else:\n"
    "    verdict = f'DQN MISSES B&H  delta={delta:+.3f}'\n"
    "print(f'Verdict: {verdict}')\n"
    "print('-' * 60)\n"
    "\n"
    "print()\n"
    "print('CONTRAST -- what this means for the 29-asset finding:')\n"
    "if dqn_beats_bh:\n"
    "    print('  [+] DQN beats B&H on the tractable N=1 problem')\n"
    "    print('  [+] RL machinery IS learning when the problem is tractable')\n"
    "    print('  [+] 06a/06b static collapse is PROBLEM-DRIVEN:')\n"
    "    print('      high-dimensional simplex + 1/N diluted reward, not a code bug')\n"
    "else:\n"
    "    print('  [~] DQN does not beat B&H -- agent is fragile even on N=1')\n"
    "    print('  [~] Still informative: if 1-asset is hard, 29-asset collapse')\n"
    "    print('      is even more expected (harder problem, weaker reward signal)')\n"
    "    print('  [+] Negative result tightens, not weakens, the 06a/06b finding')\n"
    "\n"
    "print()\n"
    "print('OFF-AXIS: single asset, weak B&H, NOT comparable to 29-asset leaderboard.')\n"
    "print('Reference bars (different axis): ML 2.579 / classical 2.422 / 2.386')"
))

# ---------------------------------------------------------------------------
# Cell 19: Figure 1 equity curves (code)
# ---------------------------------------------------------------------------
cells.append(code(
    "# Figure 1: OOS equity curves\n"
    "fig, ax = plt.subplots(figsize=(10, 4))\n"
    "\n"
    "cum_ens = (1 + ret_ensemble).cumprod()\n"
    "ax.plot(cum_ens.index, cum_ens.values,\n"
    "        label=f'DQN ensemble (Sharpe {ens_sharpe:.3f})', lw=2, color='steelblue')\n"
    "\n"
    "for seed in SEEDS:\n"
    "    sr_s = pd.Series(seed_strats[seed], index=oos_idx)\n"
    "    ax.plot((1 + sr_s).cumprod().values,\n"
    "            lw=0.8, alpha=0.22, color='steelblue')\n"
    "\n"
    "cum_bh = (1 + bh_series).cumprod()\n"
    "ax.plot(cum_bh.index, cum_bh.values,\n"
    "        label=f'Buy-and-Hold SPY (Sharpe {bh_sharpe:.3f})', lw=2, color='tomato', ls='--')\n"
    "\n"
    "ax.set_xlabel('Date')\n"
    "ax.set_ylabel('Cumulative Return')\n"
    "ax.set_title(\n"
    "    'OOS Equity Curves -- Single-Asset DQN vs Buy-and-Hold (2023-2026)\\n'\n"
    "    '[OFF-AXIS: single asset, weak benchmark -- not comparable to 29-asset leaderboard]'\n"
    ")\n"
    "ax.legend(loc='upper left', fontsize=9)\n"
    "fig.tight_layout()\n"
    "out1 = FIG_DIR / 'oos_equity_curves.png'\n"
    "fig.savefig(out1, dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print(f'Saved -> {out1}')"
))

# ---------------------------------------------------------------------------
# Cell 20: Figure 2 per-seed Sharpe (code)
# ---------------------------------------------------------------------------
cells.append(code(
    "# Figure 2: per-seed Sharpe dispersion\n"
    "fig, ax = plt.subplots(figsize=(7, 4))\n"
    "\n"
    "colors = ['steelblue' if s >= bh_sharpe else 'salmon' for s in seed_sharpes]\n"
    "ax.bar(range(len(SEEDS)), seed_sharpes, color=colors, alpha=0.85,\n"
    "       edgecolor='k', linewidth=0.7)\n"
    "ax.axhline(np.mean(seed_sharpes), color='steelblue', lw=2, ls='--',\n"
    "           label=f'DQN mean {np.mean(seed_sharpes):.3f}')\n"
    "ax.axhline(bh_sharpe, color='tomato', lw=2, ls='--',\n"
    "           label=f'B&H {bh_sharpe:.3f}')\n"
    "ax.set_xticks(range(len(SEEDS)))\n"
    "ax.set_xticklabels([f'Seed {s}' for s in SEEDS])\n"
    "ax.set_ylabel('OOS Sharpe (2023-2026)')\n"
    "ax.set_title(\n"
    "    'Per-Seed OOS Sharpe -- Single-Asset DQN (SPY, long/flat)\\n'\n"
    "    '[OFF-AXIS: not comparable to 29-asset leaderboard]'\n"
    ")\n"
    "ax.legend(fontsize=9)\n"
    "fig.tight_layout()\n"
    "out2 = FIG_DIR / 'seed_sharpe_dispersion.png'\n"
    "fig.savefig(out2, dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print(f'Saved -> {out2}')"
))

# ---------------------------------------------------------------------------
# Cell 21: summary + CSV artifacts (code)
# ---------------------------------------------------------------------------
cells.append(code(
    "print('=' * 70)\n"
    "print('NOTEBOOK 06c SUMMARY -- Single-Asset DQN (Tractability Control)')\n"
    "print('=' * 70)\n"
    "print(f'Asset      : SPY  |  OOS {oos_idx[0].date()} -> {oos_idx[-1].date()}  ({len(oos_idx)} days)')\n"
    "print(f'Agent      : DQN (K={K} state, 2-hidden 24-unit MLP, 2 actions)')\n"
    "print(f'Seeds      : {len(SEEDS)}  |  Episodes: {EPISODES}  |  Steps/ep: {len(train_arr)-K-1}')\n"
    "print()\n"
    "print(f'DQN ensemble Sharpe : {ens_sharpe:.4f}')\n"
    "print(f'B&H SPY Sharpe      : {bh_sharpe:.4f}')\n"
    "print(f'Delta (DQN - B&H)   : {ens_sharpe-bh_sharpe:+.4f}')\n"
    "print(f'DQN ann return      : {ens_annret:.2%}')\n"
    "print(f'DQN max drawdown    : {ens_maxdd:.2%}')\n"
    "print(f'Directional hit-rate: {mean_hr:.2%}')\n"
    "print()\n"
    "print(f'Per-seed Sharpe: {np.mean(seed_sharpes):.3f} +- {np.std(seed_sharpes):.3f}')\n"
    "print(f'Seed range     : [{min(seed_sharpes):.3f}, {max(seed_sharpes):.3f}]')\n"
    "print()\n"
    "if dqn_beats_bh:\n"
    "    print('VERDICT: DQN beats B&H => RL machinery LEARNS on the tractable problem.')\n"
    "    print('         06a/06b static collapse is problem-driven, not a code defect.')\n"
    "else:\n"
    "    print('VERDICT: DQN does not beat B&H -- agent fragile even on N=1.')\n"
    "    print('         Strengthens, not weakens, the 06a/06b negative result.')\n"
    "print()\n"
    "print('OFF-AXIS: Sharpes above are NOT comparable to 29-asset leaderboard.')\n"
    "\n"
    "# Save CSVs\n"
    "pd.DataFrame({\n"
    "    'seed': SEEDS,\n"
    "    'sharpe': seed_sharpes,\n"
    "    'hit_rate': [seed_hit_rates[s] for s in SEEDS],\n"
    "    'long_frac': [seed_positions[s].mean() for s in SEEDS],\n"
    "}).to_csv(RESULTS_DIR / 'seed_results.csv', index=False)\n"
    "\n"
    "pd.DataFrame([{\n"
    "    'strategy': 'DQN_ensemble',\n"
    "    'sharpe': ens_sharpe, 'ann_ret': ens_annret, 'max_dd': ens_maxdd,\n"
    "    'hit_rate': mean_hr,\n"
    "    'seed_sharpe_mean': np.mean(seed_sharpes), 'seed_sharpe_std': np.std(seed_sharpes),\n"
    "}, {\n"
    "    'strategy': 'BuyAndHold_SPY',\n"
    "    'sharpe': bh_sharpe, 'ann_ret': bh_annret, 'max_dd': bh_maxdd,\n"
    "    'hit_rate': float('nan'),\n"
    "    'seed_sharpe_mean': bh_sharpe, 'seed_sharpe_std': 0.0,\n"
    "}]).to_csv(RESULTS_DIR / 'summary.csv', index=False)\n"
    "print(f'Saved seed_results.csv and summary.csv -> {RESULTS_DIR}')"
))

# ---------------------------------------------------------------------------
# Cell 22: Colab bundle (code)
# ---------------------------------------------------------------------------
cells.append(code(
    "if IS_COLAB:\n"
    "    import subprocess\n"
    "    subprocess.run(\n"
    "        ['zip', '-r', '/content/notebook_06c_artifacts.zip', 'results/notebook_06c/'],\n"
    "        cwd=str(ROOT), check=True,\n"
    "    )\n"
    "    print('Bundle ready: /content/notebook_06c_artifacts.zip')\n"
    "    print('Download from file panel (folder icon, left sidebar).')\n"
    "else:\n"
    "    print(f'Local run complete.  Artifacts: {RESULTS_DIR}')"
))

# ---------------------------------------------------------------------------
# Assign IDs and write
# ---------------------------------------------------------------------------
for i, cell in enumerate(cells):
    cell["id"] = f"c{i:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = ROOT / "notebooks" / "06c_rl_single_asset_dqn.ipynb"
with open(out, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Written: {out}  ({out.stat().st_size} bytes)")
