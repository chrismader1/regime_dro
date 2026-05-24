# regime_dro

Distributionally Robust Portfolio Optimization with Wasserstein ambiguity sets, conditioned on regime labels.

Generic production tool. Paper-specific experiments (comparison vs MVO/DRO, SPX benchmarking, jackknife, synthetic data) live in [`regime_dro_experiments`](https://github.com/chrismader1/regime_dro_experiments).

## Install

```
pip install git+https://github.com/chrismader1/regime_dro.git
```

Optional GPU acceleration:

```
pip install "regime_dro[gpu] @ git+https://github.com/chrismader1/regime_dro.git"
```

## Quick start

```python
import pandas as pd
from regime_dro import run_regdro

prices = pd.read_csv("my_prices.csv", index_col=0, parse_dates=True)

CONFIG = {
    "RSLDS": [{"config": "[g,v]", "n_regimes": 6, "dim_latent": 2}],
    "DATA":       {"start_dt": "2017", "end_dt": None},
    "REBAL":      {"min_lookback": 11, "max_lookback": 252,
                   "rebalance_period": 4, "freq": "W-FRI",
                   "regdro_rebal_method": "periodic"},
    "PORTFOLIO":  {"risk_budget": 0.20, "risk_free_rate": 0.0, "max_cash": 0.0,
                   "max_pos_size": 0.05, "no_shorting": True, "no_leverage": True,
                   "sigma_shrinkage_lambda": 0.1, "delta_name": "bootstrap_np",
                   "epsilon_sigma": 1e-6},
    "EXECUTION":  {"execution_delay": 1, "trading_cost": 0.0020},
    "DELTA_DEFAULTS": {
        "bootstrap_np": {"delta_method": "bootstrap_np", "alpha": 0.05, "B": 100, "seed": 0},
    },
    "results_csv":      "data/gridsearch_results.csv",
    "segments_parquet": "data/gridsearch_segments.parquet",
    "dro_pickle":       "out/regdro_result.pkl.gz",
}

out = run_regdro(prices, CONFIG, verbose=True)
# out["daily"]    — daily PnL series (net of costs, with execution delay)
# out["holdings"] — month-end effective weights
# out["summary"]  — OOS stats dict
# out["fit"]      — piecewise fit (weights per regime-change date)
```

## Public API

### Core algorithm
- **Delta construction (Wasserstein radius):** `compute_delta`, `wasserstein2_gaussian`, `sliced_w2_empirical`, `bootstrap_np_block_delta`, `bootstrap_gaussian_block_delta`
- **Optimizer:** `solve_optimizer`, `solve_dro`, `psd_factor_LtL`
- **Fit wrappers:** `fit_mvo`, `fit_dro`, `fit_regime_dro`, `fit_mvo_rebalanced`, `fit_dro_rebalanced`, `fit_dro_reverse`, `fit_regime_dro_rev_constSigma`
- **Window moments:** `compute_mean_from_window`, `compute_cov_from_window`
- **Evaluation:** `evaluate_portfolio`, `evaluate_regime_independently`, `portfolio_stats`, `portfolio_stats_multipiece`, `stats_from_series`
- **Hypothesis tests:** `hypothesis_tests`, `paired_onesided_less`, `superiority_paired`, `paired_two_sided_test_with_ci`
- **Reporting:** `oos_summary`, `print_oos_table`, `print_single_portfolio_block`, `print_regime_block`
- **IO:** `load_out`, `save_out`

### Orchestration
- **Pipeline:** `run_regdro` — end-to-end generic RegDRO run on any price panel + labels
- **Artifacts:** `load_artifacts`, `select_best_config`, `labels_from_segments_df`, `map_labels_to_calendar`, `snap_start_prev`, `permissible_tuples_from_CONFIG`, `tuples_in_results_csv`
- **Context:** `regdro_decision_context`, `make_solver_cfg_from_CONFIG`, `make_index_rebal`, `make_index_union`, `pnl_with_delay_and_cost`, `period_ends`, `gross_exp_on_window`
- **Schema:** `validate_artifacts`, `REQUIRED_RESULTS_COLS`, `REQUIRED_SEGMENTS_COLS`

## Regime label file-contract

`regime_dro` does not fit regime labels. The user supplies two files matching these schemas.

**results_csv** — per-(security, config, n_regimes, dim_latent) scoring table.

| column     | dtype   | description                                           |
|------------|---------|-------------------------------------------------------|
| security   | string  | Ticker or identifier.                                 |
| config     | string  | Feature-set label (e.g. "[g,v]", "factor2_ff3").      |
| n_regimes  | int     | Number of regimes.                                    |
| dim_latent | int     | Latent dimension.                                     |
| score      | float   | Model-selection criterion. Higher is better.          |

**segments_parquet** — daily regime label time series.

| column     | dtype          | description                          |
|------------|----------------|--------------------------------------|
| security   | string         | Ticker or identifier.                |
| date       | datetime64[ns] | Observation date.                    |
| config     | string         | Matches results_csv.                 |
| n_regimes  | int            | Matches results_csv.                 |
| dim_latent | int            | Matches results_csv.                 |
| z          | int            | Regime label at that date.           |

Schema validation runs automatically on `load_artifacts`.

## Citation

Mader, C. *Distributionally Robust Portfolio Optimization under Regime-Switching Dynamics.* Johns Hopkins University.

## License

MIT.
