# regime_dro - Distributionally Robust Portfolio Optimization under Regime-Switching Dynamics

from regime_dro.arrays import (
    asnumpy_strict,
    asxp,
    _to_xp,
    _is_cupy_array,
)

from regime_dro.io import (
    load_out,
    save_out,
)

from regime_dro.delta import (
    compute_delta,
    wasserstein2_gaussian,
    sliced_w2_empirical,
    bootstrap_np_block_delta,
    bootstrap_gaussian_block_delta,
)

from regime_dro.optimizer import (
    solve_optimizer,
    solve_dro,
    psd_factor_LtL,
)

from regime_dro.windows import (
    compute_mean_from_window,
    compute_cov_from_window,
    _window_start,
)

from regime_dro.fit import (
    fit_mvo,
    fit_dro,
    fit_regime_dro,
    fit_dro_reverse,
    fit_regime_dro_rev_constSigma,
    fit_mvo_rebalanced,
    fit_dro_rebalanced,
    _feasible_placeholder,
    _all_zero_weights,
)

from regime_dro.evaluate import (
    evaluate_portfolio,
    evaluate_regime_independently,
    stats_from_series,
    _max_drawdown_from_series,
    portfolio_stats,
    portfolio_stats_multipiece,
    _avg_holding_period_from_marks,
)

from regime_dro.hypothesis import (
    hypothesis_tests,
    paired_onesided_less,
    superiority_paired,
    paired_two_sided_test_with_ci,
)

from regime_dro.report import (
    oos_summary,
    print_oos_table,
    print_single_portfolio_block,
    print_regime_block,
    _print_mu_by_name,
    _section,
)

from regime_dro.schema import (
    validate_artifacts,
    REQUIRED_RESULTS_COLS,
    REQUIRED_SEGMENTS_COLS,
)

from regime_dro.artifacts import (
    load_artifacts,
    map_labels_to_calendar,
    snap_start_prev,
    select_best_config,
    labels_from_segments_df,
    available_cfg_tuples_from_parquet,
    permissible_tuples_from_CONFIG,
    tuples_in_results_csv,
)

from regime_dro.context import (
    regdro_decision_context,
    make_solver_cfg_from_CONFIG,
    gross_exp_on_window,
    make_index_rebal,
    make_index_union,
    expand_daily_weights,
    pnl_with_delay_and_cost,
    period_ends,
)

from regime_dro.pipeline import (
    run_regdro,
)
