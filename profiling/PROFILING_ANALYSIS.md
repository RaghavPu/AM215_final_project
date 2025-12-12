# Performance Profiling Analysis

## Summary

| Metric | Value |
|--------|-------|
| **Total Runtime** | 492.3 seconds (~8.2 minutes) |
| **Classification** | **CPU-bound** |
| **Primary Bottleneck** | State computation & pandas indexing |

## Profiling Command

```bash
python -m cProfile -s cumtime run.py --model markov 2>&1 | head -80
```

## Top 10 Bottlenecks

| Rank | Function | Time (s) | % of Total | Location |
|------|----------|----------|------------|----------|
| 1 | `run_cross_validation` | 449.8 | 91.4% | `cross_validation.py:190` |
| 2 | `compute_inventory_metrics` | 242.8 | 49.3% | `metrics.py:103` |
| 3 | `inventory_to_states` | 240.6 | 48.9% | `metrics.py:22` |
| 4 | pandas `__setitem__` | 185.1 | 37.6% | `indexing.py:883` |
| 5 | `MarkovModel.fit` | 144.9 | 29.4% | `markov.py:51` |
| 6 | pandas `__getitem__` | 105.8 | 21.5% | `series.py:1107` |
| 7 | `track_inventory` | 51.5 | 10.5% | `cross_validation.py:75` |
| 8 | `load_trip_data` | 34.9 | 7.1% | `data_loader.py:10` |
| 9 | CSV parsing | 25.7 | 5.2% | `c_parser_wrapper.py:222` |
| 10 | `_build_global_fallback` | 29.0 | 5.9% | `markov.py:170` |

## Analysis

### CPU-Bound Classification

This codebase is **CPU-bound**, not I/O-bound. Evidence:

1. **Data loading is only 7.1%** of total time (34.9s out of 492.3s)
2. **91% of time is in cross-validation** - pure computation
3. **Pandas indexing operations dominate** - `__setitem__` and `__getitem__` account for ~59% of runtime

### Bottleneck Breakdown

```
Cross-Validation Loop (91.4%)
├── State Computation (48.9%)
│   └── inventory_to_states() - classifying empty/normal/full
├── Pandas Indexing (37.6%)
│   └── DataFrame row-by-row operations
├── Model Fitting (29.4%)
│   └── Building Markov transition matrices
└── Inventory Tracking (10.5%)
    └── Simulating bike movements hour-by-hour
```

### Why No Parallelization?

Despite being CPU-bound, parallelization is not beneficial because:

1. **Sequential dependencies**: Each timestep depends on the previous state
2. **Cross-validation structure**: Folds must be processed sequentially (training data must precede test data)
3. **Pandas overhead**: The bottleneck is in pandas indexing, which doesn't parallelize well

### Potential Optimizations (Future Work)

1. **Vectorize state computation**: Replace row-by-row iteration with vectorized numpy operations
2. **Use numpy arrays**: Avoid pandas DataFrames in hot loops
3. **Precompute station capacities**: Cache static data outside loops
4. **Sparse matrices**: Use scipy.sparse for transition matrices with many zeros

## Raw Profiling Output

See `profiling_output.txt` for the complete cProfile output.

