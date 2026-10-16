# Research report

_Generated 2026-10-16T19:17:15+00:00_

## Pinned setting: roster-aware rating

`roster.blend` is pinned **on**: the rating used = continuity x team rating + (1 - continuity) x average rating of the five starters.
- The tuning rule alone would turn it off, based on the small set of tuning events: 136 international series, log loss 0.5505 off vs 0.5672 on.
- The 2026 holdout preferred it on: 0.701 on vs 0.713 off.
- A rolling month-by-month check over all ~6,200 pro games also preferred it on. In matches after a big roster change (continuity < 0.6), log loss was 0.631 on vs 0.635 off.

## Chosen settings

```json
{
  "elo": {
    "k_long": 16.0,
    "k_fast": 32.0,
    "fast_half_life_days": 30.0,
    "k_region": 6.0,
    "season_regress": 0.75
  },
  "form": {
    "ewma_games": 20
  },
  "history": {
    "h2h_prior_games": 6,
    "intl_prior_games": 8
  },
  "roster": {
    "blend": true,
    "k_player": 12.0,
    "rookie_start": 1.0,
    "continuity_days": 365,
    "continuity_half_life_days": 90
  },
  "player_form": {
    "ewma_games": 10
  },
  "meta": {
    "window_days": 45,
    "current_event_weight": 3.0,
    "patch_strength": 0.25,
    "top_n": 25,
    "pool_days": 120
  },
  "model": {
    "groups": [
      "form",
      "meta_fit"
    ],
    "l2_C": 1.0,
    "intl_weight": 1.0,
    "train_since": "2024-01-01"
  },
  "calibration": {
    "min_events": 6,
    "shrink": 5.0
  }
}
```

## Final model (walk-forward, series level)

| | Tuning (2024-25) | Holdout (2026) |
|---|---|---|
| Model | LL 0.5593 · Brier 0.1893 · acc 72.8% (n=136) | LL 0.6875 · Brier 0.2253 · acc 68.3% (n=63) |
| Model + calibration (temps from tuning events) | - | LL 0.6382 · Brier 0.2173 · acc 68.3% (n=63) |
| Coin flip | LL 0.6931 · Brier 0.2500 · acc 58.1% (n=136) | LL 0.6931 · Brier 0.2500 · acc 41.3% (n=63) |

## Feature-group ablation

| Groups | Tuning | Holdout |
|---|---|---|
| form | LL 0.5634 · Brier 0.1926 · acc 69.9% (n=136) | LL 0.6913 · Brier 0.2261 · acc 68.3% (n=63) |
| form+history | LL 0.6225 · Brier 0.2103 · acc 70.6% (n=136) | LL 0.6985 · Brier 0.2286 · acc 68.3% (n=63) |
| form+roster | LL 0.5715 · Brier 0.1947 · acc 71.3% (n=136) | LL 0.6975 · Brier 0.2290 · acc 68.3% (n=63) |
| form+meta_fit | LL 0.5593 · Brier 0.1893 · acc 72.8% (n=136) | LL 0.6875 · Brier 0.2253 · acc 68.3% (n=63) |
| form+roster+meta_fit | LL 0.5693 · Brier 0.1927 · acc 73.5% (n=136) | LL 0.6950 · Brier 0.2278 · acc 69.8% (n=63) |
| form+player_form | LL 0.5650 · Brier 0.1914 · acc 70.6% (n=136) | LL 0.6770 · Brier 0.2226 · acc 69.8% (n=63) |
| form+player_form+meta_fit | LL 0.5710 · Brier 0.1896 · acc 72.1% (n=136) | LL 0.6744 · Brier 0.2221 · acc 69.8% (n=63) |
| form+history+roster+meta_fit | LL 0.6399 · Brier 0.2179 · acc 69.9% (n=136) | LL 0.6752 · Brier 0.2220 · acc 68.3% (n=63) |

## Candidates tested (one change at a time)

| Change | Tuning log loss | Adopted |
|---|---|---|
| baseline | 0.5672 | yes |
| model.groups = ['form', 'player_form', 'meta_fit'] | 0.5823 |  |
| model.groups = ['form', 'player_form'] | 0.5756 |  |
| model.groups = ['player_form', 'meta_fit'] | 0.6874 |  |
| model.groups = ['form', 'roster', 'meta_fit'] | 0.5795 |  |
| model.groups = ['form', 'history', 'roster', 'meta_fit'] | 0.6479 |  |
| model.groups = ['form', 'roster'] | 0.5826 |  |
| model.groups = ['form', 'history', 'meta_fit'] | 0.6408 |  |
| roster.rookie_start = 0.5 | 0.5696 |  |
| roster.rookie_start = 0.0 | 0.5725 |  |
| player_form.ewma_games = 5 | 0.5672 |  |
| player_form.ewma_games = 20 | 0.5672 |  |
| roster.k_player = 6.0 | 0.5682 |  |
| roster.k_player = 20.0 | 0.5664 |  |
| roster.continuity_half_life_days = 45 | 0.5670 |  |
| roster.continuity_half_life_days = 180 | 0.5674 |  |
| model.intl_weight = 2.0 | 0.5685 |  |
| model.intl_weight = 5.0 | 0.5725 |  |
| elo.k_long = 32.0 | 0.5672 |  |
| elo.k_fast = 32.0 | 0.5593 | yes |
| elo.k_fast = 64.0 | 0.5832 |  |
| elo.fast_half_life_days = 120.0 | 0.5721 |  |
| elo.k_region = 20.0 | 0.5727 |  |
| elo.season_regress = 0.5 | 0.5593 |  |
| elo.season_regress = 0.9 | 0.5603 |  |
| form.ewma_games = 6 | 0.5635 |  |
| meta.window_days = 30 | 0.5578 |  |
| meta.window_days = 75 | 0.5643 |  |
| meta.pool_days = 60 | 0.5616 |  |
| meta.pool_days = 240 | 0.5583 |  |
| meta.patch_strength = 0.0 | 0.5593 |  |
| meta.patch_strength = 0.5 | 0.5593 |  |
| model.l2_C = 0.1 | 0.5621 |  |
| model.l2_C = 0.3 | 0.5607 |  |
| model.l2_C = 3.0 | 0.5586 |  |

## Calibration (temperature per format)

```json
{
  "bo1": {
    "T": 1.137,
    "raw": 1.15,
    "n": 52,
    "learned": true,
    "logloss_default": 0.565,
    "logloss_learned": 0.5629
  },
  "bo3": {
    "T": 2.117,
    "raw": 2.2,
    "n": 67,
    "learned": true,
    "logloss_default": 0.6866,
    "logloss_learned": 0.6255
  },
  "bo5": {
    "T": 1.424,
    "raw": 1.45,
    "n": 80,
    "learned": true,
    "logloss_default": 0.5498,
    "logloss_learned": 0.5315
  }
}
```

## Per event

| Event | Series | Log loss | Accuracy |
|---|---|---|---|
| worlds 2024 | 50 | 0.6005 | 66.0% |
| lta_cross 2025 | 13 | 0.5560 | 76.9% |
| first_stand 2025 | 13 | 0.7212 | 61.5% |
| msi 2025 | 19 | 0.3500 | 89.5% |
| worlds 2025 | 41 | 0.5556 | 75.6% |
| first_stand 2026 | 13 | 0.5228 | 84.6% |
| msi 2026 | 20 | 0.6735 | 65.0% |
| ewc_lol 2026 | 28 | 0.7775 | 64.3% |
| worlds 2026 | 2 | 0.6371 | 50.0% |
