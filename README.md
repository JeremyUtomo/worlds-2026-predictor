# Worlds 2026 Predictor

A League of Legends Worlds 2026 prediction model for Pick'Ems. It is built the same way as
[fastf1-predictor](https://github.com/JeremyUtomo/fastf1-predictor):
- it saves its predictions before each match;
- it scores them once the result is in;
- it keeps correcting itself from its mistakes.

## What the model looks at

| Group | Features (all team A minus team B) |
|---|---|
| **Form** | fast team rating that drifts back to the long-run rating; recent win rate; trend; gold diff @15; gold diff per minute |
| **History** | long-run rating with a region layer, learned only from cross-region games; international win rate (2y); head-to-head (2y, shrunk); Worlds experience of the current roster |
| **Roster** | player ratings that follow players between teams; roster continuity (how much of today's starting five played the team's games over the past year) |
| **Player form** | each starter's recent lane gold lead and CS lead at 15 minutes vs their opponent in the same role, and KDA; follows players across teams |
| **Meta fit** | how meta-heavy each player's champion pool is; how well they win on meta champions; meta depth (ban resilience); over-reliance on one meta pick; style fit (early-game strength × how decisive early leads are in this meta) |

**Roster changes.** A team's record only counts if the same players earned it.
- The rating the model uses is continuity × team rating + (1 − continuity) × average rating of the five starters.
- So a rebuilt team is judged on its new players' own records, and a stable team is unchanged.
- The team's international record and head-to-head fade with continuity.
- The starting five is the lineup announced for that match, so there is no lag after a roster swap.

**Meta** itself is a champion score: pick share × win rate over the last 45 days of pro games.
- International games are weighted ×3.
- The 26.20 Worlds-patch buffs and nerfs in `settings/patch_changes.yaml` adjust the scores. That adjustment fades as games are played on the new patch.

An L2 logistic regression turns the features into P(win a game). Series odds (Bo1, Bo3, Bo5) are then worked out exactly from that.
- `simulate.py` runs the rest of the event with real results locked in: the Play-In (double elimination), Swiss, then knockouts.
- In Swiss, a match is Bo1 unless advancement or elimination is on the line, in which case it is Bo3. That matches Riot's 2026 schedule.
- `pickems.py` picks the slots with the best expected Pick'Ems score.

## Data
Everything is downloaded automatically from the public lolesports API:
- series results for LCK, LPL, LEC, LCS/LTA, CBLOL, LCP/PCS/VCS/LJL, First Stand, MSI, EWC and Worlds;
- per-game live stats: champions, players, gold at 15 minutes, game length, winner.

Finished data is cached in `data/cache/`, so an update only downloads what is new.

## Dashboard: Swiss bracket
Every Swiss match, round by round:
- **Projected bracket** (laid out like the official one): Riot's Round 1 draw, then the favourite wins every match, with each team's win % on every card. Before Riot publishes the draw, Round 1 follows the official rules: Pool 1 vs Pool 4, Pool 2 vs Pool 3, no same-region pairs (`config.SWISS_POOLS`).
- **Every likely matchup:** for each round and record group (e.g. Round 3 · 2-0), the pairings that come up most often across the simulations, how often each happens, and both teams' win %.

## Dashboard: Upcoming matches
The first tab lists the next Worlds matches. Each one shows:
- the predicted winner and win chances;
- the factors behind the call;
- an expandable panel with team form (form rating, recent win rate, gold at 15, last 5 series, roster continuity) and each starter's player form.

With **Auto-update after each match** on, the app checks Riot's Worlds schedule every 2 minutes. When a match finishes it re-runs the full loop below, so team form, player form and player ratings include the new games. A **Last model update** box then shows:
- whether the pre-match prediction was right;
- which teams' odds moved the most.

## Learning loop

```bash
python main.py update      # fetch -> score past predictions -> recalibrate -> log new predictions -> picks
```

1. **Log:** every Worlds match whose teams are known and that hasn't started is written to `data/prediction_log.csv`. Once a match starts, its row is never changed.
2. **Score:** once a match is played, the run checks whether the favourite won and records log loss and Brier score.
   - A **mistakes report** lists each upset and which feature group (form, history or meta fit) pushed toward the team that lost.
3. **Recalibrate:** one temperature is fitted per format (Bo1, Bo3, Bo5).
   - The data is a walk-forward backtest of every international event plus the scored live predictions.
   - With few matches, the temperature is pulled toward 1 using `T = 1 + (T_raw-1)·n/(n+5)`.
4. **Refit:** the model is retrained with the new Worlds games weighted up. Meta scores update from the champions actually being picked on 26.20.
5. **Research:** `python evaluate.py --tune` tests one change at a time on 2024–25 international events, with 2026 events held out.
   - A change is kept only if log loss improves by more than 0.003 and Brier score doesn't get worse.
   - It writes `model_config.json` and `research_report.md`, including an ablation of each feature group.

## Current results (walk-forward, series level)
Full details are in `research_report.md`.
- **2026 events (First Stand, MSI, EWC, held out):** ~70% of series called correctly.
  - Log loss is 0.657 after self-calibration. A coin flip scores 0.693.
  - The calibration temperatures were learned only from 2024–25 events.
- **Roster-aware rating:** in matches after a big roster change, log loss improves from 0.635 to 0.631 across ~6,200 pro games.
- **Team-level history** (international win rate, head-to-head) made predictions worse in the tuning run, so the model leaves it out.
- **Bo3 temperature is ~2.3:** Bo3 series are much less predictable than independent games would imply, so the model flattens them a lot.

## Usage

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python main.py update            # first run downloads ~2 years of games (~30 min)
python main.py pickems           # Pick'Ems recommendation from cached data
python main.py match GEN T1 --bo 5
python main.py meta              # champion meta tiers + team meta fit
python evaluate.py --tune        # re-tune the model
streamlit run app.py             # dashboard
```

When Riot draws the quarterfinals, put the matchups in `settings/bracket.yaml`:

```yaml
quarterfinals:   # QF1 & QF2 meet in SF1, QF3 & QF4 in SF2
  - [GEN, TES]
  - [BLG, G2]
  - [HLE, AL]
  - [T1, DK]
```

Pick'Ems scoring is set in `config.PICKEMS`. The exact-record bonus is an estimate.

## Deployment (worlds.jeremyutomo.com)
The setup copies fastf1-predictor's: Docker Compose on Ubuntu 24.04, Caddy for automatic HTTPS, and a pull-based auto-deploy.

| Service | What it does |
|---|---|
| `app` | The Streamlit site (`WORLDS_MODE=server`). It only reads the model state and reloads within a minute of each update. |
| `updater` | `python live.py --loop`. Every 2 minutes it checks Riot's Worlds schedule. After a finished match, or when new pairings are drawn, it re-runs the full loop: fetch, refit team and player form, score, recalibrate, log predictions, simulate. |
| `caddy` | HTTPS for `$DOMAIN` (default `worlds.jeremyutomo.com`), proxying to `app:8501`. |

Runtime state lives in `/var/lib/worlds` on the host, mounted at `/data`. A fresh server starts from the snapshot in `seed/`, so the first start doesn't need to download 2 years of games.

**First deploy:**
1. Add a DNS **A record**: `worlds.jeremyutomo.com` → the server's IP.
2. On the server, run:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/JeremyUtomo/worlds-2026-predictor/main/deploy/setup.sh | sudo bash
   ```

After that, every push to `main` that passes CI is deployed within about 2 minutes by `worlds-deploy.timer`.

**Same server as fastf1-predictor?** Both stacks run their own Caddy on ports 80/443, and only one can hold those ports. So either:
- run Worlds on its own small instance, or
- let a single Caddy serve both sites: add a `worlds.jeremyutomo.com { reverse_proxy worlds-app:8501 }` block to it, and attach both stacks to a shared Docker network.

