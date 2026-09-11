# Does a learned policy actually beat the heuristic it replaces

[![tests](https://github.com/JAYANSHUBADLANI/inventory-pricing-rl/actions/workflows/tests.yml/badge.svg)](https://github.com/JAYANSHUBADLANI/inventory-pricing-rl/actions/workflows/tests.yml)

A sequential inventory replenishment decision, calibrated to a real, sharply
seasonal SKU, comparing a tuned order up to heuristic, a contextual bandit,
tabular Q learning, and a REINFORCE policy gradient agent, then testing
whether any advantage survives when the evaluation conditions are not
exactly the ones training was tuned against.

## Headline result

A REINFORCE policy gradient agent beats the tuned heuristic by roughly 60
percent on held out reward, -425 versus -1079, and keeps that advantage in
every one of eight out of sample robustness conditions tested. A
contextual bandit and a tabular Q learning agent do not beat the heuristic
in any condition: both converge to the same degenerate policy of almost
never ordering, a real and reproducible finding described below, not a
result I am glossing over because it favors the more interesting method.

## The problem

An operations team stocks StockCode 85048, "15CM CHRISTMAS GLASS BALL 20
LIGHTS", a real item from the Online Retail II dataset already used
elsewhere in this portfolio (see `Online Retail Project`). Demand for this
item is close to zero for most of the year and spikes sharply in the weeks
before Christmas, a calibrated peak seasonal multiplier of roughly 25 times
the off season baseline. That is real, uninvented non stationary structure:
seasonal, and severe enough that a policy has to actually plan ahead rather
than just react.

Environment mechanics: periodic review with a 3 day lead time between
placing an order and it arriving, a holding cost of 0.05 per unit per day,
a lost sale penalty of 1.20 per unit of unmet demand, and a fixed cost of
2.0 charged whenever an order is placed at all, which makes ordering on
every single day a genuinely bad idea and rewards a policy that orders
deliberately rather than constantly. Orders are discretized to steps of 4
units up to a maximum of 40 per order. See `src/demand_calibration.py` for
how the seasonal curve and the negative binomial demand dispersion were
fit from the raw transaction data, and `src/environment.py` for the full
simulation.

## The four policies

**Heuristic.** An order up to level, s and S, rule, the kind an operations
team runs today. Tuned by grid search against the training demand
distribution, landing on s=4, S=40, before any learned policy was compared
against it.

**Contextual bandit.** Sees inventory position, days until the next
delivery, season, and calendar position as context, picks an order
quantity to maximize immediate reward with a linear model per action
updated online, ridge regression plus epsilon greedy exploration. Has no
mechanism to value an action based on what happens several days later.

**Tabular Q learning.** Discretizes the same observation into a table and
learns a value for each state, action pair through temporal difference
updates, which in principle lets it value an order today based on the
stockouts it prevents several days later once the lead time has passed.

**REINFORCE.** A two layer neural network maps the continuous observation
directly to action probabilities, trained with Monte Carlo policy
gradient. No hand chosen discretization.

## Results

In sample, 300 held out episodes, 95 percent confidence interval:

| Policy | Mean reward | 95% CI |
|---|---|---|
| Heuristic (s=4, S=40) | -1079.3 | (-1085.8, -1072.9) |
| Contextual bandit | -1513.1 | (-1518.2, -1507.9) |
| Tabular Q learning | -1513.1 | (-1518.2, -1507.9) |
| REINFORCE | -425.3 | (-428.7, -421.8) |

The bandit and tabular Q learning rows are identical to three decimal
places. That is not a copy paste error. Both converge to the same
degenerate policy, almost never placing an order, and evaluated against
the same demand draws that produces byte identical reward trajectories.
See the debugging notes in `PROGRESS.md` for how this was confirmed rather
than assumed.

Training cost, environment interactions to reach the reported performance:
the heuristic needed 1,606,000 interactions during grid search tuning and
zero at deployment, since it is a fixed rule from then on. The bandit,
tabular Q learning, and REINFORCE each used 1,825,000 interactions, 5000
training episodes of 365 days, and need to keep training or at least keep
their learned parameters to act at deployment time. A team weighing
whether to build any of this against just running the heuristic should
weigh that cost against the outcome above: only one of the three learned
approaches earned it back.

Determinism: re-running evaluation with the same seed against the same
trained policy produces the same result for all four policies, checked
over 20 episodes each. RL training itself was not re-run multiple times
from different seeds to check training stability; that would be a
reasonable next step and is not covered here.

### Robustness, the part that actually earns this its keep

Every trained policy, evaluated out of sample against conditions it never
saw during training: demand shifted up, a different lead time than trained
against, and a regime shift, a permanent step change in baseline demand,
timed at points training never saw and in both directions.

| Condition | Heuristic | Bandit | Tabular Q | REINFORCE |
|---|---|---|---|---|
| Baseline, in sample | -1067.5 | -1504.8 | -1504.8 | **-422.6** |
| Demand +10% | -1198.7 | -1653.7 | -1653.7 | **-453.5** |
| Demand +25% | -1388.4 | -1887.7 | -1887.7 | **-535.3** |
| Lead time +2 days | -1232.4 | -1504.8 | -1504.8 | **-595.8** |
| Lead time -2 days | -824.4 | -1504.8 | -1504.8 | **-390.5** |
| Regime shift, day 60, up | -1780.9 | -2385.5 | -2385.5 | **-980.7** |
| Regime shift, day 200, up | -1756.3 | -2378.2 | -2378.2 | **-973.3** |
| Regime shift, day 120, down | -586.9 | -772.4 | -772.4 | **-355.6** |

Bold marks the best policy in each row. REINFORCE's advantage survives
every condition tested, including lead times and regime shift timings it
was never trained against. The bandit and tabular Q learning's shared
degenerate policy also stays remarkably stable across conditions, which
makes sense: a policy that almost never orders barely reacts to changes in
demand or lead time in the first place, so its cost changes only because
the environment's cost accounting changes, not because the policy adapts.

## Why the bandit and tabular Q learning both failed the same way

Two independent bugs were found and fixed during development and are
documented in full in `PROGRESS.md`, along with the actual numbers at each
stage. The short version:

A season feature with a roughly 500x dynamic range, 0.05 to 25, was fed
into a tabular Q learning bucketing scheme built assuming something closer
to a 0 to 3 range. The entire Christmas window collapsed into one
saturated bucket, so the table could not distinguish a mildly busy day from
the single highest demand day of the year. Fixed with a log transform on
the season feature. Fixing this alone did not change the result.

What did not change after that fix, and was confirmed by trying three very
different exploration schedules and getting the same answer to one decimal
place each time, is that both the bandit and tabular Q learning converge
to an almost always order zero policy. This is a real property of the
reward structure and the state representation available to these two
methods, not undertraining: a fixed cost of 2.0 charged nearly every time
an order is placed, against 365 days a year of which only a few weeks
carry real stockout risk, makes never ordering a strong, easily discovered
local optimum, and single step temporal difference updates plus a coarse
discretization were not enough to reliably discover and credit the
narrower, better timed alternative that the exhaustively grid searched
heuristic found.

A separate implementation bug in the REINFORCE agent, an unnormalized
inventory feature causing a feedback loop through the network's random
initialization, and a scalar baseline that could not represent how return
magnitude naturally shrinks toward the end of a fixed length episode, are
also documented in `PROGRESS.md`. Both were fixed before the results above
were produced.

## Sharpest ways this could be wrong

The demand model is calibrated to one real SKU's sales pattern but is
still a negative binomial approximation with a smoothed seasonal
multiplier, not the actual generating process, and this SKU's extreme
seasonality, near zero most of the year, may not represent demand patterns
for less extreme items where the fixed ordering cost trade off that
trapped the bandit and Q learning would look different.

The REINFORCE result depends on the position indexed baseline fix
documented in `PROGRESS.md`, which is a reasonable approximation given a
fixed training calendar but is not a real learned value function. A proper
actor critic implementation was not attempted and might behave differently,
better or worse.

The robustness conditions tested are all conditions I chose. They cover
demand level, lead time, and regime shift timing and direction, but they
were not adversarially searched for the worst case against REINFORCE
specifically, so "survives every condition tested" should not be read as
"survives any plausible condition."

Tabular Q learning's discretization was fixed once, for the season bug,
and not iterated further once three exploration schedules gave the same
answer. It is plausible that a finer state discretization, for example
splitting the top season bucket further, would close some or all of the
gap to REINFORCE. That was not tried, and is noted as left undone in
`PROGRESS.md` rather than assumed away.

## Reproducing this

```
cd inventory-pricing-rl
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy matplotlib torch pytest
cd src
# demand_calibration.py reads the cleaned Online Retail II extract from a sibling
# checkout and will not resolve from a fresh clone. Its output is committed at
# data/daily_demand_real.csv and data/demand_calibration.json, so skip it and
# start at run_experiment.py unless you are regenerating the calibration.
python3 demand_calibration.py
python3 run_experiment.py
python3 robustness.py
python3 make_figures.py
cd ..
python3 -m pytest tests/ -v
```

All random seeds are pinned; the main results above come from seed 7,
recorded in `results/in_sample_results.json`.

## Test suite

24 of 24 tests pass, covering the environment's cost accounting and lead
time mechanics, the heuristic's reorder logic, the contextual bandit's
update rule, the Q table's discretization bounds and update direction, and
the REINFORCE agent's input normalization and position baseline.
