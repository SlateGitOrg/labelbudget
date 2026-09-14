# labelbudget

> An active-learning benchmark reporting labels-to-target-accuracy with variance across seeds, next to the random baseline.

> **Implementation note.** The runnable core is pure Python standard library. Real text datasets (AG News, 20 Newsgroups) and sentence-transformers embeddings are replaced by a synthetic generator with a *planted* linear Bayes boundary (Bayes balanced accuracy 0.841), scikit-learn by a hand-written L2 Newton/IRLS logistic regression, Plotly by an ASCII median/IQR band chart, DuckDB by in-memory results, pytest by unittest. BALD is not implemented (it needs a Bayesian/ensemble model); uncertainty, margin, information density, core-set and random are.

`COMPACT` · **AI / ML Engineering** · Intermediate · ~5 days · Healthcare - clinical note classification with expensive expert labelling

**Primary language:** Python
**Tags:** `active-learning`, `benchmarking`, `reproducibility`, `label-efficiency`, `evaluation`

---

## The problem

Labelling requires a clinician at 90 an hour. A team labels 5,000 examples because that felt safe. They might have reached the same accuracy with 900 well-chosen ones, or they might have needed 8,000 - and nobody measured, so the next project guesses again, and the guess is the budget.

## ⭐ The differentiator

Reports **labels-required-to-reach-target-accuracy with variance across many seeds**, not a single accuracy-versus-labels curve. Active learning results are notoriously seed-sensitive: a single run routinely shows a 2x advantage that vanishes on replication. This benchmark runs every strategy across many seeds and reports whether the difference is real - and keeps the **random-sampling baseline prominent**, which is the baseline active-learning results most often beat by very little.

This is the sentence to lead with when someone asks you to walk through the
project. Everything else in this repo exists to make it true and to prove it.

## Data

Public text classification datasets (AG News, 20 Newsgroups, and openly licensed clinical-adjacent sets) with a simulated labelling oracle and a configurable cost model.

> No paid API key is required to run or demo this project. Where a paid
> service would add value it is wired as an optional enhancement behind an
> interface with an offline mock as the default implementation.

## Stack

- Python: scikit-learn, sentence-transformers (local embeddings)
- DuckDB, Plotly
- pytest

## Core capabilities

- Pluggable acquisition strategies: uncertainty, margin, core-set, BALD, and random
- Multi-seed experiment runner operating under a fixed labelling budget model
- Labels-to-target-accuracy as the headline metric, with variance bands per strategy
- Cold-start analysis identifying where each strategy is *worse* than random
- Cost report translating label counts into money at a configurable hourly rate

## Repository layout

```
src/strategies/
src/runner/
src/report/
results/
test/
```

## Build plan

1. Runner and the random baseline first. Everything is measured against random.
2. One seed per strategy, then thirty. Compare your conclusions from each - that comparison is the lesson.
3. Labels-to-target metric and variance bands.
4. Cost report last; it is what makes this land with a manager.

## Testing strategy

Assert reproducibility given a seed. Assert the **random baseline is always run and reported**. Assert that strategy ranking is stable across a stated number of seeds **before any comparative claim is emitted** - the guard that stops the tool reporting a difference that is noise.

Tests assert **correctness**, not merely that the code runs. A green suite on
this repo is a claim about behaviour under adversarial conditions; treat any
test that would pass against a deliberately broken implementation as a bug in
the test.

## Quality & safety layer

The tool refuses to declare a winner when confidence bands overlap, reporting 'indistinguishable at this seed count' instead.

## Measurable outcome

> A chart with variance bands showing two strategies that look different on one seed and are statistically indistinguishable across thirty - and the label budget that actually reaches target accuracy.

State it in these terms — business units, not technical ones — in your CV
bullet and in the first thirty seconds of describing the project.

## Measured results

From `python -m src.demo` (30 seeds, pool 300, seed set 4 random labels, batch 8, budget 164, target = 97% of the full-pool score; mean full-pool balanced accuracy 0.832 against a planted Bayes ceiling of 0.841; runtime 21.8s on the author machine):

- **One seed lies.** Seed 0: uncertainty reached target in 20 labels vs random 84 ("4.2x better"). Seed 6: random 76 vs uncertainty 156 ("2.1x better" the other way). Across 30 seeds uncertainty "wins" 14, random 13, ties 3.
- **Thirty seeds: indistinguishable.** Uncertainty vs random: mean paired difference -5.9 labels, 95% bootstrap CI [-24.3, +12.8], sign test p = 1.000. Core-set vs random: -14.7 labels, CI [-28.5, +1.3], p = 0.108. The guard prints "indistinguishable at 30 seeds" for both and refuses any claim at 5 seeds.
- **Median labels to target:** random 76 (IQR 44-84, 2/30 censored), uncertainty 52 (36-92, 1/30), core-set 48 (28-68, 3/30). The medians look like a 30% saving; the paired test says it is not resolvable.
- **Power:** paired-difference sd is 52.9 labels (uncertainty) and 41.9 (core-set); 30 seeds can only detect about 27 and 21 labels respectively, and a 10-label effect would need 220 and 138 seeds.
- **Budget:** labels that reached target on 90% of seeds: core-set 115 (1,035 at 90/hour and an assumed 6 min/label), random 118 (1,062), uncertainty 134 (1,206). A strategy that misses target on more than 10% of seeds gets "buy more than the budget" instead of a number.
- **The guard detects real effects:** a deliberately bad strategy (label the most confident points) is flagged worse than random at 10 seeds in the test suite, while margin and uncertainty (provably identical for binary problems) produce zero paired difference and are never separated. On planted-null simulations (200 trials, 30 seeds each) the guard's false-positive rate stays under 5% + 3 SE, while a single-seed comparison names a "winner" more than 80% of the time.

### Limitations

- Synthetic Gaussian data with a linear boundary, not clinical text; the regime where active learning helps (e.g. heavy imbalance, `regime="imbalanced"` in `src/dataset.py`) is generated but not in the timed demo.
- Small scale (pool 300, 12 features) so that 30 seeds finish in about a minute of pure Python; effect sizes will differ on real embeddings.
- Runs that never reach target are censored at budget + batch, which is a lower bound; censor counts are printed next to every median, and the 90% budget quantile can include those censored values.
- Density and margin strategies are implemented and tested but left out of the demo for time; margin is identical to uncertainty in the binary case.
- The 6 minutes per label cost is an assumption exposed as a parameter, not a measurement.

## Interview questions this project answers

- **Why is active learning often no better than random?**
- **How many seeds do you need before believing a benchmark?**
- **How would you budget a labelling programme?**

## What this deliberately is *not*

- Not an annotation tool. It tells you how much annotation to buy.


## Run it now

```bash
python -m unittest discover -s tests -v   # the suite
python -m src.demo                        # the 60-second artefact
```

Requires Python 3.11+. The runnable core uses **only the standard
library** (including `sqlite3`), so there is nothing to install.

## Getting started

```bash
cd labelbudget
python -m src.demo                          # 30-seed benchmark + guard + budget report
python -m unittest discover -s tests -v     # 25 tests
```

Docker is supported but optional — every path above works on a plain
Windows/macOS/Linux laptop without a cloud account.

## Definition of done

- [ ] The differentiator above is implemented, and a test proves it
- [ ] The measurable outcome is produced by a command anyone can run
- [ ] `README` explains the one decision a generic version gets wrong
- [ ] CI runs the full suite on every push and is green on `main`
- [ ] A recruiter can see the headline artefact in under 60 seconds

## Licence

MIT — see [LICENSE](LICENSE).
