# Progress log

Picked an inventory replenishment problem rather than dynamic pricing. The
demand side is calibrated to a real SKU, StockCode 85048 in the Online
Retail II data already in this portfolio, "15CM CHRISTMAS GLASS BALL 20
LIGHTS". It is basically zero demand most of the year and spikes hard
around Christmas, a peak seasonal multiplier near 25x baseline. That gave
me a genuine non stationary demand shape for free instead of inventing one.

Environment: periodic review, lead time 3 days, holding cost 0.05 per unit
per day, lost sale penalty 1.20 per unit, a fixed cost of 2.0 whenever an
order is placed, order quantity discretized to steps of 4 up to a max of
40. Observation is inventory on hand, days until the next inbound order
arrives, a log transformed seasonal multiplier, day of year fraction, and
days since a regime shift if one is active.

Heuristic: order up to level, s and S, tuned by grid search on the training
distribution before touching any learned policy. Landed on s=4, S=40.

First full run, 800 training episodes each for the bandit, tabular Q
learning, and REINFORCE. Heuristic won clearly, -1074 versus -1509 for the
bandit and around -1500 to -1600 for Q learning and REINFORCE. Bumped
training to 4000 episodes to make sure this was not just an undertrained
comparison.

At 4000 episodes something was clearly wrong, not just a fair loss. The
bandit and tabular Q learning produced the exact same mean reward and the
exact same confidence interval, to three decimal places. Two different
model classes do not coincidentally land on identical numbers unless they
are taking the identical action on every state along the evaluated
trajectory. REINFORCE got dramatically worse with more training, -128798,
worse than a policy that always orders the maximum amount.

Debugged in order:

First bug: the season feature fed into everything was the raw multiplier,
which for this SKU ranges from about 0.05 to 25. The tabular Q learning
bucketing formula assumed a roughly 0 to 3 range, so the entire Christmas
window from a mild uptick through the actual peak collapsed into a single
saturated bucket. The table could not tell a slightly busy day from the
single busiest day of the year, which is exactly the day ordering ahead of
time matters most. Fixed by feeding log1p of the multiplier into the
observation instead of the raw value, which compresses the range to about
0 to 3.3, and rescaled the bucketing divisor to match.

That fix alone did not change the bandit or Q learning result. Rolled out
the trained Q table policy through the actual Christmas window and it
placed a zero order every single day, all the way through the peak, while
the properly tuned heuristic keeps reordering. Tried three different
exploration schedules for Q learning, from the original epsilon floor of
0.05 up to 0.25 with far more training episodes, and got the same -1509
result each time to one decimal place. That ruled out undertrained
exploration as the explanation. Sanity checked the reward scale directly:
an always order zero policy scores about -1500, an always order maximum
policy scores about -128800, a uniform random policy scores about -63000.
Always ordering zero is a strong, easy to find local optimum given a fixed
cost of 2.0 charged on almost every one of 365 days if you order often,
against a real but comparatively rare Christmas stockout cost. Both the
bandit and tabular Q learning found that same local optimum and got stuck
there. This is a real finding about this reward structure and this state
representation, not a bug, and I am reporting it as the honest result for
those two methods rather than continuing to chase a different answer.

Second bug, this one a real implementation mistake: fed the raw
observation directly into the REINFORCE policy network. Inventory ranges
roughly 0 to 60 while every other feature sits in roughly 0 to 3.3. Checked
the freshly initialized, completely untrained network's output across a
few states and it already strongly favored the maximum order action
whenever inventory was high, purely from random initialization interacting
with the unnormalized scale, nothing learned yet. That is a feedback loop
waiting to happen: order more, inventory rises, the network becomes even
more confident in ordering the max again. Normalized the inventory feature
before it reaches the network. That alone slowed the collapse but did not
stop it.

Third bug, also a real mistake: the discounted return at the start of a
365 day episode and the return at day 360 are on completely different
scales just from having fewer future steps left to discount, regardless of
whether the policy is any good. A single scalar baseline, whether computed
per episode or as one running average across episodes, cannot represent
that shape, so it handed early timesteps in every episode a large
magnitude advantage and late timesteps a small one, independent of the
actual action taken. Since the environment's calendar is fixed across
training episodes here, same starting day of year every time, timestep t
means the same calendar day every episode, so a baseline indexed by
position in the episode is a cheap, honest stand in for a real value
function. Replaced the scalar baseline with one running average per
timestep position. Loss immediately became well behaved: -66611 in the
first 200 episodes, improving every subsequent chunk, converging to about
-425 by 5000 episodes and still slightly improving. Re-ran the full
pipeline at 5000 training episodes for all three learned methods once this
was in place.

Wrote a pytest suite covering the environment's cost accounting and lead
time mechanics, the heuristic's reorder logic, the bandit's update rule,
the Q table's discretization bounds and update direction, and the
REINFORCE agent's normalization and position baseline shape. One test
failed on first run for a reason unrelated to the code: it assumed the
seasonal multiplier stays constant across a six day window, which it does
not in real calibrated data, so the two states being compared were not
actually holding season fixed. Rewrote the test to compare two environment
instances at the identical calendar day instead. 24 of 24 pass.

Final result: REINFORCE beats the tuned heuristic in sample and in every
one of eight out of sample robustness conditions tested (demand up 10 and
25 percent, longer and shorter lead time, and regime shifts timed at three
different points and in both directions). The bandit and tabular Q
learning do not beat the heuristic in any condition tested. The advantage
that matters here is not full RL versus a memoryless bandit in the
abstract, both a bandit and a tabular Q learning agent landed in the same
degenerate policy, it is that a continuous function approximator found a
meaningfully different and better policy than a hand discretized state
space could represent. That is the finding I would not have predicted
going in and would not have found without debugging the two identical
looking failures separately instead of assuming the first explanation for
both.

Left undone: did not try an actor critic method with a real learned value
function, which is the more principled version of what the position
indexed baseline is approximating. Did not try increasing tabular Q
learning's state resolution further, for example splitting the top season
bucket further or adding a discretized day of year feature directly,
which might close some of its gap; I stopped once three different
exploration schedules gave the same answer and treated that as sufficient
evidence rather than open ended hyperparameter search.
