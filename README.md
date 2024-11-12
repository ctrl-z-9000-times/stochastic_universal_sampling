# Stochastic Universal Sampling

This package implements the stochastic universal sampling (SUS) algorithm for
the rand crate. The SUS algorithm is essentially a random selection algorithm.
SUS guarantees that highly-weighted samples will not dominate the selection
beyond their proportional weight. This is useful for evolutionary algorithms.
For more information see:
<https://en.wikipedia.org/wiki/Stochastic_universal_sampling>
