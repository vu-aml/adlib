# Data Poisoning Tests

This README is designed to give a quick overview about data poisoning (DP)
adversaries, learners, and tests in the `adlib` library. The available
DP adversaries are `label-flipping`, `k-insertion`, `data-modification`, and
`data-transform`. The available DP learners are `trim`, `atrim`, `irl`, and
`orl`. There is a testing suite defined in
`adlib/tests/learners/dp_learner_test.py`. In this file, the class
`TestDataPoisoningLearner` is used to provide a contained testing environment
that targets by default a ~20% poisoning percentage. To see usage, see the
aforementioned file. If you want to run many tests of all learners with a
specified attacker, you should use
`adlib/tests/learners/dp_learner_many_test.py`. For example, to run 50
`label-flipping` attacks and get detailed output of how every learner performs,
do `python3 adlib/tests/learners/dp_learner_many_test.py 50 label-flipping`.
This will output 50 files with the resulting tests in verbose mode.
