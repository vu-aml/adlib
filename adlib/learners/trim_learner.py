# trim_learner.py
# A learner that implements the TRIM algorithm described in "Manipulating
# Machine Learning- Poisoning Attacks and Countermeasures for Regression
# Learning" found at https://arxiv.org/pdf/1804.00308.pdf.
# Matthew Sedam

from adlib.learners.learner import learner


class TRIM_Learner(learner):
    """
    A learner that implements the TRIM algorithm described in the paper
    mentioned above.
    """

    def __init__(self):
        learner.__init__(self)
        raise NotImplementedError

    
