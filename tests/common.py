import sys
import os
import pytest

TEST_NUMPY = True
try:
    import numpy
except ImportError:
    TEST_NUMPY = False

TEST_SCIPY = True
try:
    import scipy
except ImportError:
    TEST_SCIPY = False

TEST_SKLEARN = True
try:
    import sklearn
except ImportError:
    TEST_SKLEARN = False
