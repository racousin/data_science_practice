# Present so pytest puts draft/ on sys.path: tests import `scoring` and the
# `tasks` package the way build_dataset.py does, and `env` from the package
# root: the env.py the platform runs, not a copy.
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
