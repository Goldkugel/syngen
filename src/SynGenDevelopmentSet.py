"""
Draws the development set (hpo.development.csv), the evaluation set
(hpo.evaluation.csv) and the few-shot examples (hpo.fewshot.json) from
hpo.csv, see writeDevelopmentSet() in SynGenUtils.py. The preparation run
draws them only when missing or outdated; this script draws them in any case,
e.g. after changing the seed.

    python3 ./src/SynGenDevelopmentSet.py
"""

import sys

sys.dont_write_bytecode = True

from SynGenUtils  import *
from config       import *

from logger       import Logger, lockedFile

l = Logger(config = configFile)

def main() -> None:
    """
    Draws the development and evaluation set and the few-shot examples.
    """
    startRun(experimentRequired = False)
    l.printHeader("Drawing the Development and Evaluation Set")

    # Under the lock of the preparation runs, which draw the sets as well.
    with lockedFile(outputFileTransformedDevelopment):
        with l.step("Drawing the development and evaluation set"):
            writeDevelopmentSet()


if __name__ == "__main__":
    main()
