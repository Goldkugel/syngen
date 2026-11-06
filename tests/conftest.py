"""
Runs the tests in a temporary copy of the repository's configuration, so
that the logs and outputs of the scripts go there, and with experiment 0 on
the command line (config.py reads sys.argv when it is imported).
"""

import os
import shutil
import sys
import tempfile

repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

workingDirectory = tempfile.mkdtemp(prefix = "syngen_tests_")
shutil.copytree(os.path.join(repository, "config"), os.path.join(workingDirectory, "config"))
shutil.copytree(os.path.join(repository, "prompts"), os.path.join(workingDirectory, "prompts"))
os.chdir(workingDirectory)

sys.path.insert(0, os.path.join(repository, "src"))
sys.argv = [sys.argv[0], "0"]
