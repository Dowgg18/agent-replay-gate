import pathlib
import sys

# The demo package lives at the repo root, next to src/. Tests import it as the
# system under test.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
