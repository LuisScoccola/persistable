import sys
sys.path.append("../")

from persistable import Persistable, PersistableInteractive
import numpy as np


import pickle
with open("example.pckl", "rb") as handle:
    X = pickle.load(handle)


p = Persistable(X)
pi = PersistableInteractive(p, jupyter = False, debug=True)
