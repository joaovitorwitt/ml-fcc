import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cols = ["fLength", "fWidth" "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

file = pd.read_csv("sample-data/magic04.data")
print(file)