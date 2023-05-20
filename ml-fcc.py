import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

df = pd.read_csv("sample-data/magic04.data", names=cols)

# convert g & h to 1 and 0
# g stands for gamma
# h stands for hadron
df["class"] = (df["class"] == "g").astype(int)

print(df.head())