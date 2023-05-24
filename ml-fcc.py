import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

df = pd.read_csv("sample-data/magic04.data", names=cols)

# convert g & h to 1 and 0
# g stands for gamma
# h stands for hadron
df["class"] = (df["class"] == "g").astype(int)


for label in cols[:-1]:
    # separate the information into the ones that are gamma or hadron
    plt.hist(df[df["class"] == 1][label], color="blue", label="gamma", alpha=0.7, density=True)
    plt.hist(df[df["class"] == 0][label], color="red", label="hadron", alpha=0.7, density=True)
    # title of the histogram
    plt.title(label)
    # y label
    plt.ylabel("Probability")
    # x label with the names equal to the current columns
    plt.xlabel(label)
    plt.legend()
    plt.show()


# traing, validate, test datasets

"""""
    
    df.sample(frac=1): randomly shuffle all the rows
    frac=1: specifies that we want the entire dataset
    len(df): gets the total number of rows in the DataFrame 'df'
    0.6 * len(df): calculates 60% of the rows desired for the training set
    0.8 * len(df): calculates 80% of the rows desired for both training and validing set

"""""

train, valid, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])


# scaling our dataset
def scale_dataset(dataframe):
    # extracts the features from the dataframe, assuming the features are in all columns except for the last one
    x = dataframe[dataframe.cols[:-1]].values
    # assuming that the target value is in the last column
    y = dataframe[dataframe.cols[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # horizontally stacks the scales features 'x' and the target variable 'y'
    data = np.hstack((x, np.reshape(y, (-1, 1))))

    return data, x, y


