import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logisticReg import LogisticRegression
from sklearn.model_selection import train_test_split

df=pd.read_csv("exams.csv")
df_passed=df[df["admitted"]==1]
df_failed=df[df["admitted"]==0]

plt.scatter(df_passed["exam_1"],df_passed["exam_2"],color='b')
plt.scatter(df_failed["exam_1"],df_failed["exam_2"],color='r')
plt.show()