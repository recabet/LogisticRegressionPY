import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticReg import my_LogisticRegression

df = pd.read_csv("exams.csv")
df = df.sample(len(df)).copy()
df_passed = df[df["admitted"] == 1]
df_failed = df[df["admitted"] == 0]
df.head()

plt.scatter(df_passed["exam_1"], df_passed["exam_2"], color='g')
plt.scatter(df_failed["exam_1"], df_failed["exam_2"], color='r')


plt.show()

df2 = df.copy()

weights = np.random.rand(df.shape[1])
y = df["admitted"]
y_train = df2["admitted"][:90]
y_test = df2["admitted"][90:]
df_train = df2[:90].copy()
df_test = df2[90:].copy()
df_train["exam_1"] = (df_train["exam_1"] - df_train["exam_1"].min()) / (
		df_train["exam_1"].max() - df_train["exam_1"].min())
df_train["exam_2"] = (df_train["exam_2"] - df_train["exam_2"].min()) / (
		df_train["exam_2"].max() - df_train["exam_2"].min())
df_test["exam_1"] = (df_test["exam_1"] - df2[:90]["exam_1"].min()) / (
		df2[:90]["exam_1"].max() - df2[:90]["exam_1"].min())
df_test["exam_2"] = (df_test["exam_2"] - df2[:90]["exam_2"].min()) / (
		df2[:90]["exam_2"].max() - df2[:90]["exam_2"].min())
inputs = np.array([np.ones(len(df_train)), df_train["exam_1"], df_train["exam_2"]]).T

model=my_LogisticRegression()
w = model.fit(df_train, y_train)

print(y_test)
print(model.predict(df_test))