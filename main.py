#%%
import quandl 
import numpy as np 
import pandas as pd
# %%
auth_tok="cVfyYytxe6FXZijAcN8D"
df = quandl.get("WIKI/AAPL", trim_start = "2010-12-12", trim_end = "2020-12-12")


# %%
df.head()
# %%
df.describe()
# %%
df.info()
# %%
data = pd.DataFrame(df, columns=['Adj. Close', 'Adj. Volume', 'PCT_Change', 'HL_PCT'])
# %%
data.isnull().any()
# %%
data.fillna("-99999")
# %%
import math
# %%
y = df["Adj. Close"]
x = df.drop(["Adj. Close"], axis=1)
# %%
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
# %%
y.shape()
# %%
y
# %%
type(y)
# %%
y = y.values
# %%
y = y.reshape((-1, 1)) 
# %%
y
# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# %%
x_train.head()
# %%
lr = LinearRegression()
# %%
lr.fit(x_train, y_train)
# %%
pred = lr.predict(x_test)
# %%
from sklearn import metrics
# %%
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred))
# %%
