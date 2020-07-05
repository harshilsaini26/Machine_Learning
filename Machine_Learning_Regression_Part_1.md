**Python Program for Linear Regression**
CASE STUDY TOPIC : Google Stocks
I know that stock prediction don't follow linear regression, but still it's for practise only.

Here we import the necessary libraries and some important functions.


```python
import pandas as pd
import csv
import math
import numpy as np
import time
from datetime import date, timedelta
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
```

Here we are importing the Data from our hard disk.


```python
#import you data 
df = pd.read_csv(' ', index_col='Date',parse_dates=True)
#print top 10 rows
df.head(10)
```

As seen from the last line of code, there are many columns and we not need so many columns. Therefore, we need to segregate the columns.


```python
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
print(df)
```

We need to predict the next values of the share, so we need to introduce some new vaules into our data.


```python
#'HL_PCT' is the percent differenct between the highest val of share and the closing value of the share for any respective time.
df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']) * 100
#'PCT_change' is the percent difference between the closing and the openening value of the share.
df['PCT_change'] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']) * 100
#the final data table now contains the following columns:
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
```

We now need a coulumn that hols the value for the prediction.


```python
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)
```

We now make a columns named 'label', at the last position.


```python
df['label'] = df[forecast_col].shift(-forecast_out)
```


```python
x = np.array(df.drop(['label'],1)) #an array 'x' without the 'label' column
x = preprocessing.scale(x) 
x_lately = x[-forecast_out:] 
x = x[:-forecast_out] #'x' array now has the same number of rowns as 'y'

df.dropna(inplace=True)

y = np.array(df['label']) #as 'label' coulmn will hold our final output, therefore y is given the 'label' coulmn.
y = np.array(df['label'])
```


```python
print(x.shape)
print(y.shape)
```

Here we split the data into training and testing.


```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```

Linear Regeression model training is done next.


```python
first_model = LinearRegression(n_jobs = -1) #'n_jobs' will divide the model into maximum number of parts and all parts run simultaneously. 
first_model.fit(x_train,y_train)
```

We use the pickle library to save our trained model so as to test different data's without training again and again.


```python
with open('linearregression.pickle','wb') as f:
    pickle.dump(first_model, f)

pickle_in = open('linearregression.pickle','rb')
first_model = pickel.load(pickel_in)
```


```python
accuracy = first_model.score(x_test, y_test)
print(accuracy)
```

Now we need to test our model and predict the value for the next number of days as required.


```python
forecast_set = first_model.predict(x_lately)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan
```

To make a graph incorporating the dates at x-axis and the corresponding share value at y-axis, we need to use the time stamp methodology.

This method is not the best one but can be used. A much more simpler appoach is using the pandas timestamp function.


```python
last_date = df.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
```

Now we plot the data. XD


```python
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
```
