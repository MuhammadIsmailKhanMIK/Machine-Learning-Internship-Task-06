import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Q.1: Import data and check null values, check column info and the descriptive statistics of the data.
# Import data
tips = pd.read_csv('tips.csv')

# Check null values
print("Null values in each column:")
print(tips.isnull().sum())
print("\n")

# Check column info
print("Column information:")
print(tips.info())
print("\n")

# Descriptive statistics
print("Descriptive statistics:")
print(tips.describe())
print("\n")

# Q.2: Have a look at the tips given to the waiters according to:
# • the total bill paid
# • number of people at a table
# • and the day of the week

# Plot tips vs total_bill
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_bill', y='tip', data=tips)
plt.title('Tips vs Total Bill')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()

# Plot tips vs size
plt.figure(figsize=(10, 6))
sns.scatterplot(x='size', y='tip', data=tips)
plt.title('Tips vs Size')
plt.xlabel('Size of Party')
plt.ylabel('Tip')
plt.show()

# Plot tips vs day
plt.figure(figsize=(10, 6))
sns.boxplot(x='day', y='tip', data=tips)
plt.title('Tips vs Day')
plt.xlabel('Day of the Week')
plt.ylabel('Tip')
plt.show()

# Q.3: Have a look at the tips given to the waiters according to:
# • the total bill paid
# • the number of people at a table
# • and the gender of the person paying the bill

# Plot tips vs total_bill, size, and sex
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_bill', y='tip', data=tips, hue='sex')
plt.title('Tips vs Total Bill and Sex')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()

# Q.4: Have a look at the tips given to the waiters according to:
# • the total bill paid
# • the number of people at a table
# • and the time of the meal

# Plot tips vs total_bill and time
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_bill', y='tip', data=tips, hue='time')
plt.title('Tips vs Total Bill and Time')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()

# Q.5: Now check the tips given to the waiters according to the days to find out which day the most tips are given to the waiters:

# Pie chart of tips vs day
tips_by_day = tips.groupby('day')['tip'].sum()
plt.figure(figsize=(6, 6))
plt.pie(tips_by_day.values, labels=tips_by_day.index, autopct='%1.1f%%', startangle=90)
plt.title('Tips by Day')
plt.show()

# Q.6: Look at the number of tips given to waiters by gender of the person paying the bill to see who tips waiters the most:

# Pie chart of tips vs sex
tips_by_sex = tips.groupby('sex')['tip'].sum()
plt.figure(figsize=(6, 6))
plt.pie(tips_by_sex.values, labels=tips_by_sex.index, autopct='%1.1f%%', startangle=90)
plt.title('Tips by Sex')
plt.show()

# Q.7: Now check the tips given to the waiters according to the days to find out which day the most tips are given to the waiters:

# This is already done in Q.5, so no need to repeat.

# Q.8: Let's see if a smoker tips more or a non-smoker:

# Pie chart of tips vs smoker
tips_by_smoker = tips.groupby('smoker')['tip'].sum()
plt.figure(figsize=(6, 6))
plt.pie(tips_by_smoker.values, labels=tips_by_smoker.index, autopct='%1.1f%%', startangle=90)
plt.title('Tips by Smoker')
plt.show()

# Q.9: Now let's see if most tips are given during lunch or dinner:

# Pie chart of tips vs time
tips_by_time = tips.groupby('time')['tip'].sum()
plt.figure(figsize=(6, 6))
plt.pie(tips_by_time.values, labels=tips_by_time.index, autopct='%1.1f%%', startangle=90)
plt.title('Tips by Time')
plt.show()

# Q.10: Before training a waiter tips prediction model, do some data transformation by transforming the categorical values into numerical values:

# Data transformation
tips['sex'] = tips['sex'].map({'Male': 1, 'Female': 0})
tips['smoker'] = tips['smoker'].map({'Yes': 1, 'No': 0})
tips['day'] = tips['day'].map({'Sun': 0, 'Sat': 1, 'Thur': 2, 'Fri': 3})
tips['time'] = tips['time'].map({'Lunch': 0, 'Dinner': 1})

# Q.11: Now split the data into training and test sets. Then train a machine learning model (Linear Regression) for the task of waiter tips prediction.

# Split data into training and test sets
X = tips[['total_bill', 'sex', 'smoker', 'day', 'time', 'size']]
y = tips['tip']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Q.12: Check your model prediction. It should show the following output by this input: {total_bill:24.50, "sex":1, "smoker":0, "day":0, "time":1, "size":4}

# Make prediction
input_data = pd.DataFrame([[24.50, 1, 0, 0, 1, 4]], columns=['total_bill', 'sex', 'smoker', 'day', 'time', 'size'])
prediction = model.predict(input_data)
print(f'Prediction for input {input_data.values}: {prediction}')

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean squared error: {mse}')

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred - y_test)
plt.xlabel('Actual Tips')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()