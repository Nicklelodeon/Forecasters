import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import datetime


df = pd.read_csv("./src/TOTALSA.csv")
# create graph for general trend (number of cars sold against years)
new_lst = []
for x in df['DATE']:
    i, j, k = x.split('-')
    new_lst.append(datetime.datetime(int(i), int(j), int(k)))

df['new_date'] = new_lst
plt.plot('new_date', 'TOTALSA', data=df)
plt.xlabel('Year')
plt.ylabel('Number of cars sold (millions)')
plt.title('Number of cars sold against time')
plt.show()

print(df['TOTALSA'].mean())
print(df['TOTALSA'].std())

# create graph for yearly trend of cars sold
years = [int(x[:4]) for x in df['DATE']]
months = [int(x[5:7]) for x in df['DATE']]
df['Year'] = years
df['Month'] = months

count1 = 1
for i in range(1976, 2024):
    if i == 1976:
        plt.title('Distribution of monthly demand over 48 years')
    plt.yticks([])
    new_df = df[df['Year'] == i]
    # print(df['Year'] == i)
    # print(df[df['Year'] == i])
    plt.subplot(10, 5, count1)
    plt.plot('Month', 'TOTALSA', data=new_df)
    count1 += 1
    plt.xticks(range(1, 13, 2))
plt.yticks([])

plt.show()



















