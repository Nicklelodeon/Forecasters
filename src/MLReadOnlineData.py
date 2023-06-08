import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import datetime


# data = {}

# df = pd.read_csv("./src/Historical Product Demand.csv")
# df[['Year', 'Month', 'Day']] = df.Date.str.split("/", expand = True)
# bo = df.groupby(['Product_Code']).count()["Warehouse"]>1000
# index = bo[bo == True].index
# final_df = df[df['Product_Code'].isin(index)]
# final_df['Order_Demand'] = final_df['Order_Demand'].apply(lambda x: x.replace('(','').replace(')',''))
# final_df['Order_Demand'] = final_df['Order_Demand'].astype('int')
# #print(final_df)
# new_df = final_df.groupby(['Product_Code', 'Year', 'Month']).agg({'Order_Demand': 'sum'})
# print(type(new_df['Order_Demand']['Product_0011']))
# sli = new_df['Order_Demand']['Product_0011']
# sli = sli.to_frame()
# sli.to_csv("/Users/nicholas/Documents/Misc/internship A*STAR/Work/test.csv")

# new_sli = pd.read_csv("/Users/nicholas/Documents/Misc/internship A*STAR/Work/test.csv")
# print(new_sli)
# sns.relplot(kind="line", x="Month", y="Order_Demand", hue="Year", data=new_sli)
# plt.show()


df = pd.read_csv("./src/norway_new_car_sales_by_month.csv")

# store = {}

# for i in range(len(df) - 1):
#     x = df.iloc[i]
#     if x['Year'] in store:
#         store[x['Year']].append(x['Quantity'])
#     else:
#         store[x['Year']] = [x['Quantity']]

# new_df = pd.DataFrame.from_dict(store)
# new_df.to_csv("/Users/nicholas/Documents/Misc/internship A*STAR/Work/cleaned_car_data.csv")



# visualise data spread
new_lst = []
for i in range(df.shape[0]):
    x = df.iloc[i]
    new_lst.append(datetime.datetime(int(x['Year']), int(x['Month']), 1) )

df['date'] = new_lst

plt.plot('date', 'Quantity', data=df)
plt.show()
# print(df)
# count1 = 1
# for i in range(2007, 2017):
#     plt.yticks([])
#     new_df = df[df['Year'] == i]
#     plt.subplot(2, 5, count1)
#     plt.plot('Month', 'Quantity', data=new_df)
#     count1 += 1
#     plt.xticks(range(1, 13, 2))
# plt.yticks([])
# plt.show()

















