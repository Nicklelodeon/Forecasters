import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

data = {}

df = pd.read_csv("./src/Historical Product Demand.csv")
df[['Year', 'Month', 'Day']] = df.Date.str.split("/", expand = True)
bo = df.groupby(['Product_Code']).count()["Warehouse"]>1000
index = bo[bo == True].index
final_df = df[df['Product_Code'].isin(index)]
final_df['Order_Demand'] = final_df['Order_Demand'].apply(lambda x: x.replace('(','').replace(')',''))
final_df['Order_Demand'] = final_df['Order_Demand'].astype('int')
#print(final_df)
new_df = final_df.groupby(['Product_Code', 'Year', 'Month']).agg({'Order_Demand': 'sum'})
print(type(new_df['Order_Demand']['Product_0011']))
sli = new_df['Order_Demand']['Product_0011']
sli = sli.to_frame()
sli.to_csv("/Users/nicholas/Documents/Misc/internship A*STAR/Work/test.csv")

new_sli = pd.read_csv("/Users/nicholas/Documents/Misc/internship A*STAR/Work/test.csv")
print(new_sli)
sns.relplot(kind="line", x="Month", y="Order_Demand", hue="Year", data=new_sli)
plt.show()


# sns.relplot(x="Month", y)

# print(new_df[new_df['Product_Code'] == 'Product_0011'])
# new_df = new_df.groupby(['Product_Code', 'Year'])['Order_Demand'].apply(list)
# new_df.to_csv("/Users/nicholas/Documents/Misc/internship A*STAR/Work/data.csv")











