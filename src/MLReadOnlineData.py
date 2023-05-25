import pandas as pd 
import numpy as np 

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
new_df = new_df.groupby(['Product_Code', 'Year'])['Order_Demand'].apply(list)
new_df.to_csv("/Users/nicholas/Documents/Misc/internship A*STAR/Work/data.csv")









