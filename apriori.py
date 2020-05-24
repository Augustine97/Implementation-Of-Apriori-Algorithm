from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import pandas as pd

dataset=[['Milk','Onion','Nutmeg','KidneyBeans','Eggs','Yogurt'],
['Dill','Onion','Nutmeg','KidneyBeans','Eggs','Yogurt'],
['Milk','Apple','KidneyBeans','Eggs'],
['Milk','Unicorn','Corn','KidneyBeans','Yogurt'],
['Corn','Onion','Onion','KidneyBeans','Ice-Cream','Eggs']]


te=TransactionEncoder()
Trans_array=te.fit(dataset).transform(dataset)
df=pd.DataFrame(Trans_array,columns=te.columns_)
print(df)
ap=apriori(df,min_support=0.6,use_colnames=True)
print(ap)
ap['Length']=ap['itemsets'].apply(lambda x:len(x))
print(ap)
print(ap[(ap['Length']==2) & (ap['support']>=0.8 )])
