import pandas as pd

df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
#print(df.index)
#print(df.dtypes)
#print(df.describe())
#print(df['median_income'].value_counts(normalize= True))
#print(df.loc[:5,['median_income']])
print(df['median_income'].info())

