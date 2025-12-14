import pandas as pd
df = pd.read_csv("dataset_contoh.csv", thousands=',', quotechar='"', engine='python')
print(df.dtypes)
print(df.head().to_string())