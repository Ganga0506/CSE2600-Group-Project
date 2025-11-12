import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/gangamahesh/Desktop/Fall25/cse 2600/CSE2600-Group-Project/data/Wildfire_Dataset.csv")

df.shape
df.info()
df.head()
df.describe()

df['Wildfire'].value_counts()
df['Wildfire'].value_counts(normalize=True)

# Class distribution
df['Wildfire'].value_counts().plot(kind='bar', color=['skyblue', 'seagreen'])
plt.title('Wildfire Occurrence')
plt.xlabel('Wildfire')
plt.ylabel('Count')
plt.show()

df.isna().sum()

# Convert Wildfire to binary for correlation
df['Wildfire_binary'] = df['Wildfire'].map({'No': 0, 'Yes': 1})
numeric_cols = df.select_dtypes(include='number').columns
corr = df[numeric_cols].corr()['Wildfire_binary'].sort_values(ascending=False)
print(corr)

# Compare variables for Wildfire vs Non-Wildfire
df_filtered = df[
    (df['tmmx'] < 1000) &
    (df['tmmn'] < 1000) &
    (df['pr'] < 50) &
    (df['vpd'] < 10000) &
    (df['rmin'] <= 100)
]

df_sample = df_filtered.sample(50000, random_state=42)

features = ['tmmx', 'tmmn', 'pr', 'vpd', 'rmin']

for f in features:
    sns.boxplot(x='Wildfire', y=f, data=df_sample)
    plt.title(f'{f} vs Wildfire Occurrence (Cleaned)')
    plt.show()

df['month'] = df['datetime'].dt.month
df.groupby('month')['Wildfire_binary'].mean().plot(kind='bar', color='coral')
plt.title('Average Wildfire Probability by Month')
plt.xlabel('Month')
plt.ylabel('Proportion of Fire Days')
plt.show()