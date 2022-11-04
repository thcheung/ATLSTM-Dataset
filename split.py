import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

df = pd.read_csv('./full.csv')
#df = df.drop_duplicates(subset=['sentence'],keep='first')
#df = df[df.sentence != None]
"""
df_majority = df[df.label=='Positive']
df_minority = df[df.label=='Negative']

# Downsample majority class
df_majority_upsampled = resample(df_majority, 
                                 replace=False,     # sample with replacement
                                 n_samples=15000,    # to match majority class
                                 random_state=42) # reproducible results

# Combine majority class with upsampled minority class
df_downsampled = pd.concat([df_majority_upsampled, df_minority])


# Display new class counts
print(df_downsampled.label.value_counts())

train, test = train_test_split(df_downsampled, train_size = 0.8)

train.to_csv('./dataset/train.csv',index=False)


dev, test = train_test_split(test, train_size = 0.50)
dev.to_csv('./dataset/dev.csv',index=False)
test.to_csv('./dataset/test.csv',index=False)

"""
train, test = train_test_split(df, train_size=0.8)
train.to_csv('./train.csv', index=False)
dev, test = train_test_split(test, train_size=0.50)
dev.to_csv('./val.csv', index=False)
test.to_csv('./test.csv', index=False)
