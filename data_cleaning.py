import pandas as pd
import numpy as np

df = pd.read_csv(
    r"D:\jyestha project\1st_assignment\Listings.csv",
    encoding='latin1',
    low_memory=False
)

# Fix text
df['name'] = df['name'].astype(str)

# Fill missing values
df['name'] = df['name'].fillna('Unknown')
df['host_location'] = df['host_location'].fillna('Unknown')

df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())
df['review_scores_rating'] = df['review_scores_rating'].fillna(df['review_scores_rating'].median())

# Fill remaining review columns
cols = [
    'review_scores_accuracy',
    'review_scores_cleanliness',
    'review_scores_checkin',
    'review_scores_communication',
    'review_scores_location',
    'review_scores_value'
]

for col in cols:
    df[col] = df[col].fillna(df[col].median())

# Convert types
df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')

df['instant_bookable'] = df['instant_bookable'].map({'t': 1, 'f': 0})
df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0})
df['host_has_profile_pic'] = df['host_has_profile_pic'].map({'t': 1, 'f': 0})
df['host_identity_verified'] = df['host_identity_verified'].map({'t': 1, 'f': 0})

# Drop useless column
df.drop(columns=['district'], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle outliers
df = df[df['price'] < 1000]
df = df[df['bedrooms'] > 0]

# Feature engineering
df['price_per_person'] = df['price'] / df['accommodates']

# Save
df.to_csv("cleaned_airbnb.csv", index=False)

print("Data cleaning completed ✅")