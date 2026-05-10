import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv(
    r"d:\jyestha project\cleaned_airbnb.csv",
    encoding='latin1',
    low_memory=False
)


# ------------------------------
# 1. PRICE DISTRIBUTION
# ------------------------------


plt.figure(figsize=(8,5))
plt.hist(df['price'], bins=30)

plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")

plt.show()

# ------------------------------
# 2. ROOM TYPE DISTRIBUTION
# ------------------------------

room_counts = df['room_type'].value_counts()

plt.figure(figsize=(8,5))
plt.bar(room_counts.index, room_counts.values)

plt.title("Room Type Distribution")
plt.xlabel("Room Type")
plt.ylabel("Count")

plt.xticks(rotation=10)

plt.show()

# ------------------------------
# 3. TOP 10 CITIES BY LISTINGS
# ------------------------------

top_cities = df['city'].value_counts().head(10)

plt.figure(figsize=(10,5))
plt.bar(top_cities.index, top_cities.values)

plt.title("Top 10 Cities by Listings")
plt.xlabel("City")
plt.ylabel("Number of Listings")

plt.xticks(rotation=45)

plt.show()

# ------------------------------
# 4. AVERAGE PRICE BY ROOM TYPE
# ------------------------------

avg_price = df.groupby('room_type')['price'].mean()

plt.figure(figsize=(8,5))
plt.bar(avg_price.index, avg_price.values)

plt.title("Average Price by Room Type")
plt.xlabel("Room Type")
plt.ylabel("Average Price")

plt.xticks(rotation=10)

plt.show()

# ------------------------------
# 5. REVIEW SCORE DISTRIBUTION
# ------------------------------

plt.figure(figsize=(8,5))
plt.hist(df['review_scores_rating'], bins=20)

plt.title("Review Score Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")

plt.show()

# ------------------------------
# 6. INSTANT BOOKING PIE CHART
# ------------------------------

booking = df['instant_bookable'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(
    booking.values,
    labels=['No', 'Yes'],
    autopct='%1.1f%%'
)

plt.title("Instant Booking Availability")

plt.show()

print("EDA Completed Successfully ✅")