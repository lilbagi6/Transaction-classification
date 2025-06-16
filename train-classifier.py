#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.layers import BatchNormalization


# In[2]:


# Load train data
file_path = "4000_train.csv"
df_raw = pd.read_csv(file_path)

df_raw.head()


# In[3]:


# Drop unnecessary columns and clean the data
df_raw.drop(columns=["Unnamed: 0", "Transaction Number"], inplace=True)
df_raw.dropna(subset=["Category"], inplace=True)

df_raw["Transaction Date"] = pd.to_datetime(df_raw["Transaction Date"], format="%d/%m/%Y")
df_raw["weekday"] = df_raw["Transaction Date"].dt.day_name()
df_raw["Transaction Description"] = df_raw["Transaction Description"].str.lower().str.replace(r"[^a-z0-9\s]", "", regex=True)

# Create a new DataFrame
df = df_raw[["Transaction Description", "weekday", "Amount", "Category"]]
df.head(10)


# In[4]:


# Data preprocessing
## Text data (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_text = vectorizer.fit_transform(df["Transaction Description"]).toarray()

## Categorical data (One-hot encoding)
ohe = OneHotEncoder(sparse_output=False)
X_weekday = ohe.fit_transform(df[["weekday"]])

## Num data (Scaler)
scaler = StandardScaler()
X_balance = scaler.fit_transform(df[["Amount"]])

# Concatenate all features
X = np.hstack((X_text, X_weekday, X_balance))

# Label encoding
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["Category"].str.split(',')) 


# In[5]:


# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


# Build the model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(y.shape[1], activation='sigmoid')  # Мulti-label → sigmoid
])


# In[13]:


# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)


# In[15]:


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# In[17]:


# Load final test dataset
new_file_path = "Rest_Test.csv"
df_new = pd.read_csv(new_file_path)

df_new["Transaction Description"] = df_new["Transaction Description"].str.lower().str.replace(r"[^a-z0-9\s]", "", regex=True)
df_new["weekday"] = pd.to_datetime(df_new["Transaction Date"], format="%d/%m/%Y").dt.day_name()

# Transform new data
X_new_text = vectorizer.transform(df_new["Transaction Description"]).toarray()
X_new_weekday = ohe.transform(df_new[["weekday"]])
X_new_balance = scaler.transform(df_new[["Amount"]])

# Combine new data
X_new = np.hstack((X_new_text, X_new_weekday, X_new_balance))


# In[19]:


# Final predictions and probabilities
predictions_new = model.predict(X_new)
probabilities_new = np.max(predictions_new, axis=1) * 100  
y_pred_labels_new = np.argmax(predictions_new, axis=1)


# In[21]:


# Final dataframe with predictions and probabilities
df_new["Predicted Category"] = mlb.classes_[y_pred_labels_new]
df_new["Prediction Probability (%)"] = probabilities_new

df_new.head(20)


# In[25]:


df_new.to_csv('predictions.csv', index=False)

