# -*- coding: utf-8 -*-
"""
Created on Apr 13 10:25:23 2024


@author: N Ouben
"""

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime


# Load the data
baby_center_data = pd.read_csv('./preprocessed_baby_center.csv')
reddit_data = pd.read_csv('./preprocessed_reddit.csv')


#Keywords' dictionnaries
alcohol_keywords = ['alcohol', 'drinking', 'wine', 'beer', 'liquor', 'vodka', 'whiskey', 'rum', 'champagne', 'cocktail']
smoking_keywords = ['tobacco', 'smoking', 'cigarette', 'nicotine', 'cigar', 'e-cigarette', 'vaping', 'smoke', 'hookah', 'pipe']
bmi_keywords = ['obese', 'BMI','bmi', 'body mass index', 'overweight', 'weight', 'fat', 'diet', 'nutrition', 'calories', 'exercise']
health_problem_keywords = ['problem','problems', 'consequences', 'harm', 'dangerous', 'unhealthy', 'issue', 'illness', 'disease', 'condition', 'adhd','fdas','hyperactivity','eclampsia','preterm','lead','seizure']


# Detect the keywords in the comments
def contains_keywords(text, keywords):
    '''
    Parameters
    ----------
    text : string
        comments' textual content.
    keywords : string
        keywords from the keyword dictionnaries.

    Returns
    -------
    bool, string
        keyword present in the comment.
    '''
    
    if not isinstance(text, str):
        return False
    return any(keyword in text for keyword in keywords)


# Identify the column containing the comment with the keyword
comment_column = 'Processed Comment'  # Adjust according to the shown column
original_comment_column = 'Comment'  # Show the column with the entire comments


# Convert the values of the comments into strings
baby_center_data[comment_column] = baby_center_data[comment_column].astype(str)
reddit_data[comment_column] = reddit_data[comment_column].astype(str)


# Add columns for indicators like: alcohol, smoking, BMI, and child heath issues
baby_center_data['alcohol'] = baby_center_data[comment_column].apply(lambda x: 1 if contains_keywords(x, alcohol_keywords) else 0)
baby_center_data['smoking'] = baby_center_data[comment_column].apply(lambda x: 1 if contains_keywords(x, smoking_keywords) else 0)
baby_center_data['BMI'] = baby_center_data[comment_column].apply(lambda x: 1 if contains_keywords(x, bmi_keywords) else 0)
baby_center_data['child_health'] = baby_center_data[comment_column].apply(lambda x: 1 if contains_keywords(x, health_problem_keywords) else 0)

reddit_data['alcohol'] = reddit_data[comment_column].apply(lambda x: 1 if contains_keywords(x, alcohol_keywords) else 0)
reddit_data['smoking'] = reddit_data[comment_column].apply(lambda x: 1 if contains_keywords(x, smoking_keywords) else 0)
reddit_data['BMI'] = reddit_data[comment_column].apply(lambda x: 1 if contains_keywords(x, bmi_keywords) else 0)
reddit_data['child_health'] = reddit_data[comment_column].apply(lambda x: 1 if contains_keywords(x, health_problem_keywords) else 0)


# Combine the datasets
data = pd.concat([baby_center_data, reddit_data])


# elect the indicators and the target
X = data[['alcohol', 'smoking', 'BMI']]
y = data['child_health']


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Build the model
model = Sequential()
model.add(Input(shape=(X_train_scaled.shape[1],)))  # Use Input to specify the shape of input data
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # 'sigmoid' for a binary classification


# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Preset TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


# Train the model on the train set
# try different epochs
    
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
#history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=10,
 #                   batch_size=32, callbacks=[early_stopping, tensorboard_callback])
#history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50,
 #                   batch_size=32, callbacks=[early_stopping, tensorboard_callback])
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100,
                    batch_size=32, callbacks=[early_stopping, tensorboard_callback])


# Evaluate the model by calculating loss and accuracy
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')


# Predictions on test data
y_pred = (model.predict(X_test_scaled) > 0.6).astype("int32")


# Extra metrics for a better evaluation
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\nExtra metrics:")
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
print(f'Accuracy: {accuracy}')


# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# Show some predictions on a batch of comments from the test set
sample_indices = X_test.index[:50]  # increase the number of shown comments
sample_comments = data.loc[sample_indices, original_comment_column]
sample_predictions = y_pred[:50]


print("\nPredictions on a few comments:")
print("Predicted as 'No Problem':")
for comment, prediction in zip(sample_comments, sample_predictions):
    if prediction[0] == 0:
        print(f"Comment: {comment}")
        print(f"Prediction (0 = No Problem, 1 = Problem): {prediction[0]}")
        print("-" * 50)

print("\nPredicted as 'Problem':")
for comment, prediction in zip(sample_comments, sample_predictions):
    if prediction[0] == 1:
        print(f"Comment: {comment}")
        print(f"Prediction (0 = No Problem, 1 = Problem): {prediction[0]}")
        print("-" * 50)