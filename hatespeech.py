import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# Load the dataset
data = pd.read_csv("C:\\Users\\samri\\OneDrive\\Desktop\\Tweets.csv")
# printing all columns of the dataframe
#print(data.columns.tolist())
#print(data.head())
# Split the dataset into training and test sets
train_data = data.sample(frac=0.8, random_state=1)
test_data = data.drop(train_data.index)
# Evaluate the model
# Vectorize the text data
vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(train_data["text"])
test_vectors = vectorizer.transform(test_data["airline_sentiment"])
# Train the model
clf = MultinomialNB()
clf.fit(train_vectors, train_data["airline_sentiment"])
predictions=clf.predict(test_vectors)
print("How much accuracy of dataset to predict hatespeech:-")
print("Accuracy: ",accuracy_score(test_data["airline_sentiment"],predictions))
plt.pie(data['airline_sentiment'].value_counts().values,
		labels = data['airline_sentiment'].value_counts().index,
		autopct='%1.1f%%')
plt.show()
text=input("ENTER:-  ")
result = clf.predict(vectorizer.transform([text]))
print("Input:", text)
print("Output:", result)
