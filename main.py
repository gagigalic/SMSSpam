import nltk
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords

#remove any trailing whitespace characters
messages = [line.rstrip() for line in open("SMSSpamCollection")]

for mess_no, messages in enumerate(messages[:10]):
    print(mess_no, messages)

import pandas as pd
messages = pd.read_csv("SMSSpamCollection", sep = "\t", names = ["label", "message"])
print(messages.head())

messages.describe()
messages.groupby("label").describe()

#new column lenght
messages["lenght"] = messages["message"].apply(len)
print(messages.head())

#distribution of lenght
messages["lenght"].plot.hist(bins = 50)
plt.savefig("lenght.png")
plt.close()

#the longest message
longest =messages[messages["lenght"] == 910]["message"].iloc[0]
print(longest)

#ham vs spam
messages.hist(column="lenght", by = "label", bins = 60, figsize=(12,4))
plt.savefig("plot.png")
plt.close()


def text_process(mess):
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#clean version of words in list
print(messages["message"].head(5).apply(text_process))

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

pip = Pipeline([
    ('bow', CountVectorizer()),  # CountVectorizer for text to word count
    ('tfidf', TfidfTransformer()),  # TF-IDF transformer
    ('model', MultinomialNB())  # Multinomial Naive Bayes classifier
])

from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.2)

pip.fit(msg_train,label_train)

from sklearn.metrics import classification_report, confusion_matrix

predictions = pip.predict(msg_test)

print(classification_report(predictions,label_test))
print(confusion_matrix(predictions,label_test))

# Example of classifying new messages
new_messages = ["Hello, this is a test message.", "Congratulations, you've won a prize!"]

## Preprocess the new messages
preprocessed_messages = [" ".join(text_process(message)) for message in new_messages]

## Use the trained model to predict whether the new messages are spam or ham
predictions = pip.predict(preprocessed_messages)

# Display the predictions
for message, prediction in zip(new_messages, predictions):
    print(f"Message: {message}")
    print(f"Prediction: {prediction}\n")