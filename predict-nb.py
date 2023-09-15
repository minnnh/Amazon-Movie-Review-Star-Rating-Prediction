import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.cluster import KMeans
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

def cv(X, y):
    scores = []
    neighbors=np.arange(1,20)
    for k in range(1, 20):
        knn_model = KNeighborsClassifier(n_neighbors=k)

        score = cross_val_score(knn_model, X, y, cv=10, scoring='accuracy', n_jobs=-1)
        scores.append(np.mean(score))
        
    # plot
    neighbors=np.arange(1,20)
    plt.figure(figsize=(12,10))
    plt.title("k-fold cross validation (with k=10) ")
    plt.plot(neighbors, scores, label="Scores")
    plt.legend()
    plt.xlabel("Number of Neighbour")
    plt.ylabel("Accuracy")
    plt.show()
    
    max_score = max(scores)
    depth = scores.index(max_score)
    print(f"The best score is {max_score}, the depth is {neighbors[depth]}")
    return neighbors[depth]

# Load files into DataFrames
X_train = pd.read_csv("./data/X_train.csv")
X_submission = pd.read_csv("./data/X_test.csv")

# X_train['Summary'] = X_train['Summary'].fillna('')
# X_train['Text'] = X_train['Text'].fillna('')

# X_submission['Summary'] = X_submission['Summary'].fillna('')
# X_submission['Text'] = X_submission['Text'].fillna('')

X_train = X_train.sample(n = 300000)

# Split training set into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(
        X_train.drop(['Score'], axis=1),
        X_train['Score'],
        test_size=1/4.0,
        random_state=0
    )

# This is where you can do more feature selection
X_train_processed = X_train.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'Time', 'Date'])
X_test_processed = X_test.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary','Time','Date'])
X_submission_processed = X_submission.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'Score','Date', 'Time'])
# Learn the model
model = GaussianNB().fit(X_train_processed, Y_train)

# Predict the score using the model
Y_test_predictions = model.predict(X_test_processed)
X_submission['Score'] = model.predict(X_submission_processed)

# Evaluate your model on the testing set
print("Accuracy on testing set = ", accuracy_score(Y_test, Y_test_predictions))
print("Accuracy on testing set = ", mean_squared_error(Y_test, Y_test_predictions))

# Plot a confusion matrix
cm = confusion_matrix(Y_test, Y_test_predictions, normalize='true')
sns.heatmap(cm, annot=True)
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Create the submission file
submission = X_submission[['Id', 'Score']]
submission.to_csv("./data/submission_gnb.csv", index=False)

