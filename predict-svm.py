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
from sklearn.model_selection import GridSearchCV

"""
def cv(X, y):
    scores = []
    C_range=list(np.arange(0.1,6,0.1))
    for k in C_range:
        svc = SVC(kernel='linear', C=k)
        score = cross_val_score(svc, X, y, cv=10, scoring='accuracy', n_jobs=-1)
        scores.append(np.mean(score))
        
    # plot
    neighbors=np.arange(1,10)
    plt.figure(figsize=(12,10))
    plt.title("k-fold cross validation (with k=10) ")
    plt.plot(neighbors, scores, label="Scores")
    plt.legend()
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.show()
    
    max_score = max(scores)
    depth = scores.index(max_score)
    print(f"The best score is {max_score}, the depth is {depth}")
    return depth
"""
def cv(X, y):

    scores = []
    C_range= [1,3,5,10,40,60,80,100]
    for this_C in [1,3,5,10,40,60,80,100]:
        svc = SVC(kernel='linear',C=this_C).fit(X,y)
        #svc = SVC(kernel='linear', C=k)
        score = cross_val_score(svc, X, y, cv=10, scoring='accuracy', n_jobs=-1)
        scores.append(np.mean(score))
        print(f'c = {this_C}, score: {score}')
    # plot
    neighbors=C_range
    plt.figure(figsize=(12,10))
    plt.title("k-fold cross validation (with k=10) ")
    plt.plot(neighbors, scores, label="Scores")
    plt.legend()
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.show()
    
    max_score = max(scores)
    depth = scores.index(max_score)
    print(f"The best score is {max_score}, the depth is {C_range[depth]}")
    return C_range[depth]

# Load files into DataFrames
X_train = pd.read_csv("./test/X_train.csv")
X_submission = pd.read_csv("./test/X_test.csv")


X_train = X_train.sample(n = 100000)

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
#md = cv(X_train_processed, Y_train)
#model = SVC(kernel='linear', C=md).fit(X_train_processed, Y_train)
model = SVC(kernel='linear',probability=True).fit(X_train_processed, Y_train)

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
submission.to_csv("./data/submission_svm.csv", index=False)
