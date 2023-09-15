import numpy as np
import nltk
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import plot_tree

def cv(X, y):
    scores = []
    neighbors=np.arange(3,15)
    for k in range(3, 15):
        decision_tree = DecisionTreeClassifier(max_depth = k).fit(X, y)
        score = cross_val_score(decision_tree, X, y, cv=10, scoring='accuracy', n_jobs=-1)
        scores.append(np.mean(score))
    # plot
    neighbors=np.arange(3,15)
    plt.figure(figsize=(12,10))
    plt.title("k-fold cross validation (with k=10) ")
    plt.plot(neighbors, scores, label="Scores")
    plt.legend()
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.show()
    
    max_score = max(scores)
    depth = scores.index(max_score)
    print(f"The best score is {max_score}, the depth is {neighbors[depth]}")
    return neighbors[depth]

# Load files into DataFrames
X_train = pd.read_csv("./data/X_train.csv")
X_submission = pd.read_csv("./data/X_test.csv")

# Split training set into training and testing set
X_train = X_train.sample(n = 300000)
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
from sklearn.tree import plot_tree

md = cv(X_train_processed, Y_train)

model=DecisionTreeClassifier(max_depth=md).fit(X_train_processed,Y_train)

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
submission.to_csv("./data/submission_dctree.csv", index=False)
