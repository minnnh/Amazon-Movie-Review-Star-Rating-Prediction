import pandas as pd
import nltk
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

Lemma = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('stopwords')
snow_stemmer = SnowballStemmer(language='english')

def stem(text):
    return ' '.join([SnowballStemmer("english").stem(word) for word in word_tokenize(text)])


def process(df):
    # This is where you can do all your processing

    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Helpfulness'] = df['Helpfulness'].fillna(0)
    
    df['Summary'] = df['Summary'].fillna('')
    df['Text'] = df['Text'].fillna('')
    df['Summary'] = df['Summary'].str.lower()
    df['Text'] = df['Text'].str.lower()
    
    # clean Summary
    df['stemmed_Summary'] = df['Summary'].apply(stem)
    df['stemmed_Summary'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in df['stemmed_Summary']]
    
    # print('--------------test 1',df.isna().sum())

    # rate Summary
    vectorizer = TfidfVectorizer(max_df=.5, min_df=.01).fit(df['stemmed_Summary'])
    df1 = vectorizer.transform(df['stemmed_Summary'])
    df1 = pd.DataFrame(df1.toarray(), columns= vectorizer.get_feature_names())
    df['sentiment'] = df1.sum(axis = 1)
    
    # get the most frequent words
    df2 = df1[['great', 'good', 'love', 'bad', 'best', 'fun']].copy()
    df = pd.concat([df, df2], axis=1)

    # length of Text and Summary
    df['ReviewLength'] = df.apply(lambda row : len(row['Text'].split()) if type(row['Text']) == str else 0, axis = 1)
    df['SummaryLength'] = df.apply(lambda row : len(row['Summary'].split()) if type(row['Summary']) == str else 0, axis = 1)
    df['ReviewLength'] = df['ReviewLength'].replace(np.nan, 0)
    df['SummaryLength'] = df['SummaryLength'].replace(np.nan, 0)

    # deal with time data
    df['Date'] = pd.to_datetime(df['Time'], unit='s')
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Hour'] = df['Date'].dt.hour
    
    #df = df.drop(columns = ['HelpfulnessNumerator', 'HelpfulnessDenominator'])
    # print('------------test 2',df.isna().sum())

    return df
    

print('process done')
# Load the dataset
trainingSet = pd.read_csv("./data/train.csv")

# Process the DataFrame
train_processed = process(trainingSet)

# Load test set
submissionSet = pd.read_csv("./data/test.csv")

# print('------------test 3',train_processed.isna().sum())
# Merge on Id so that the test set can have feature columns as well
testX= pd.merge(train_processed, submissionSet, left_on='Id', right_on='Id')
testX = testX.drop(columns=['Score_x'])
testX = testX.rename(columns={'Score_y': 'Score'})

# print('------------test 4',train_processed.isna().sum())
# The training set is where the score is not null
trainX =  train_processed[train_processed['Score'].notnull()]

# the data used for the submission
testX.to_csv("./data/X_test.csv", index=False)
# print('------------test 5',trainX.isna().sum())
# the data used to train
trainX.to_csv("./data/X_train.csv", index=False)

print('done')