import sys
import numpy as np
import pandas as pd
import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine

# nlp libraries
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download(['stopwords', 'punkt', 'wordnet'])

# sklearn libraries
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV 
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """Load data from the database.

    Args:
        database_filepath (str): The relative path to the databse file

    Returns:
        X (Dataframe): Feature Dataframe
        y (Dataframe): Target Data Array
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(table_name='tbl_disastermessages', con=engine)
    X = df['message']
    y = df.iloc[:,4:]
    return X, y


def tokenize(text):
    """Tokenzize input text into words and returns the base from the word based on english.

    Args:
        text (str): input message

    Returns:
        tokens (list of str): A returning list of tokens found in the input message
    """
    # Normalize and tokenize and remove punctuation.
    words = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))

    # Remove stopwords
    tokens = [t for t in words if t not in stopwords.words('english')]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens
    

def build_model():
    """Build the pipeline and grid search classification model.

    Returns:
    cv (GridSeach Object): Classification model
    """
    #Random Forest classifier Pipeline
    pipeline_rf = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ]) 

    # Grid search parameters
    parameters_rf = {
        'clf__estimator__n_estimators': [5,10],
        'clf__estimator__min_samples_split':[4,6]
    }

    cv = GridSearchCV(pipeline_rf,param_grid = parameters_rf, 
        verbose=4, 
        )

    return cv


def evaluate_model(model, X_test, y_test): #, category_names):
    """Evaluate the model and print classifcation report.

    Args:
        model (Object): The classification model.
        X_test (Dataframe): _description_
        y_test (Dateframe): _description_
        category_names (_type_): _description_
    """
    # Predict the model
    y_pred = model.predict(X_test)

    # Printing the classification report for each label
    for i, column in enumerate(y_test):
        print('Feature {}: {}'.format(i+1, column))
        print(classification_report(y_test[column], y_pred[:, i]))
    accuracy = (y_pred == y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))


def save_model(model, model_filepath):
    """Save the model as a pickle object.

    Args:
        model (object): classifcation model
        model_filepath (str): the path where the model is saved
    """
    # Export a pickle file for the final model
    with open (model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """Run the functions in the right order and parameters."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()