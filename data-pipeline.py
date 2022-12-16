# import packages
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


def load_data(data_file):
    """Load the necessary data.

    Loads the message file and the categorrie file 
    and merge them into one Dataframe based on a inner join on the column name 'id'

    Args:
        messages_filepath (str): The filepath to the messages.csv data as a string
        categories_filepath (str): The filepath to the caregories.csv data as a string

    Return:
        df (DataFrame): A pandas dataframe with the merged messages and categories
    """
    # read in files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,how="inner", on="id")

    # clean data
    # categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.head(1)

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')
    
    # drop duplicates
    df.drop_duplicates(inplace=True)

    # load to database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    try: 
        df.to_sql('tbl_disastermessages', engine, index=False)  
    except:
        print("Exception: The table seems to exists already")

    # define features and label arrays
    X = df['message']
    y = df.iloc[:,4:]

    return X, y


def build_model():
    # text processing and model pipeline
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



    # define parameters for GridSearchCV
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
        cv=3,
        n_jobs=-1
        )



    # create gridsearch object and return as final model pipeline


    return model_pipeline


def train(X, y, model):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # fit model


    # output model test results


    return model


def export_model(model):
    # Export model as a pickle file
    filename = 'classifier.pkl'
    with open (filename, 'wb') as f:
        pickle.dump(cv_rf, f)



def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline
