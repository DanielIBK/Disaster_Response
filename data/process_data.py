"""
ETL Preprocessing for the Web App.

Runs the process routine to load extract and clean the input
data from diffrent csv files and saved them as a ready to go database

    Returns:
        DisasterData.db: Clean Database with necessary data
"""
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load the necessary data.

    Loads the message file and the categorrie file 
    and merge them into one Dataframe based on a inner join on the column name 'id'

    Args:
        messages_filepath (str): The filepath to the messages.csv data as a string
        categories_filepath (str): The filepath to the caregories.csv data as a string

    Return:
        df (DataFrame): A pandas dataframe with the merged messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,how="inner", on="id")

    return df

def clean_data(df):
    """Clean the loaded data.
    
    Function:
      Cleans the Dataframe df and combines them into one large df

    Args:
        df (dataframe): The input dataframe which should be cleaned

    Return:
      df (DataFrame): A cleaned dataframe with merged data from messages and categories
    """
    # Split the multi column 'categories' into separate category columns based on seperater ';'.
    categories = df['categories'].str.split(';', expand=True)

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
    
    # Replace the 2 in the column related with 1
    
    categories.related.replace(2,1,inplace=True)
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')
    
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """Safe the data as SQL Lite.

    Args:
        df (dataframe): The Dateframe which should be safed in a database
        database_filename (string): The file path as a string to the database 
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    try: 
        df.to_sql('tbl_disastermessages', engine, index=False)  
    except:
        print("Exception: The table seems to exists already")


def main():
    """Start the necessary function for the Main module."""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()