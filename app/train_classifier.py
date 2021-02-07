import sys
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
nltk.download('stopwords','wordnet')
from nltk.corpus import stopwords, wordnet
import re
from collections import defaultdict

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    
    engine = create_engine('sqlite:///' + str (database_filepath))
    df = pd.read_sql ('data', engine)
    X = df ['message']
    y = df.iloc[:,4:]
    ## Since child_alone collumn contains only 0s, I dropped it
    y=y.drop('child_alone',axis=1)
    category_names=y.columns.values
    
    return X, y, category_names


def tokenize(text):
    
    """
    Convert text into tokens
    
    Input:
        text - message that needs to be tokenized
    Output:
        clean_tokens - list of tokens from the given message
    """
    
    # remove url place holder
    
    url_regex= r'(https?://\S+)'
    text = re.sub(url_regex, 'urlplaceholder',text)
    
    #tokenize message into words 
    
    tokens=word_tokenize(text)
    
    #remove the stop words 
    
    filtered_tokens=[w for w in tokens if not w in stopwords.words('english')]
    
    #remove punctuation and tokens containing non alphabetic symbols
    
    alpha_tokens=[token.lower() for token in filtered_tokens if token.isalpha()]
    
    # make a default dictionary for the pos tagging 
    tag_map = defaultdict(lambda : wordnet.NOUN)
    tag_map['J'] = wordnet.ADJ
    tag_map['V'] = wordnet.VERB
    tag_map['R'] = wordnet.ADV

    #lemmatize tokens using pos tags from defaulct dict
    
    clean_tokens=[]
    lmtzr = WordNetLemmatizer()
    for token, tag in pos_tag(alpha_tokens):
        clean_tokens.append(lmtzr.lemmatize(token, tag_map[tag[0]]))
    
    
    return clean_tokens
    

class ContainsHelpNeed(BaseEstimator, TransformerMixin):
    
    """
    This custom transformer extracts the messages which start with verb 
    creates new feature consisting of 1 (True) and 0 (False) values.
    
    """       

    def filter_verb(self, text):
        words=tokenize(text)
        if 'help' in words or 'need' in words:
            return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.filter_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    
    """           
    Output: best model based on GridSearchCV, which whill be used 
    for evaluation 
            
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('need_help_transformer', ContainsHelpNeed())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {'classifier__estimator__n_estimators': [40,70,100] }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred=model.predict(X_test)
    
    print(classification_report(y_true=Y_test.values,y_pred=Y_pred,target_names= category_names))
    


def save_model(model, model_filepath):
    
    pickle_param = open(model_filepath, 'wb')
    pickle.dump(model,pickle_param)
    pickle_param.close()

    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
