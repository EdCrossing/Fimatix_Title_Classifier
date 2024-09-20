import pandas as pd
import numpy as np
import chardet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def load_data(train_path, test_path):
    """
    Load training and testing data from CSV files with specified encoding.
    """
    #can set up a chardet for the encoding
    #encoding_train = chardet.detect(train_path.read())
    #encoding_test = chardet.detect(test_path.read())
    #print(encoding_train)

    encoding = 'cp1252'

    #treating '.' as NaN + drop extra delimiters
    train_df = pd.read_csv(train_path, encoding=encoding, na_values='.')
    test_df = pd.read_csv(test_path, encoding=encoding, na_values='.')
    train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
    test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

    print(f"Training df columns: {train_df.columns.tolist()}")
    print(f"Testing df columns: {test_df.columns.tolist()}")

    return train_df, test_df

def preprocess_data(df, label_encoder=None, scaler=None, imputer=None, tfidf_vectorizer=None, is_train=True):
    """
    Preprocess the dataframe:
    - Encode categorical features
    - Handle non-numeric values in numerical features
    - Impute missing values
    - Vectorize text data using TF-IDF
    - Scale numerical features
    """
    #Feature categories
    categorical_features = ['IsBold', 'IsItalic', 'IsUnderlined', 'FontType']
    numerical_features = ['Left', 'Right', 'Top', 'Bottom']
    text_feature = 'Text'
    #convert to string
    for feature in categorical_features:
        df[feature] = df[feature].astype(str)

    if is_train:
        #swap true/false to 1,0
        label_encoder = {}
        for feature in categorical_features:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
            label_encoder[feature] = le
    else:
        #handle unseen labels
        for feature in categorical_features:
            le = label_encoder[feature]
            df[feature] = df[feature].apply(lambda x: x if x in le.classes_ else 'Unknown')
            if 'Unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'Unknown')
            df[feature] = le.transform(df[feature])

    #convert text to TF-IDF vectors
    df[text_feature] = df[text_feature].fillna('')
    print(f"NaNs in 'Text' column: {df[text_feature].isnull().sum()}")

    if is_train:
        tfidf_vectorizer = TfidfVectorizer(max_features=500,stop_words='english')
        text_tfidf = tfidf_vectorizer.fit_transform(df[text_feature]).toarray()
    else:
        text_tfidf = tfidf_vectorizer.transform(df[text_feature]).toarray()

    #add TFIDF to dataframe
    tfidf_df = pd.DataFrame(text_tfidf, columns=[f"tfidf_{i}" for i in range(text_tfidf.shape[1])])
    df = df.drop(columns=[text_feature])
    df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    #handle non-numeric values in numerical features
    for feature in ['Left', 'Right', 'Top', 'Bottom']:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')

    #fill missing values
    if is_train:
        imputer = SimpleImputer(strategy='mean')
        df[numerical_features] = imputer.fit_transform(df[numerical_features])
    else:
        df[numerical_features] = imputer.transform(df[numerical_features])

    #scale locations
    if is_train:
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
    else:
        df[numerical_features] = scaler.transform(df[numerical_features])

    # Check for any remaining NaN values
    if df[numerical_features].isnull().sum().sum() > 0:
        print("Warning: There are still NaN values in numerical features after imputation.")

    #check for labeling
    if 'Label' not in df.columns:
        if is_train:
            raise ValueError("Training data must contain 'Label' column.")
        else:
            y = None
            X = df
            return X, y

    if is_train:
        X = df.drop(columns=['Label'])
        y = df['Label']
        return X, y, label_encoder, scaler, imputer, tfidf_vectorizer
    else:
        X = df.drop(columns=['Label'])
        y = df['Label']
        return X, y

def prepare_datasets(train_path, test_path):
    train_df, test_df = load_data(train_path, test_path)
    #preprocess training data
    X, y, label_encoder, scaler, imputer, tfidf_vectorizer = preprocess_data(train_df, is_train=True)
    #preprocess testing data
    X_test, y_test = preprocess_data(test_df, label_encoder=label_encoder, scaler=scaler, imputer=imputer, tfidf_vectorizer=tfidf_vectorizer, is_train=False)

    print("Splitting training data into training and validation sets...")
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Data preparation complete.")
    return X_train, X_val, y_train, y_val, X_test, y_test, label_encoder, scaler, imputer, tfidf_vectorizer
