import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB

def decision_tree(df):
    features = ["energy", "mean", "potencia", "zero_crossing_rate"]
    X = df[features]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"]) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    decoded_labels = label_encoder.inverse_transform([0, 1, 2])  
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print("📊 Matriz de confusión (con clases decodificadas):")
    conf_matrix_df = pd.DataFrame(conf_matrix, index=decoded_labels, columns=decoded_labels)
    print(conf_matrix_df)


def naive_bayes_classifier(df):
    features = ["energy", "mean", "potencia", "zero_crossing_rate"]
    X = df[features]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    decoded_labels = label_encoder.inverse_transform([0, 1, 2]) 
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print("📊 Matriz de confusión (con clases decodificadas):")
    conf_matrix_df = pd.DataFrame(conf_matrix, index=decoded_labels, columns=decoded_labels)
    print(conf_matrix_df)
