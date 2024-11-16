import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def get_clean_data():
    data = pd.read_csv('data/data.csv')
    data = data.drop(['Unnamed: 32', 'id'], axis=1)  # Correction de la suppression de colonnes
    data['diagnosis'] = data['diagnosis'].astype('category').cat.codes
    return data

def creat_model(data):
    X = data.drop(['diagnosis'], axis=1)  # Spécifier axis=1 pour supprimer la colonne
    y = data['diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Entraînement du modèle
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test du modèle
    y_pred = model.predict(X_test)
    print('Accuracy of our model:', accuracy_score(y_test, y_pred))
    print('Classification report:', classification_report(y_test, y_pred))

    return model, scaler

def main():
    data = get_clean_data()
    model, scaler = creat_model(data)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()
