import pandas as pd
import numpy as np

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import joblib

diabetes_dataset = pd.read_csv('./data/diabetes.csv')

class Model:

    def train(self):
        self.initizalize_test_data()
        self.model = DecisionTreeClassifier()
        self.model.fit(self.X_train, self.Y_train)

    def initizalize_test_data(self):
        X = diabetes_dataset.drop(columns='Outcome', axis=1)
        Y = diabetes_dataset['Outcome']

        X = scale(X)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2)

    @property
    def accuracy(self):
        self.initizalize_test_data()
        model_pred = self.model.predict(self.X_test)
        return accuracy_score(self.Y_test, model_pred)

    def save_model(self, filename):
        joblib.dump(self.model, filename)

    def load_model(self, filename):
        self.model = joblib.load(filename)

    def predict(self, array):
        data = np.array(array)
        data = data.reshape(1, -1)
        prediction = self.model.predict(data)
        return prediction[0]