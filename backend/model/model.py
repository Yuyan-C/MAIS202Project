import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load


df = pd.read_csv('/Users/chenyuyan/PycharmProjects/MAIS202_project/venv/covtype.csv')
x = df['Horizontal_Distance_To_Hydrology'].values
y = df['Vertical_Distance_To_Hydrology'].values
distance = np.sqrt(np.square(x) + np.square(y))
df = df.drop(['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology'], axis=1)
df.insert(3, "Distance_To_Hydrology", distance)
# set the number of training set, validation set, and testing set
# get the number of features
num_features = len(df.columns) - 1
# set X and y
scaler = MinMaxScaler()
X = scaler.fit_transform(df.iloc[:, :num_features])
y = df.iloc[:, num_features:].to_numpy().reshape(len(df.index))
# set training, validation, and testing sets
# 64% training, 16% validation, 20% testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)  #
ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_resample(X_train, y_train)


class COVModel:

    def train(self):
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        dump(clf, '/Users/chenyuyan/PycharmProjects/MAIS202_project/venv/backend/model/classifier.joblib')
        print("predicting training set...")
        print(datetime.now().strftime("%H:%M:%S.%f"))
        y_train_pred = clf.predict(X_train)
        print("predicting validation set...")
        print(datetime.now().strftime("%H:%M:%S.%f"))
        y_valid_pred = clf.predict(X_valid)
        print("predicting testing set...")
        print(datetime.now().strftime("%H:%M:%S.%f"))
        y_test_pred = clf.predict(X_test)
        print("training accuracy: {}".format(accuracy_score(y_train, y_train_pred)))
        print("validation accuracy: {}".format(accuracy_score(y_valid, y_valid_pred)))
        print("testing accuracy: {}".format(accuracy_score(y_test, y_test_pred)))
        print("training accuracy (balanced): {}".format(balanced_accuracy_score(y_train, y_train_pred)))
        print("validation accuracy (balanced): {}".format(balanced_accuracy_score(y_valid, y_valid_pred)))
        print("testing accuracy (balanced): {}".format(balanced_accuracy_score(y_test, y_test_pred)))


if __name__ == "__main__":
    model = COVModel()
    model.train()
