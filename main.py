# Load libraries
from pandas import read_csv as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


import pandas as pd
dataset1=pd.read_csv(r"C:\Users\Ratin Rayaneh\PycharmProjects\spammer\content_polluters.csv")
print(dataset1)

dataset2=pd.read_csv(r"C:\Users\Ratin Rayaneh\PycharmProjects\spammer\legitimate_users.csv")
print(dataset2)

dataset3=pd.read_csv(r"C:\Users\Ratin Rayaneh\PycharmProjects\spammer\mergedData.csv")
print(dataset3)

# shape
print(dataset1.shape)
print(dataset2.shape)
print(dataset3.shape)

# head
print(dataset3.head(20))