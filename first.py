# Load libraries
import math
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, normalize, minmax_scale, MinMaxScaler
from sklearn.preprocessing import Normalizer
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
import xlsxwriter
import xlrd


def Quadratic_discriminant (x,i,S_1,m_1,S_2,m_2,p_i):
    if i == 1:
        S_i = S_1
        m_i = m_1
    if i == 2:
        S_i = S_2
        m_i = m_2
    part1 = -0.5 * math.log( np.linalg.det(S_i) )
    print("part1: " + str(part1))

    part2 = np.dot( np.dot(x , np.linalg.inv(S_i)) , x)
    print("part2: " + str(part2))

    part3 = np.dot(np.dot(x , np.linalg.inv(S_i)) , m_i)
    print("part3: " + str(part3))

    part4 = np.dot(np.dot(m_i , np.linalg.inv(S_i)) , m_i)
    print("part4: " + str(part4))
    result = part1 - 0.5 * (part2 -2 * part3  + part4)+ math.log(p_i)
    return result


# Load dataset
firstLoad = input("First Load from url?(y/n)")
X = np.zeros([41499,5], dtype = float)
y = []
if firstLoad == 'y':
    dataset = pd.read_csv('mergedData.csv')
    #data = 'mergedData.csv'
    #names = ['NumberOfFollowings', 'NumberOfFollowers', 'NumberOfTweets', 'LengthOfScreenName', 'LengthOfDescriptionInUserProfile', 'BadUser']
    #dataframe = pd.read_csv(data, names=names)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    #X[:, [0]] = 1 - normalize(X[:, 0, None], norm='max', axis=0)
    array = dataset.values
    workbook = xlsxwriter.Workbook('mergedData.xlsx')
    worksheet = workbook.add_worksheet()
    #x=array[0,5]
    #array[1, [0]] = 1 - normalize(array[1, 0, None], norm='max', axis=0)

    for row, data in enumerate(array):
        print("row: " + str(row) + " data: " + str(data))
        col = 0
        for element in data:
            worksheet.write(row, col, element)
            col = col + 1
    workbook.close()
    exit()

else:
    wb = xlrd.open_workbook('mergedData.xlsx')
    sh = wb.sheet_by_index(0)
    input("number of rows: " + str(sh.nrows) + " number of columns: " + str(sh.ncols))
    for i in range(sh.nrows):
        for j in range(sh.ncols):
            element = sh.cell_value(i,j)
            print("i: " + str(i) + " j: " + str(j) + " element: " + str(element))
            if j==sh.ncols-1:
                y.append(element)
            else:
                X[i][j] = element


X_trains, X_test, Y_trains, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_trains, Y_trains, test_size=0.20, random_state=1)

S = np.cov(X_train.T)
m = np.mean(X_train,axis = 0)

X_train_1 = [0,0 , 0, 0, 0]
X_train_2 = [0,0 , 0, 0, 0]

for i in range(np.size(X_train,0)):
    if 0 == Y_train[i]:
        row_to_be_added = np.array(X_train[i,:])
        X_train_1 = np.vstack((X_train_1, row_to_be_added))

    if 1 == Y_train[i]:
        row_to_be_added = np.array(X_train[i,:])
        X_train_2 = np.vstack((X_train_2, row_to_be_added))

X_train_1 = np.delete(X_train_1, 0, 0)
X_train_2 = np.delete(X_train_2, 0, 0)

S_1 = np.cov(X_train_1.T)
m_1 = np.mean(X_train_1,axis = 0)

S_2 = np.cov(X_train_2.T)
m_2 = np.mean(X_train_2,axis = 0)

p_1=np.divide( X_train_1.size,X_train.size)
p_2=np.divide( X_train_1.size,X_train.size)
print("S_1" + str(S_1))
print("m_1" + str(m_1))


row_to_be_added1=[0]
for i in range(np.size(X_validation,0)):
    result1 = Quadratic_discriminant(X_validation[i, :], 1, S_1, m_1, S_2, m_2, p_1)
    result2 = Quadratic_discriminant(X_validation[i, :], 2, S_1, m_1, S_2, m_2, p_2)
    if result1>result2:
	    row_to_be_added1.append(0.)
    else:
        row_to_be_added1.append(1.)
row_to_be_added1 = np.delete(row_to_be_added1, 0)
print ("Quadratic_discriminant accuracyvalidation:", accuracy_score(Y_validation, row_to_be_added1))


row_to_be_added9=[0]
for i in range(np.size(X_test,0)):
    result1 = Quadratic_discriminant(X_test[i, :], 1, S_1, m_1, S_2, m_2, p_1)
    result2 = Quadratic_discriminant(X_test[i, :], 2, S_1, m_1, S_2, m_2, p_2)
    if result1>result2:
	    row_to_be_added9.append(0.)
    else:
        row_to_be_added9.append(1.)
row_to_be_added9 = np.delete(row_to_be_added9, 0)
print ("Quadratic_discriminant accuracy:", accuracy_score(Y_test, row_to_be_added9))



#______LinearDA______

def LDA(x,i,S,m_1,m_2,p_i):
    if i == 1:
        m_i = m_1
    if i == 2:
        m_i = m_2
    part1 = np.dot(np.dot(x , np.linalg.inv(S)) , m_i)
    print("part1: " + str(part1))

    part2 = np.dot(np.dot(m_i , np.linalg.inv(S)) , m_i)
    print("part2: " + str(part2))
    result = part1 - 0.5 * (part2) + math.log(p_i)
    return result


S = np.cov(X_train.T)
m = np.mean(X_train,axis = 0)
m_1 = np.mean(X_train_1,axis = 0)
m_2 = np.mean(X_train_2,axis = 0)


print("S" + str(S))
print("m_1" + str(m_1))

row_to_be_added2=[0]
for i in range(np.size(X_validation,0)):
    result3 = LDA(X_validation[i, :], 1, S, m_1, m_2, p_1)
    result4 = LDA(X_validation[i, :], 2, S, m_1, m_2, p_2)
    if result3>result4:
	    row_to_be_added2.append(0.)
    else:
        row_to_be_added2.append(1.)
row_to_be_added2 = np.delete(row_to_be_added2, 0)
print ("LDA accuracyvalidation:", accuracy_score(Y_validation, row_to_be_added2))

row_to_be_added10=[0]
for i in range(np.size(X_test,0)):
    result3 = LDA(X_test[i, :], 1, S, m_1, m_2, p_1)
    result4 = LDA(X_test[i, :], 2, S, m_1, m_2, p_2)
    if result3>result4:
	    row_to_be_added10.append(0.)
    else:
        row_to_be_added10.append(1.)
row_to_be_added10 = np.delete(row_to_be_added10, 0)
print ("LDA accuracy:", accuracy_score(Y_test, row_to_be_added10))


#__________NavieBayes_________

def NavieBayes(x,m_1_1,s_1):
	result =  math.pow(np.divide(x-(m_1_1),s_1),2)
	return result

s_1=np.cov(X_train[:,0])
s_2=np.cov(X_train[:,1])
s_3=np.cov(X_train[:,2])
s_4=np.cov(X_train[:,3])
s_5=np.cov(X_train[:,4])

m_1_1=m_1[0]
m_1_2=m_1[1]
m_1_3=m_1[2]
m_1_4=m_1[3]
m_1_5=m_1[4]

m_2_1=m_2[0]
m_2_2=m_2[1]
m_2_3=m_2[2]
m_2_4=m_2[3]
m_2_5=m_2[4]


row_to_be_added3=[0]
for i in range(np.size(X_validation,0)):

    result11 = NavieBayes(X_validation[i, 0],m_1_1,s_1)
    result12 = NavieBayes(X_validation[i, 1],m_1_2,s_2)
    result13 = NavieBayes(X_validation[i, 2],m_1_3,s_3)
    result14 = NavieBayes(X_validation[i, 3],m_1_4,s_4)
    result15 = NavieBayes(X_validation[i, 4],m_1_5,s_5)

    final1= - 0.5 * (result11+result12+result13+result14+result15) + math.log(p_1)


    result111 = NavieBayes(X_validation[i, 0],m_2_1,s_1)
    result112 = NavieBayes(X_validation[i, 1],m_2_2,s_2)
    result113 = NavieBayes(X_validation[i, 2],m_2_3,s_3)
    result114 = NavieBayes(X_validation[i, 3],m_2_4,s_4)
    result115 = NavieBayes(X_validation[i, 4],m_2_5,s_5)

    final2= - 0.5 * (result111+result112+result113+result114+result115) + math.log(p_2)

    if final1>final2:
	    row_to_be_added3.append(0.)
    else:
        row_to_be_added3.append(1.)
row_to_be_added3 = np.delete(row_to_be_added3, 0)
print ("NavieBayes accuracyvalidation:", accuracy_score(Y_validation, row_to_be_added3))

row_to_be_added11=[0]
for i in range(np.size(X_test,0)):

    result11 = NavieBayes(X_test[i, 0],m_1_1,s_1)
    result12 = NavieBayes(X_test[i, 1],m_1_2,s_2)
    result13 = NavieBayes(X_test[i, 2],m_1_3,s_3)
    result14 = NavieBayes(X_test[i, 3],m_1_4,s_4)
    result15 = NavieBayes(X_test[i, 4],m_1_5,s_5)
    final1=-0.5*(result11+result12+result13+result14+result15)+math.log(p_1)

    result111 = NavieBayes(X_test[i, 0],m_2_1,s_1)
    result112 = NavieBayes(X_test[i, 1],m_2_2,s_2)
    result113 = NavieBayes(X_test[i, 2],m_2_3,s_3)
    result114 = NavieBayes(X_test[i, 3],m_2_4,s_4)
    result115 = NavieBayes(X_test[i, 4],m_2_5,s_5)
    final2=-0.5*(result111+result112+result113+result114+result115)+math.log(p_2)

    if final1>final2:
	    row_to_be_added11.append(0.)
    else:
        row_to_be_added11.append(1.)
row_to_be_added11 = np.delete(row_to_be_added11, 0)
print ("NavieBayes accuracy:", accuracy_score(Y_test, row_to_be_added11))


#_______Euclidean_distance_______

def Euclidean_distance(x,m_1_1):
	result =  math.pow(x-(m_1_1),2)
	return result


row_to_be_added7=[0]
for i in range(np.size(X_validation,0)):

    result117 = Euclidean_distance(X_validation[i, 0],m_1_1)
    result127 = Euclidean_distance(X_validation[i, 1],m_1_2)
    result137 = Euclidean_distance(X_validation[i, 2],m_1_3)
    result147 = Euclidean_distance(X_validation[i, 3],m_1_4)
    result157 = Euclidean_distance(X_validation[i, 4],m_1_5)

    final17= - 0.5 * (result117+result127+result137+result147+result157) + math.log(p_1)


    result1117 = Euclidean_distance(X_validation[i, 0],m_2_1)
    result1127 = Euclidean_distance(X_validation[i, 1],m_2_2)
    result1137 = Euclidean_distance(X_validation[i, 2],m_2_3)
    result1147 = Euclidean_distance(X_validation[i, 3],m_2_4)
    result1157 = Euclidean_distance(X_validation[i, 4],m_2_5)

    final27= - 0.5 * (result1117+result1127+result1137+result1147+result1157) + math.log(p_2)

    if final17>final27:
	    row_to_be_added7.append(0.)
    else:
        row_to_be_added7.append(1.)
row_to_be_added7 = np.delete(row_to_be_added7, 0)
print ("Euclidean_distance accuracyvalidation:", accuracy_score(Y_validation, row_to_be_added7))

row_to_be_added8=[0]
for i in range(np.size(X_test,0)):

    result117 = Euclidean_distance(X_test[i, 0],m_1_1)
    result127 = Euclidean_distance(X_test[i, 1],m_1_2)
    result137 = Euclidean_distance(X_test[i, 2],m_1_3)
    result147 = Euclidean_distance(X_test[i, 3],m_1_4)
    result157 = Euclidean_distance(X_test[i, 4],m_1_5)


    final17= -0.5 * (result117+result127+result137+result147+result157) + math.log(p_1)

    result1117 = Euclidean_distance(X_test[i, 0],m_2_1)
    result1127 = Euclidean_distance(X_test[i, 1],m_2_2)
    result1137 = Euclidean_distance(X_test[i, 2],m_2_3)
    result1147 = Euclidean_distance(X_test[i, 3],m_2_4)
    result1157 = Euclidean_distance(X_test[i, 4],m_2_5)

    final27= -0.5 * (result1117+result1127+result1137+result1147+result1157) + math.log(p_2)

    if final17>final27:
	    row_to_be_added8.append(0.)
    else:
        row_to_be_added8.append(1.)
row_to_be_added8 = np.delete(row_to_be_added8, 0)
print ("Euclidean_distance accuracy:", accuracy_score(Y_test, row_to_be_added8))


#______NearestMean______

def NearestMean(x,i,m_1,m_2):
	if i == 1:
		m_i = m_1

	if i == 2:
		m_i = m_2

	result = -math.pow(np.linalg.norm(x-m_i),2)
	return result

row_to_be_added6=[0]
for i in range(np.size(X_validation,0)):
    result1 = NearestMean(X_validation[i, :], 1,m_1, m_2)
    result2 = NearestMean(X_validation[i, :], 2,m_1, m_2)

    if result1>result2:
	    row_to_be_added6.append(0.)
    else:
        row_to_be_added6.append(1.)
row_to_be_added6 = np.delete(row_to_be_added6, 0)
print ("NearestMean accuracyvalidation:", accuracy_score(Y_validation, row_to_be_added6))

row_to_be_added13=[0]
for i in range(np.size(X_test,0)):
    result1 = NearestMean(X_test[i, :], 1,m_1, m_2)
    result2 = NearestMean(X_test[i, :], 2,m_1, m_2)

    if result1>result2:
	    row_to_be_added13.append(0.)
    else:
        row_to_be_added13.append(1.)
row_to_be_added13 = np.delete(row_to_be_added13, 0)
print ("NearestMean accuracy:", accuracy_score(Y_test, row_to_be_added13))


#______TemplateMatch______

def Template_matching(x,i,S,m,S_1,m_1,S_2,m_2,p_i):
	if i == 1:
		S_i = S_1
		m_i = m_1

	if i == 2:
		S_i = S_2
		m_i = m_2

	result = np.dot(m_i.T,x)
	return result

row_to_be_added5=[0]
for i in range(np.size(X_validation,0)):
    result1 = Template_matching(X_validation[i, :], 1, S, m, S_1, m_1, S_2, m_2, p_1)
    result2 = Template_matching(X_validation[i, :], 2, S, m, S_1, m_1, S_2, m_2, p_2)

    if result1>result2:
	    row_to_be_added5.append(0.)
    else:
        row_to_be_added5.append(1.)
row_to_be_added5 = np.delete(row_to_be_added5, 0)

print ("Template_matching accuracyvalidation:", accuracy_score(Y_validation, row_to_be_added5))

row_to_be_added12=[0]
for i in range(np.size(X_test,0)):
    result1 = Template_matching(X_test[i, :], 1, S, m, S_1, m_1, S_2, m_2, p_1)
    result2 = Template_matching(X_test[i, :], 2, S, m, S_1, m_1, S_2, m_2, p_2)

    if result1>result2:
	    row_to_be_added12.append(0.)
    else:
        row_to_be_added12.append(1.)
row_to_be_added12 = np.delete(row_to_be_added12, 0)

print ("Template_matching accuracy:", accuracy_score(Y_test, row_to_be_added12))

def w(u):
    if abs(u)< 0.5:
        return 1
    else:
        return 0

def Navie_estimator(x,x1,h):
    s=0
    z = 0
    for j in range(np.size(x1, 0)):
        for k in range(np.size(x1, 1)):
            z=z+math.pow(x[k]-x1[j,k],2)
            z=np.divide(z,math.pow(h,5))
            s=s+w(z)
            result=(np.divide(1,(np.size(X_test,0))*math.pow(h,5))*s)
            return result

row_to_be_added90 = [0]
for i in range(np.size(X_validation,0)):
    finalresult1=Navie_estimator(X_validation[i, :],X_train_1,e)
    finalresult2=Navie_estimator(X_validation[i, :],X_train_2,e)
    if finalresult1 >finalresult2:
        row_to_be_added90.append(0.)
    else: row_to_be_added90.append(1.)
    row_to_be_added90 = np.delete(row_to_be_added90, 0)
    print ("Navie_estimator accuracyvalidation:", accuracy_score(Y_validation, row_to_be_added90))