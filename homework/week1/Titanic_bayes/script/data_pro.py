# -*- coding: utf-8 -*-
import pandas as pd #数据分析
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_train = pd.read_csv(".\\homework\\week1\\Titanic_bayes\\train.csv")
# (1) 看列名
total_num = data_train.shape[0]
#print(data_train.columns)

# (2) 看每列性质，空值和类型
#print(data_train.info())

# (3) 看每列统计信息
#print(data_train.describe())
df_train = data_train.drop(['PassengerId','Name','Ticket'],axis = 1)
total_survived = df_train['Survived'].sum()
total_nosurvived = total_num - total_survived
survived_df = df_train[df_train[ 'Survived'] == 1 ]
#简单起见，只提取有用特征，包括Pclass,Sex,Age,SibSp,Parch,fare,embark
Pclass1_rate = df_train['Pclass'][df_train['Pclass'] == 1].count()/total_num
Pclass2_rate = df_train['Pclass'][df_train['Pclass'] == 2].count()/total_num
Pclass3_rate = df_train['Pclass'][df_train['Pclass'] == 3].count()/total_num
Pclass1_Survived = survived_df['Pclass'][survived_df['Pclass'] == 1].count()/total_survived
Pclass2_Survived = survived_df['Pclass'][survived_df['Pclass'] == 2].count()/total_survived
Pclass3_Survived = survived_df['Pclass'][survived_df['Pclass'] == 3].count()/total_survived

male_rate = df_train['Sex'][df_train['Sex'] == 'male'].count()/total_num
female_rate = df_train['Sex'][df_train['Sex'] == 'female'].count()/total_num
male_Survived = survived_df['Sex'][survived_df['Sex'] == 'male'].count()/total_survived
female_Survived = survived_df['Sex'][survived_df['Sex'] == 'female'].count()/total_survived

# 求年龄的平均值，标准差以及丢失值的数量
average_age_titanic   = df_train["Age"].mean()
std_age_titanic       = df_train["Age"].std()
count_nan_age_titanic = df_train["Age"].isnull().sum()

# 求年龄随机数，范围在 (mean - std， mean + std)
# rand_age = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
# print("rand_age type:\n",type(rand_age).__name__)

# 本来想将随机数填充进 Age 的丢失值中，但是程序报错，所以先用平均值填充
df_train["Age"][np.isnan(df_train['Age'])].fillna(average_age_titanic) 

#titanic_df['Age'].describe()
children_rate = df_train['Age'][df_train['Age'] <= 12].count()/total_num
juvenile_rate = df_train['Age'][(df_train['Age'] > 12) & (df_train['Age'] < 18) ].count()/total_num
adults_rate = df_train['Age'][(df_train['Age'] >= 18) & (df_train['Age'] < 65) ].count()/total_num
agedness_rate = df_train['Age'][df_train['Age'] >= 65 ].count()/total_num


children_survived = survived_df['Age'][survived_df['Age'] <= 12 ].count()/total_survived
juvenile_survived_ = survived_df['Age'][(survived_df['Age'] > 12) & (survived_df['Age'] < 18) ].count()/total_survived
adults_survived = survived_df['Age'][(survived_df['Age'] >= 18) & (survived_df['Age'] < 65)].count()/total_survived
agedness_survived = survived_df['Age'][survived_df['Age'] >= 65].count()/total_survived



sibsp_rate = df_train['SibSp'][df_train['SibSp'] != 0].count()/total_num
nosibsp_rate = df_train['SibSp'][df_train['SibSp'] == 0].count()/total_num
sibsp_survived = survived_df['SibSp'][survived_df['SibSp'] != 0].count()/total_survived
nosibsp_survived = survived_df['SibSp'][survived_df['SibSp'] == 0].count()/total_survived

parch_rate = df_train['Parch'][df_train['Parch'] !=0 ].count()/total_num
noparch_rate = df_train['Parch'][df_train['Parch'] ==0 ].count()/total_num
parch_survived = survived_df['Parch'][survived_df['Parch'] != 0].count()/total_survived
noparch_survived = survived_df['Parch'][survived_df['Parch'] == 0].count()/total_survived

#简单起见，只把票价分为两类讨论，分别为高于和低于平均票价
average_fare_titanic   = df_train["Fare"].mean()
highfare_rate = df_train['Fare'][df_train['Fare'] >= average_fare_titanic].count()/total_num
lowfare_rate = df_train['Fare'][df_train['Fare'] < average_fare_titanic].count()/total_num
highfare_survived = survived_df['Fare'][survived_df['Fare'] >= average_fare_titanic ].count() /total_survived
lowfare_survived = survived_df['Fare'][survived_df['Fare'] < average_fare_titanic ].count() /total_survived

#embark缺失值用众数s填充
df_train['Embarked'] = df_train['Embarked'].fillna('S')
embarkS_rate = df_train['Embarked'][df_train['Embarked'] == 'S'].count()/total_num
embarkQ_rate = df_train['Embarked'][df_train['Embarked'] == 'Q'].count()/total_num
embarkC_rate = df_train['Embarked'][df_train['Embarked'] == 'C'].count()/total_num
embarkS_survived = survived_df['Embarked'][survived_df['Embarked'] == 'S'].count()/total_survived
embarkQ_survived = survived_df['Embarked'][survived_df['Embarked'] == 'Q'].count()/total_survived
embarkC_survived = survived_df['Embarked'][survived_df['Embarked'] == 'C'].count()/total_survived

def predict_passenger(Pclass,Sex,Age,SibSp,Parch,Fare,Embark):   
    switchPclass_survived = {1:Pclass1_Survived,2:Pclass2_Survived,3:Pclass3_Survived}
    switchPclass_rate = {1:Pclass1_rate,2:Pclass2_rate,3:Pclass3_rate}
    test_Pclass_survived = switchPclass_survived[Pclass]
    test_Pclass_rate = switchPclass_rate[Pclass]
    
    switchSex_survived = {'male':male_Survived,'female':female_Survived}
    switchSex_rate = {'male':male_rate,'female':female_rate}
    test_Sex_survived = switchSex_survived[Sex]
    test_Sex_rate = switchSex_rate[Sex]

    if Age <= 12:
        tmp_age = 'child'
    elif (Age > 12) & (Age <18):
        tmp_age = 'juvenile'
    elif (Age > 18) & (Age < 65):
        tmp_age = 'adult'
    elif Age > 65:
        tmp_age = 'aged'
    else :
        #print("enter ageage \n")
        tmp_age = 'adult' #年龄空值默认用均值代替     
    switchAge_survived = {'child':children_survived,'juvenile':juvenile_survived_,'adult':adults_survived,'aged':agedness_survived}
    switchAge_rate = {'child':children_rate,'juvenile':juvenile_rate,'adult':adults_rate,'aged':agedness_rate}
    test_age_survived = switchAge_survived[tmp_age]
    test_age_rate = switchAge_rate[tmp_age]

    if SibSp != 0:
        test_Sibsp_survived = sibsp_survived
        test_Sibsp_rate = sibsp_rate
    else:
        test_Sibsp_survived = nosibsp_survived
        test_Sibsp_rate = nosibsp_rate

    if Parch != 0:
        test_Parch_survived = parch_survived
        test_Parch_rate = parch_rate
    else:
        test_Parch_survived = noparch_survived
        test_Parch_rate = noparch_rate
        

    if Fare > average_fare_titanic:
        test_Fare_survived = highfare_survived
        test_Fare_rate = highfare_rate
        
    else:#空值当作低票价
        test_Fare_survived = lowfare_survived
        test_Fare_rate = lowfare_rate
        
    
    if Embark == 'S':
        test_Embark_survived = embarkS_survived
        test_Embark_rate = embarkS_rate
        
    elif Embark == 'Q':
        test_Embark_survived = embarkQ_survived
        test_Embark_rate = embarkQ_rate
        
    elif Embark == 'C':
        test_Embark_survived = embarkC_survived
        test_Embark_rate = embarkC_rate 
    else:#空值用众数S代替
        #print("enter embark else\n")
        test_Embark_survived = embarkS_survived
        test_Embark_rate = embarkS_rate
    #这里简单起见，假设各特征条件独立   
    predict_rate = test_Pclass_survived * test_age_survived * test_Sex_survived* \
        test_Parch_survived * test_Sibsp_survived * test_Fare_survived * test_Embark_survived/  \
        (test_Pclass_rate * test_Sex_rate * test_age_rate * test_Parch_rate * test_Sibsp_rate * test_Fare_rate * test_Embark_rate)
    #print("predict_rate:\n",predict_rate)
    if predict_rate > 0.5:
        return 'survived'
    else:
        return 'nosurvived'



    
    

