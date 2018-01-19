#泰坦尼克问题： 根据所有用户的特征判断生存与否


#1.数据载入
import pandas as pd
data_train = pd.read_table("/users/wangkaixi/desktop/train.txt",delimiter = ',')
#df_train.info()#对数据进行基本查看，发现Age和Cabin两个变量有缺失值
data_train.describe()
data_train.describe(include =['O'])

#2.数据可视化从而进行特征变量的挑选
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams['font.family'] = ['simhei']
fig = plt.figure()
fig.set(alpha = 0.2)

#查看总的获救情况
plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind = 'bar',figsize = (15,9))
plt.ylabel("人数")
plt.title("获救情况（1代表获救）",size = 10)


#查看乘客分布
plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind = 'bar')
plt.ylabel("人数")
plt.xlabel("几登舱")
plt.title("乘客分布")


#按年龄分布看乘客获救情况（年龄是一个大范围的离散变量）
plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived,data_train.Age)
plt.title("按年龄分布看乘客获救情况")
plt.ylabel("人数")


#看各等级舱的年龄分布
plt.subplot2grid((2,3),(1,0),colspan = 2)
data_train.Age[data_train.Pclass == 1].plot(kind = 'kde')
data_train.Age[data_train.Pclass == 2].plot(kind = 'kde')
data_train.Age[data_train.Pclass == 3].plot(kind = 'kde')
plt.xlabel('年龄')
plt.ylabel("舱位等级百分比")
plt.title("年龄等级概率密度分布图")
plt.legend(('1等舱','2等舱','3等舱'),loc = 'best')

#看各登船口岸上船人数
plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind = 'bar')
plt.ylabel("人数")
plt.xlabel("登船口岸")
plt.title("各登船口岸上船人数分布图")
#plt.show()

##看看各舱位等级乘客，性别，登船船口，分布情况

survived_class1 = data_train.Survived[data_train.Pclass == 1].value_counts()
survived_class2 = data_train.Survived[data_train.Pclass == 2].value_counts()
survived_class3 = data_train.Survived[data_train.Pclass == 3].value_counts()
df = pd.DataFrame({'Class1':survived_class1,'Class2':survived_class2,"Class3":survived_class3})
df.plot(kind = 'bar',stacked = True)
#plt.show()


survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({'male':survived_m,'female':survived_f})
df.plot(kind = 'bar',stacked = True)
#plt.show()

survived = data_train.Embarked[data_train.Survived == 1].value_counts()
unsurvived = data_train.Embarked[data_train.Survived == 0].value_counts()
df = pd.DataFrame({'survived':survived,'unsurvived':unsurvived})
df.plot(kind = 'bar',stacked = True)
#plt.show()


#3.特征工程

#最棘手的两个 通过randomforestregressor 回归年龄 通过null赋值给cabin

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
data_train = pd.read_table("/users/wangkaixi/desktop/train.txt",delimiter = ',')

def set_missing_age(df):

	df_age = df[['Age','SibSp','Parch','Pclass','Fare']]

	known_age = df_age[df_age.Age.notnull()].as_matrix()
	unknown_age = df_age[df_age.Age.isnull()].as_matrix()

	y = known_age[:,0]

	X = known_age[:,1:]

	rtf = RandomForestRegressor(random_state = 0, n_estimators = 2000, n_jobs = -1)
	rtf.fit(X,y)

	predictedAges = rtf.predict(unknown_age[:,1:])
	df.loc[(df.Age.isnull()),'Age'] = predictedAges

	return df,rtf


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull(),'Cabin')] = 'Yes'
    df.loc[(df.Cabin.isnull(),'Cabin')] = 'No'
    return df

data_train,rtf = set_missing_age(data_train)
data_train = set_Cabin_type(data_train) 

#data_train.head(5)

#使用 Pandas 的get_dummy 来将我们的数据因子化

dummies_Cabin = pd.get_dummies(data_train['Cabin'],prefix = 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'],prefix = 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'],prefix = 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'],prefix = 'Pclass')

df = pd.concat([data_train,dummies_Pclass,dummies_Sex,dummies_Embarked,dummies_Cabin],axis = 1)
df.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],inplace = True,axis = 1)

df.head(5)

#使用 特征缩放
import sklearn.preprocessing as preprocessing

scaler = preprocessing.StandardScaler()
age_scaler = scaler.fit(df['Age'])
fare_scaler = scaler.fit(df['Fare'])
df['Age_scaler'] = scaler.fit_transform(df['Age'],age_scaler)
df['Fare_scaler'] = scaler.fit_transform(df['Fare'],fare_scaler)


#4.模型选择

#我们选择逻辑回归模型
from sklearn import linear_model

df_train = df.filter(regex = 'Survived|Age|SibSp|Parch|Cabin.*|Embarked.*|Sex.*|P_class.*')
train_np = df_train.as_matrix()

y = train_np[:,0]

X = train_np[:,1:]

clf = linear_model.LogisticRegression(C=1.0, penalty = 'l1', tol = 1e-6)
clf.fit(X,y)

clf


#5.对test数据进行处理
data_test = pd.read_table('/users/wangkaixi/desktop/test.txt',delimiter = ',')

data_test = pd.read_table('/users/wangkaixi/desktop/test.txt',delimiter = ',')
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
data_test,rtf = set_missing_ages(data_test)
data_test = set_Cabin_type(data_test)


dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'], age_scaler)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'], fare_scaler)
df_test.head(5)

#6.拟合结果
test = df_test.filter(regex = 'Age_.*|SibSp|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})
result.to_csv('/users/wangkaixi/desktop/result.csv',index = False)



