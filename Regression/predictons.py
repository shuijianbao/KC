#数据处理
import pandas as pd
import numpy as np
#scikitlearn包引入
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate

def data_process(path):
    """
    数据预处理，将前30条数据设为训练集，返回训练集，40组自变量（粉丝数等），40个实际票房数据
    """
    data = pd.read_csv(path)
    data.head(5)
    data1 = data["票房（万）"]
    data.drop(["电影名","Unnamed: 7","票房（万）"],axis = 1,inplace = True)
    data.insert(0,'票房（万）',data1)
    data["票房（万）"].replace('-',None,inplace = True)
    data["主演微博粉丝数"].replace('-',None,inplace = True)
    data.dropna(axis=0,how='any',inplace = True)
    data["票房（万）"] = data["票房（万）"].astype("float")
    data["主演微博粉丝数"] = data["票房（万）"].astype("float")
    data2 = data.drop(["票房（万）"],axis = 1)
    train_data = data[:30]
    test_data = data[30:]
    test_data_piaofang = test_data["票房（万）"]
    test_data.drop(["票房（万）"],axis = 1,inplace = True)
    return train_data,data2,data1

def regression(train_data):
    """
    回归分析，分别用随机森林，梯度下降迭代，支持向量机做回归，返回三个学习机
    pring 打印出三个学习机的效果，以决定我们最后要选择哪个学习机进行预测
    """
    train_np = train_data.as_matrix()
    y = train_np[:,0]
    X = train_np[:,1:]
    rf = RandomForestRegressor(n_estimators=500)
    rf.fit(X,y)
    gbr = GradientBoostingRegressor(n_estimators=500)
    gbr.fit(X,y)
    svr = SVR()
    svr.fit(X,y)
    regs = [('Random forest',rf),
        ('Gradient boosting', gbr),
        ('Support vector machine', svr)]
    for reg in regs:
        cv = cross_validate(reg[1], X, y, cv=5, n_jobs=-1)
        Train_score = np.mean(cv['train_score'])
        print reg[0], ': \n', 'Train score: ',np.mean(cv['train_score']), '\n', 'Test score: ',np.mean(cv['test_score'])
    return rf,gbr,svr

def result_file(clf,data2,data1,path):
    """
    生成预测文件
    """
    predictions = clf.predict(data2)
    result = pd.DataFrame({'predictions':predictions, 'actual':data1})
    result.to_csv(path, index=False)

if __name__ == "__main__":
    train_data,data2,data1 = data_process("/users/wangkaixi/desktop/film_data.csv")
    rf,gbr,svr = regression(train_data)
    result_file(rf,data2,data1,"/Users/wangkaixi/desktop/result1.csv")


