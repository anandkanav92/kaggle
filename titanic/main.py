import pandas as pd
import graphviz
from sklearn import tree
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import time


def read_data(path="./data/"):
  return pd.read_csv(path,skip_blank_lines=True,index_col=False)

def update_gender_embarked(df=None):
  df['Sex'] = (df['Sex']=='male').astype(int)

  df.loc[df['Embarked'] == 'S', 'Embarked'] = 0
  df.loc[df['Embarked'] == 'C', 'Embarked'] = 1
  df.loc[df['Embarked'] == 'Q', 'Embarked'] = 2

  return df

TRAIN_PATH = "./data/train/train.csv"
TEST_PATH = "./data/test/test.csv"

if __name__ == '__main__':
  train_data = read_data(TRAIN_PATH)
  truth_data = train_data['Survived']
  train_data = update_gender_embarked(train_data)
  imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
  columns = ['Pclass','Sex','Age','Fare','Parch','SibSp','Embarked']
  for col in columns:
    train_data[col] = imp.fit_transform(train_data[col].values.reshape(-1,1))
  train_data = train_data.drop(columns=['Name','Ticket','Cabin'],axis=1)
  X = train_data[columns]
  y = train_data.Survived

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

  svr = GridSearchCV(DecisionTreeClassifier(max_depth='2', min_samples_split=1,min_samples_leaf=1), cv=5,
                     param_grid={"max_depth": [2, 4, 8, 16],
                                 "min_samples_leaf": [1, 5, 10],
                                 "min_samples_split": [2,5,10]})

  #print(X_train)
  t0 = time.time()
  svr.fit(X_train, y_train)
  svr_fit = time.time() - t0
  print("SVR complexity and bandwidth selected and model fitted in %.3f s"
        % svr_fit)


  # train_data = train_data.drop(columns=['Survived','Name','Ticket','Cabin'],axis=1)
  # train_data=train_data.fillna(-1)
  test_data = read_data(TEST_PATH)
  imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
  columns = ['Pclass','Sex','Age','Fare','Parch','SibSp']
  test_data = update_gender_embarked(read_data(TEST_PATH))
  for col in columns:
    test_data[col] = imp.fit_transform(test_data[col].values.reshape(-1,1))

  # test_data=test_data.fillna(-1)

  # clf = tree.DecisionTreeClassifier(max_depth=4,min_samples_split=10,min_samples_leaf=5)
  # clf = clf.fit(train_data, truth_data)
  # # dot_data = tree.export_graphviz(clf, out_file=None)
  # # graph = xdot.Source(dot_data)
  # # graph.render("Titanics")
  final = pd.DataFrame(test_data['PassengerId'], columns=['PassengerId'])
  test_data = test_data.drop(columns=['PassengerId','Name','Ticket','Cabin'],axis=1)
  final['Survived'] = svr.best_estimator_.predict(test_data).tolist()
  final.to_csv("./abc.csv",index=False, header=True)



#/Users/kanavanand/kaggle/titanic/titanic/bin:/Users/kanavanand/anaconda3/bin:/Users/kanavanand/Downloads/google-cloud-sdk/bin:/Users/kanavanand/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
