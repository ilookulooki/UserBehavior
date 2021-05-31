from sklearn.metrics import roc_auc_score
import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier

import matplotlib.pyplot as plt
import itertools
% matplotlib
inline
login_day = pd.read_csv(r'C:\Users\Administrator\Desktop\题目B\数据\login_day.csv', sep=',')
result = pd.read_csv(r'C:\Users\Administrator\Desktop\题目B\数据\result.csv', sep=',')
user_info = pd.read_csv(r'C:\Users\Administrator\Desktop\题目B\数据\user_info.csv', sep=',')
visit_info = pd.read_csv(r'C:\Users\Administrator\Desktop\题目B\数据\visit_info.csv', sep=',')

new_features = pd.merge(user_info, login_day, on='user_id')
new_features = pd.merge(new_features, visit_info, on='user_id')
new_features = pd.merge(new_features, result, on='user_id', how='left')
new_features['result'] = new_features['result'].fillna(0)


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_logloss(model):
    results = model.evals_result_
    #     print(results)
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    #     print(epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    #     print(results['validation_1']['logloss'])
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGDboost Log Loss')
    plt.show()
    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoostClassification Error')
    plt.show()


def label_encoder(data):
    labelencoder = LabelEncoder()
    for col in data.columns:
        data[col] = labelencoder.fit_transform(data[col])
    return data


def split_category(data, columns):
    print(data)
    print(columns)
    cat_data = data[columns]
    rest_data = data.drop(columns, axis=1)
    return rest_data, cat_data


def one_hot_cat(data):
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data, columns=[data.name])
    out = pd.DataFrame([])
    for col in data.columns:
        one_hot_cols = pd.get_dummies(data[col], prefix=col)
        out = pd.concat([out, one_hot_cols], axis=1)
    out.set_index(data.index)
    return out


del new_features['city_num']
del new_features['user_id']
del new_features['app_num']


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Compute confusion matrix

data  = new_features.copy()
data =data[['coupon', 'distance_day', 'course_order_num', 'login_diff_time',
       'login_day', 'model_num', 'age_month', 'login_time', 'study_num',
       'camp_num', 'video_read', 'learn_num', 'chinese_subscribe_num',
       'main_home2', 'finish_num']]

y = new_features['result'].values
X = data

categorical_cols = ['chinese_subscribe_num', 'study_num']
X[categorical_cols] = label_encoder(X[categorical_cols])


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)


X, X_cat = split_category(X, categorical_cols)
X_cat_one_hot_cols = one_hot_cat(X_cat)
X = pd.concat([X, X_cat_one_hot_cols], axis=1, ignore_index=True)
scaler = preprocessing.Normalizer().fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 666)


model1=XGBClassifier(
    objective='binary:logistic',
    disable_default_eval_metric=1,
    n_estimators=300)


train = [X_train, y_train]
eval = [X_test, y_test]
model2=model1.fit(X_train,y_train,
                  eval_metric=['logloss','auc','error'],eval_set=[train,eval])


y_pred = model2.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score,f1_score
print(f'auc score is {accuracy_score(y_test, y_pred)}')
print(sklearn.metrics.confusion_matrix(y_test, y_pred))
print(sklearn.metrics.classification_report(y_test, y_pred, digits=3))
f1 = f1_score(y_pred, y_test, average='macro')
print(f'f1_score  is {f1}')


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0,n_estimators = 500)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print("Test set accuracy score: {:.5f}".format(accuracy_score(y_test, rfc_pred,)))
print(classification_report(y_test, rfc_pred))

lr = LogisticRegression(C= 10, penalty= 'l2',class_weight='auto')
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("Test set accuracy score: {:.5f}".format(accuracy_score(y_test, lr_pred,)))
print(classification_report(y_test, lr_pred))


from mlxtend.classifier import StackingClassifier

sclf = StackingClassifier(classifiers=[rfc,model1],
                          meta_classifier=lr)
sclf.fit(X_train, y_train)
sclf.score(X_train, y_train)


sclf_pred = sclf.predict(X_test)

print("Test set accuracy score: {:.5f}".format(accuracy_score(y_test, sclf_pred,)))
print(classification_report(y_test, sclf_pred))