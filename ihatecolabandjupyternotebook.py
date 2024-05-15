from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
import sklearn
import pandas as pd

data = pd.read_csv("C://Users\lok20\Downloads\generated_data_hist_matched_bich.csv")
feat = data.iloc[:,:-1]
target = data.iloc[:,-1]

print(target.values.shape, target.values.min(), target.values.max())

param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter. The strength of the regularization is inversely proportional to C.
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Specifies the kernel type to be used in the algorithm.
    'degree': [2, 3, 4],  # Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
    'class_weight': [None, 'balanced']  # Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one.
}

tr_f, te_f, tr_t, te_t = train_test_split(feat, target)

model = SVC(verbose=True)
model.fit(tr_f, tr_t)
p = model.predict(te_f)
print(sklearn.metrics.accuracy_score(p, te_t))

#GS = GridSearchCV(model, param_grid, scoring='accuracy')
#GS.fit(feat, target)