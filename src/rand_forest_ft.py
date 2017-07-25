import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

X = [[0, 0, 1],
     [0, 1, 0],
     [1, 0, 0]
    ]

Y = [[1, 2],
     [3],
     [1]
    ]

print "RAW traget data\n"
print Y

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(Y)
print "MultiLabel transformed target data\n"
print Y

clf = RandomForestClassifier(n_estimators=10)

clf.fit(X, Y)
#test = np.array([1, 0, 0])
score = clf.predict(X)
print "Output\n"
print score

score = clf.predict_proba(X)
print "Output probabilities\n"
print score

print "Features importances"
print clf.feature_importances_

print "Classes"
print mlb.classes_
