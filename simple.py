

from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib


digits = datasets.load_digits()
classifier = svm.SVC(gamma=0.001, C=100.)

print "Predicting what this image is:"
print digits.data[-1]

# train using all but last sample
classifier.fit(digits.data[:-1], digits.target[:-1])

print "SVM guess: " + str(classifier.predict(digits.data[-1]))

# save model to disk
joblib.dump(classifier, "model")


