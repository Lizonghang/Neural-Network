# coding=utf-8
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.externals import joblib


digits = datasets.load_digits()

n_samples = len(digits.images)

images_and_labels = zip(digits.images, digits.target)

for index, (image, label) in enumerate(images_and_labels[:10]):
    plt.subplot(4, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Preview: {0}'.format(label))

# classifier = svm.SVC()
# classifier.set_params(kernel='linear')
# classifier.set_params(kernel='rbf')
classifier = svm.SVC(gamma=0.001)
classifier.fit(digits.data[:n_samples / 2], digits.target[:n_samples / 2])

# joblib.dump(classifier, 'svm.pkl')
# classifier = joblib.load('svm.pkl')

predict_result = classifier.predict(digits.data[n_samples / 2:])

expect_result = digits.target[n_samples / 2:]

print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expect_result, predict_result)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(expect_result, predict_result))

images_and_predictions = zip(digits.images[n_samples / 2:], predict_result)

for index, (image, label) in enumerate(images_and_predictions[:10]):
    plt.subplot(4, 5, index + 11)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Predict: {0}'.format(label))

plt.show()
