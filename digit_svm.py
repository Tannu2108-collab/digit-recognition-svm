from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# Load dataset
digits = datasets.load_digits()

# Flatten data
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.3, random_state=42
)

# Model
model = svm.SVC(kernel='rbf', gamma=0.001)

# Train
model.fit(X_train, y_train)

# Predict
predicted = model.predict(X_test)

# Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, predicted))
