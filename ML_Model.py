import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier

# Load data from the CSV file
data = pd.read_csv('news.csv')

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data['Summary'], data['Topic'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

# Perform grid search for the SVM classifier
svm_parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10]}
svm_classifier = svm.SVC()
svm_grid_search = GridSearchCV(svm_classifier, svm_parameters)
svm_grid_search.fit(train_vectors, train_labels)

# Train a Support Vector Machine (SVM) classifier on the training data using the best parameters from grid search
clf_svm = svm.SVC(**svm_grid_search.best_params_)
clf_svm.fit(train_vectors, train_labels)

# Perform grid search for the deep learning classifier
dl_parameters = {'hidden_layer_sizes':[(50,50,50), (50,100,50), (100,)],
                 'activation': ['tanh', 'relu'],
                 'solver': ['sgd', 'adam'],
                 'alpha': [0.0001, 0.05],
                 'learning_rate': ['constant','adaptive']}
dl_classifier = MLPClassifier(max_iter=1000)
dl_grid_search = GridSearchCV(dl_classifier, dl_parameters)
dl_grid_search.fit(train_vectors, train_labels)

# Train a deep learning classifier on the training data using the best parameters from grid search
clf_dl = MLPClassifier(**dl_grid_search.best_params_, max_iter=1000)
clf_dl.fit(train_vectors, train_labels)

# Ensemble the two classifiers using a voting classifier
voting_classifier = VotingClassifier(estimators=[('svm', clf_svm), ('dl', clf_dl)], voting='hard')
voting_classifier.fit(train_vectors, train_labels)

# Predict the labels for the testing data using the ensemble classifier
pred_labels = voting_classifier.predict(test_vectors)

# Print the classification report and export to a text file
report = classification_report(test_labels, pred_labels)
print(report)
with open('ensemble_classification_report.txt', 'w') as f:
    f.write(report)
