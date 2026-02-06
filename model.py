import pandas as pd
import numpy as np
import scipy
import scipy.sparse
import pickle

from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import vstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import classification_report

# Data Prepararing

# Load labelled datasets (train and evaluate)
df_train_label = pd.read_csv('TMDB_train.csv')
df_evaluate_label = pd.read_csv('TMDB_evaluate.csv')

# check data shape
print(df_train_label.shape)
print(df_evaluate_label.shape)

# Load the unlabelled datasets
df_train_unlabel = pd.read_csv('TMDB_unlabelled.csv')

# check data shape
print(df_train_unlabel.shape)

# Load the test datasets
df_test_data = pd.read_csv('TMDB_test.csv')

# check data shape
print(df_test_data.shape)

# Encode the 'original_language' feature in all datasets using LabelEncoder
encoder = LabelEncoder()

# Fit the encoder on the concatenated original_language lists from all datasets
encoder.fit(df_train_label['original_language'].to_list() + df_evaluate_label['original_language'].to_list() + df_test_data['original_language'].to_list() + df_train_unlabel['original_language'].to_list())

# Transform original_language feature to encoded values in all datasets
df_train_label['original_language'] = encoder.transform(df_train_label['original_language'])
df_evaluate_label['original_language'] = encoder.transform(df_evaluate_label['original_language'])
df_test_data['original_language'] = encoder.transform(df_test_data['original_language'])
df_train_unlabel['original_language'] = encoder.transform(df_train_unlabel['original_language'])

# check data shape after encoding
print(df_train_label.shape)
print(df_evaluate_label.shape)
print(df_test_data.shape)
print(df_train_unlabel.shape)

# to extract the year from a date string or integer
def extract_year(date_str):
    date_str = str(date_str)  # Ensure input is converted to string
    if '/' in date_str:   # Check if date format contains '/'
        parts = date_str.split('/')
        return int(parts[-1])  # Extract and return the last part as year
    elif '-' in date_str:  # Check if date format contains '-'
        parts = date_str.split('-')
        return int(parts[0])  # Extract and return the first part as year
    elif len(date_str) == 4 and date_str.isdigit():  # Check if date string is 4 digits and all digits
        return int(date_str)  # Return the date string as integer
    else:     # If none of the above conditions match, return the original date string
        return date_str
    
df_train_unlabel['release_year'] = df_train_unlabel['release_year'].apply(extract_year)

# remove text feature and label
X_train_label_non_text = df_train_label.drop(columns=['id','title', 'overview', 'tagline', 'production_companies', 'rate_category', 'average_rate'], axis=1)
y_train_label = df_train_label['rate_category']
y_train_label2 = df_train_label['average_rate']

X_evaluate_label_non_text = df_evaluate_label.drop(columns=['id','title', 'overview', 'tagline', 'production_companies', 'rate_category', 'average_rate'], axis=1)
y_evaluate_label = df_evaluate_label['rate_category']
y_evaluate_label2 = df_evaluate_label['average_rate']

X_test_non_text = df_test_data.drop(columns=['id','title', 'overview', 'tagline', 'production_companies'], axis=1)
X_unlabel_non_text = df_train_unlabel.drop(columns=['id','title', 'overview', 'tagline', 'production_companies'], axis=1)
X_unlabel_non_text.shape

# Train Model by Using Non-Text Features
# initialize model
zero_r_classifier = DummyClassifier(strategy="most_frequent")
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy')
lgr = LogisticRegression(max_iter=1000)
rf_classifier = RandomForestClassifier(criterion='entropy')

classifiers = {
    "Zero-R Classifier": zero_r_classifier,
    "Decision Tree": decision_tree_classifier,
    "Logistic Regression": lgr,
    "Random Forest": rf_classifier
}

# train model by using non-text feature from label dataset

# Evaluate classifiers using the selected features
print("-- Evaluate classifiers using the labeled non-text features --")
for name, classifier in classifiers.items():
    # Train classifier on selected features
    classifier.fit(X_train_label_non_text, y_train_label)
        
    # Predict
    y_pred = classifier.predict(X_evaluate_label_non_text)
        
    # Calculate accuracy
    accuracy = accuracy_score(y_evaluate_label, y_pred)
    print(f"{name} Accuracy: {accuracy:.3f}")
    # Print report with evaluation metrics
    print(classification_report(y_evaluate_label, y_pred))

# Feature Selection
# Using f_classif to do feature selection
# select most related non-text features based on labeld datasets.
from sklearn.feature_selection import SelectKBest, f_classif

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Define the number of features to select
k_values = [37, 30, 25, 20, 10, 5]

for k in k_values:
    print(f"\nTop {k} Features:")
    
    # Initialize SelectKBest with ANOVA F-value as the scoring function
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_label_non_text_selected = selector.fit_transform(X_train_label_non_text, y_train_label)
    X_evaluate_label_non_text_selected = selector.transform(X_evaluate_label_non_text)
    
    # Get the indices of the selected features
    selected_feature_indices = selector.get_support()
    
    # Get the names of the selected features
    selected_feature_names = X_train_label_non_text.columns[selected_feature_indices]
    print("Selected Features:", selected_feature_names)
    
    # Evaluate classifiers using the selected features
    for name, classifier in classifiers.items():
        # Train classifier on selected features
        classifier.fit(X_train_label_non_text_selected, y_train_label)
        
        # Predict
        y_pred = classifier.predict(X_evaluate_label_non_text_selected)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_evaluate_label, y_pred)
        print(f"{name} Accuracy: {accuracy:.3f}")

# Using self-training for models using non-text features

# Test whether the self-training model improve the performance of decision tree
X_train_both = np.vstack([X_train_label_non_text, X_unlabel_non_text])

# Create labels for unlabeled data (use -1 to represent unlabeled data)
y_both = y_train_label.to_list() + [-1] * (X_unlabel_non_text.shape[0])

# Train a semi-supervised learning model
self_training_model = SelfTrainingClassifier(decision_tree_classifier, criterion = 'k_best', k_best=1000)
self_training_model.fit(X_train_both, y_both)

y_pred_eval = self_training_model.predict(X_evaluate_label_non_text)

# Evaluate the performance
print(classification_report(y_evaluate_label, y_pred_eval))

# test prediction save in csv
y_pred = self_training_model.predict(X_test_non_text)

df_test_data['rate_category'] = y_pred

predictions = df_test_data[['id', 'rate_category']]
predictions.to_csv('dt_all_non_text_predictions(label + unlabel).csv', index=False)

# Test whether the self-training model improve the performance of logistic regression
X_train_both = np.vstack([X_train_label_non_text, X_unlabel_non_text])

# Create labels for unlabeled data (use -1 to represent unlabeled data)
y_both = y_train_label.to_list() + [-1] * (X_unlabel_non_text.shape[0])


# Train a semi-supervised learning model

self_training_model = SelfTrainingClassifier(lgr, criterion = 'k_best', k_best=1000)
self_training_model.fit(X_train_both, y_both)

y_pred_eval = self_training_model.predict(X_evaluate_label_non_text)


# Evaluate the performance
print(classification_report(y_evaluate_label, y_pred_eval))

# Test whether the self-training model improve the performance of random forest
X_train_both = np.vstack([X_train_label_non_text, X_unlabel_non_text])

# Create labels for unlabeled data (use -1 to represent unlabeled data)
y_both = y_train_label.to_list() + [-1] * (X_unlabel_non_text.shape[0])


# Train a semi-supervised learning model

self_training_model = SelfTrainingClassifier(rf_classifier, criterion = 'k_best', k_best=1000)
self_training_model.fit(X_train_both, y_both)

y_pred_eval = self_training_model.predict(X_evaluate_label_non_text)


# Evaluate the performance
print(classification_report(y_evaluate_label, y_pred_eval))

# test prediction save in csv
y_pred = self_training_model.predict(X_test_non_text)

df_test_data['rate_category'] = y_pred

predictions = df_test_data[['id', 'rate_category']]
predictions.to_csv('rf_all_non_text_predictions(label + unlabel).csv', index=False)

### Hypothesis: Data imbalance affects the performance of model prediction. Does balancing the data improve the model performance?
### Experiment: Use the class_balanced method to balance the data and compare the accuracy before and after balancing (Use the original 37 features for the experiment)

from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_label), y=y_train_label)

# train random forest using balanced data
balanced_rf_model = RandomForestClassifier(criterion='entropy', class_weight=dict(zip(np.unique(y_train_label), class_weights)))

#-----

X_train_both = np.vstack([X_train_label_non_text, X_unlabel_non_text])
print(X_train_both.shape)

# Create labels for unlabeled data (use -1 to represent unlabeled data)
y_both = y_train_label.to_list() + [-1] * (X_unlabel_non_text.shape[0])


# Train a semi-supervised learning model

self_training_model = SelfTrainingClassifier(balanced_rf_model, criterion = 'k_best', k_best=1000)
self_training_model.fit(X_train_both, y_both)

y_pred_eval = self_training_model.predict(X_evaluate_label_non_text)


# Evaluate the performance
print(classification_report(y_evaluate_label, y_pred_eval))

# test prediction save in csv
y_pred = self_training_model.predict(X_test_non_text)

df_test_data['rate_category'] = y_pred

predictions = df_test_data[['id', 'rate_category']]
predictions.to_csv('rf_all_non_text_balanced(label + unlabel).csv', index=False)

# use semi-supervised learning to improve performance
# Combine labeled and unlabeled data (non-text)

# random forest
X_train_both = np.vstack([X_train_non_text, X_unlabel_non_text])
print(X_train_both.shape)

# Create labels for unlabeled data (use -1 to represent unlabeled data)
y_both = y_train_label.to_list() + [-1] * (X_unlabel_non_text.shape[0])


# Train a semi-supervised learning model

self_training_model = SelfTrainingClassifier(rf_classifier, criterion = 'k_best', k_best=1000)
self_training_model.fit(X_train_both, y_both)

y_pred = self_training_model.predict(X_test_non_text)

df_test_data['rate_category'] = y_pred

# Evaluate the performance
print(classification_report(y_evaluate_label, y_pred))

predictions = df_test_data[['id', 'rate_category']]
predictions.to_csv('rf_non_text_predictions(label + unlabel).csv', index=False)

### Train Model by Using Text Features

# load text feature
# bow
train_concat_bow = scipy.sparse.load_npz('TMDB_text_features_bow/train_concat_bow.npz')
eval_concat_bow = scipy.sparse.load_npz('TMDB_text_features_bow/eval_concat_bow.npz')
test_concat_bow = scipy.sparse.load_npz('TMDB_text_features_bow/test_concat_bow.npz')
unlabelled_concat_bow = scipy.sparse.load_npz('TMDB_text_features_bow/unlabelled_concat_bow.npz')

print(train_concat_bow.shape)
print(eval_concat_bow.shape)
print(test_concat_bow.shape)
print(unlabelled_concat_bow.shape)

# test the accuracy of models when using text features (bow)
print("-- Evaluate classifiers using the labeled text features(bow) --")
for name, classifier in classifiers.items():
    # Train classifier on selected features
    classifier.fit(train_concat_bow.toarray(), y_train_label)
        
    # Predict
    y_pred = classifier.predict(eval_concat_bow.toarray())
        
    # Calculate accuracy
    accuracy = accuracy_score(y_evaluate_label, y_pred)
    print(f"{name} Accuracy: {accuracy:.3f}")
    print(classification_report(y_evaluate_label, y_pred))

# tfidf
train_concat_tfidf = scipy.sparse.load_npz('TMDB_text_features_tfidf/train_concat_tfidf.npz')
eval_concat_tfidf = scipy.sparse.load_npz('TMDB_text_features_tfidf/eval_concat_tfidf.npz')
test_concat_tfidf = scipy.sparse.load_npz('TMDB_text_features_tfidf/test_concat_tfidf.npz')
unlabelled_concat_tfidf = scipy.sparse.load_npz('TMDB_text_features_tfidf/unlabelled_concat_tfidf.npz')

print(train_concat_tfidf.shape)
print(eval_concat_tfidf.shape)
print(test_concat_tfidf.shape)
print(unlabelled_concat_tfidf.shape)

# test the accuracy of models when using text features (tfidf)
print("-- Evaluate classifiers using the labeled text features(tfidf) --")
for name, classifier in classifiers.items():
    # Train classifier on selected features
    classifier.fit(train_concat_tfidf.toarray(), y_train_label)
        
    # Predict
    y_pred = classifier.predict(eval_concat_tfidf.toarray())
        
    # Calculate accuracy
    accuracy = accuracy_score(y_evaluate_label, y_pred)
    print(f"{name} Accuracy: {accuracy:.3f}")
    print(classification_report(y_evaluate_label, y_pred))

### Using self-training for models using text features
# Combine unlabel text and label text（Random Forest）(bow)
all_text_data = vstack([train_concat_bow, unlabelled_concat_bow])

# Create labels for unlabeled data (we'll use -1 to represent unlabeled data)
y_text = y_train_label.to_list() + [-1] * (unlabelled_concat_bow.shape[0])    
        
self_training_model = SelfTrainingClassifier(rf_classifier, criterion = 'k_best', k_best=1000)
self_training_model.fit(all_text_data, y_text)

y_pred = self_training_model.predict(eval_concat_bow)

# Evaluate the performance
print(classification_report(y_evaluate_label, y_pred))

# Predict
y_pred = self_training_model.predict(test_concat_bow)

df_test_data['rate_category'] = y_pred

# Save to a CSV file
predictions = df_test_data[['id', 'rate_category']]
predictions.to_csv('rf_text_predictions(label unlabel)(bow).csv', index=False)

# Combine unlabel text and label text（Decision Tree）(bow)

DS = DecisionTreeClassifier(criterion='entropy',splitter='random')

DS.fit(train_concat_bow, y_train_label)

all_text_data = vstack([train_concat_bow, unlabelled_concat_bow])

# Create labels for unlabeled data (we'll use -1 to represent unlabeled data)
y_text = y_train_label.to_list() + [-1] * (unlabelled_concat_bow.shape[0])

self_training_model = SelfTrainingClassifier(DS, criterion = 'k_best', k_best=1000)
self_training_model.fit(all_text_data, y_text)

y_pred = self_training_model.predict(eval_concat_bow)

# Evaluate the performance
print(classification_report(y_evaluate_label, y_pred))

#----
y_pred_DT_text = self_training_model.predict(test_concat_bow)

df_test_data['rate_category'] = y_pred_DT_text

# Save to a CSV file

predictions = df_test_data[['id', 'rate_category']]
predictions.to_csv('dt_text_predictions(label unlabel)(bow).csv', index=False)

# Combine unlabel text and label text（Logistic Regression）(bow)
self_training_model = SelfTrainingClassifier(lgr, criterion = 'k_best', k_best=1000)
self_training_model.fit(all_text_data, y_text)

y_pred = self_training_model.predict(eval_concat_bow)

# Evaluate the performance
print(classification_report(y_evaluate_label, y_pred))

#----
y_pred_lgr_text = self_training_model.predict(test_concat_bow)

df_test_data['rate_category'] = y_pred_lgr_text

# Save to a CSV file

predictions = df_test_data[['id', 'rate_category']]
predictions.to_csv('lgr_text_predictions(label unlabel)(bow).csv', index=False)

# Combine unlabel text and label text（Random Forest）(tfidf)
all_text_data_2 = vstack([train_concat_tfidf, unlabelled_concat_tfidf])

# Create labels for unlabeled data (we'll use -1 to represent unlabeled data)
y_text_2 = y_train_label.to_list() + [-1] * (unlabelled_concat_tfidf.shape[0])

#decision_tree_classifier = DecisionTreeClassifier(criterion='entropy',splitter='random')      
        
self_training_model = SelfTrainingClassifier(rf_classifier, criterion = 'k_best', k_best=1000)
self_training_model.fit(all_text_data_2, y_text_2)

y_pred = self_training_model.predict(eval_concat_tfidf)

# Evaluate the performance
print(classification_report(y_evaluate_label, y_pred))

#----
y_pred_rf_2 = self_training_model.predict(test_concat_tfidf)

df_test_data['rate_category'] = y_pred_rf_2

# Save to a CSV file

predictions = df_test_data[['id', 'rate_category']]
predictions.to_csv('rf_text_predictions(label unlabel)(tfidf).csv', index=False)

# Combine unlabel text and label text（Decision Tree）(tfidf)

self_training_model = SelfTrainingClassifier(decision_tree_classifier, criterion = 'k_best', k_best=1000)
self_training_model.fit(all_text_data_2, y_text_2)

y_pred = self_training_model.predict(eval_concat_tfidf)

# Evaluate the performance
print(classification_report(y_evaluate_label, y_pred))

#----
y_pred_dt_2 = self_training_model.predict(test_concat_tfidf)

df_test_data['rate_category'] = y_pred_dt_2

# Save to a CSV file

predictions = df_test_data[['id', 'rate_category']]
predictions.to_csv('dt_text_predictions(label unlabel)(tfidf).csv', index=False)

# Combine unlabel text and label text（Logistic Regression）(tfidf)
self_training_model = SelfTrainingClassifier(lgr, criterion = 'k_best', k_best=1000)
self_training_model.fit(all_text_data_2, y_text_2)

y_pred = self_training_model.predict(eval_concat_tfidf)

# Evaluate the performance
print(classification_report(y_evaluate_label, y_pred))

#----
y_pred_lgr_2 = self_training_model.predict(test_concat_tfidf)

df_test_data['rate_category'] = y_pred_lgr_2

# Save to a CSV file

predictions = df_test_data[['id', 'rate_category']]
predictions.to_csv('lgr_text_predictions(label unlabel)(tfidf).csv', index=False)

# Just test whether or not the performance of model is impacted by considering text and non-text features
# concat text and non-text feature
X_train_label_tfidf = np.concatenate((train_concat_tfidf.toarray(), X_train_label_non_text.to_numpy()), axis=1)
X_evaluate_label_tfidf = np.concatenate((eval_concat_tfidf.toarray(), X_evaluate_label_non_text.to_numpy()), axis=1)
X_test_tfidf = np.concatenate((test_concat_tfidf.toarray(), X_test_non_text.to_numpy()), axis=1)

print(X_train_label_tfidf.shape)
print(X_evaluate_label_tfidf.shape)
print(X_test_tfidf.shape)

print("-- Evaluate classifiers using the labeled text features(text and non-text) --")
for name, classifier in classifiers.items():
    # Train classifier on selected features
    classifier.fit(X_train_label_tfidf, y_train_label)
        
    # Predict
    y_pred = classifier.predict(X_evaluate_label_tfidf)
        
    # Calculate accuracy
    accuracy = accuracy_score(y_evaluate_label, y_pred)
    print(f"{name} Accuracy: {accuracy:.3f}")
    print(classification_report(y_evaluate_label, y_pred))

## Evaluation
# Draw a heat map to show the performance of model when choose different model.
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Dictionary to store accuracies for each model
accuracies = {name: [] for name in classifiers}

for k in k_values:
    #print(f"\nTop {k} Features:")
    
    # Initialize SelectKBest with ANOVA F-value as the scoring function
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_label_non_text_selected = selector.fit_transform(X_train_label_non_text, y_train_label)
    X_evaluate_label_non_text_selected = selector.transform(X_evaluate_label_non_text)
    
    # Get the indices of the selected features
    selected_feature_indices = selector.get_support()
    
    # Get the names of the selected features
    selected_feature_names = X_train_label_non_text.columns[selected_feature_indices]
    #print("Selected Features:", selected_feature_names)
    
    # Evaluate classifiers using the selected features
    for name, classifier in classifiers.items():
        # Train classifier on selected features
        classifier.fit(X_train_label_non_text_selected, y_train_label)
        
        # Predict
        y_pred = classifier.predict(X_evaluate_label_non_text_selected)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_evaluate_label, y_pred)
        accuracies[name].append(accuracy)

# Convert accuracies dictionary to DataFrame for heatmap
accuracies_df = pd.DataFrame(accuracies, index=k_values)

# Plotting heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(accuracies_df, annot=True, cmap="coolwarm", fmt=".3f")
plt.title('Accuracy Changes with Different Non-text Features')
plt.xlabel('Classifier')
plt.ylabel('Number of Top Features')
plt.yticks(rotation=0)
plt.show()

# Draw confusion matrix to compare the accuracy with or without balanced datasets
# Compute confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_evaluate_label, y_pred_eval)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix of self-training with unbalanced datasets')
plt.show()

from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_label), y=y_train_label)

# train rf using balanced data
balanced_rf_model = RandomForestClassifier(criterion='entropy', class_weight=dict(zip(np.unique(y_train_label), class_weights)))

#-----

X_train_both = np.vstack([X_train_label_non_text, X_unlabel_non_text])
print(X_train_both.shape)

# Create labels for unlabeled data (use -1 to represent unlabeled data)
y_both = y_train_label.to_list() + [-1] * (X_unlabel_non_text.shape[0])


# Train a semi-supervised learning model

self_training_model = SelfTrainingClassifier(balanced_rf_model, criterion = 'k_best', k_best=1000)
self_training_model.fit(X_train_both, y_both)

y_pred_eval = self_training_model.predict(X_evaluate_label_non_text)

# Compute confusion matrix
cm = confusion_matrix(y_evaluate_label, y_pred_eval)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix of self-training with balanced datasets')
plt.show()

### Learning curve to check overfitting and underfitting
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy')

train_scores = []
val_scores = []
train_sizes = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

for train_size in train_sizes:
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train_label_non_text, y_train_label, train_size=train_size, random_state=42)
    decision_tree_classifier.fit(X_train_subset, y_train_subset)
    y_train_pred = decision_tree_classifier.predict(X_train_subset)
    y_val_pred = decision_tree_classifier.predict(X_evaluate_label_non_text)
    train_scores.append(accuracy_score(y_train_subset, y_train_pred))
    val_scores.append(accuracy_score(y_evaluate_label, y_val_pred))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, 'o-', color='r', label='Training Score')
plt.plot(train_sizes, val_scores, 'o-', color='g', label='Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve of Decision Tree')
plt.legend(loc='best')
plt.show()

lgr = LogisticRegression(max_iter=1000)

train_scores = []
val_scores = []
train_sizes = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

for train_size in train_sizes:
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train_label_non_text, y_train_label, train_size=train_size, random_state=42)
    lgr.fit(X_train_subset, y_train_subset)
    y_train_pred = lgr.predict(X_train_subset)
    y_val_pred = lgr.predict(X_evaluate_label_non_text)
    train_scores.append(accuracy_score(y_train_subset, y_train_pred))
    val_scores.append(accuracy_score(y_evaluate_label, y_val_pred))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, 'o-', color='r', label='Training Score')
plt.plot(train_sizes, val_scores, 'o-', color='g', label='Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve of Logistic Regression')
plt.legend(loc='best')
plt.show()

rf_classifier = RandomForestClassifier(criterion='entropy')

train_scores = []
val_scores = []
train_sizes = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

for train_size in train_sizes:
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train_label_non_text, y_train_label, train_size=train_size, random_state=42)
    rf_classifier.fit(X_train_subset, y_train_subset)
    y_train_pred = rf_classifier.predict(X_train_subset)
    y_val_pred = rf_classifier.predict(X_evaluate_label_non_text)
    train_scores.append(accuracy_score(y_train_subset, y_train_pred))
    val_scores.append(accuracy_score(y_evaluate_label, y_val_pred))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, 'o-', color='r', label='Training Score')
plt.plot(train_sizes, val_scores, 'o-', color='g', label='Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve of Random Forest')
plt.legend(loc='best')
plt.show()

# Train a semi-supervised learning model

self_training_model = SelfTrainingClassifier(rf_classifier, criterion = 'k_best', k_best=1000)
self_training_model.fit(X_train_both, y_both)

train_scores = []
val_scores = []
train_sizes = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

for train_size in train_sizes:
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train_label_non_text, y_train_label, train_size=train_size, random_state=42)
    self_training_model.fit(X_train_subset, y_train_subset)
    y_train_pred = self_training_model.predict(X_train_subset)
    y_val_pred = self_training_model.predict(X_evaluate_label_non_text)
    train_scores.append(accuracy_score(y_train_subset, y_train_pred))
    val_scores.append(accuracy_score(y_evaluate_label, y_val_pred))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, 'o-', color='r', label='Training Score')
plt.plot(train_sizes, val_scores, 'o-', color='g', label='Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve of Random Forest with self-training')
plt.legend(loc='best')
plt.show()

### Draw a bar chart comparison of model precision with and without Self-Training (non-text)
# Define the model names
model_names = ['Decision Tree', 'Logistic Regression', 'Random Forest']

# Define the accuracies without self-training
accuracies_without_self_training = [0.65, 0.18, 0.69]

# Define the accuracies with self-training
accuracies_with_self_training = [0.65, 0.19, 0.70]

# Set the width of the bars
bar_width = 0.2

# Set the position of the bars on the x-axis
r1 = range(len(model_names))
r2 = [x + bar_width for x in r1]

# Plotting the bars
plt.figure(figsize=(10, 6))
plt.bar(r1, accuracies_without_self_training, color='red', width=bar_width, label='Without Self-Training')
plt.bar(r2, accuracies_with_self_training, color='green', width=bar_width, label='With Self-Training')

# Adding labels
plt.xlabel('Models', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.xticks([r + bar_width/3 for r in r1], model_names)
plt.title('Comparison of Model Precision with and without Self-Training (non-text)', fontweight='bold')
plt.legend()

# Show plot
plt.show()