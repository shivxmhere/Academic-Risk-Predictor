import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score

# Helpers for path finding
base_dir = '.'
if not os.path.exists('models/X_train.npy'):
    base_dir = '..'

models_dir = os.path.join(base_dir, 'models')
outputs_dir = os.path.join(base_dir, 'outputs')
os.makedirs(outputs_dir, exist_ok=True)

print("STEP 1 - Load preprocessed data")
X_train = np.load(os.path.join(models_dir, 'X_train.npy'))
X_test  = np.load(os.path.join(models_dir, 'X_test.npy'))
y_train = np.load(os.path.join(models_dir, 'y_train.npy'))
y_test  = np.load(os.path.join(models_dir, 'y_test.npy'))

print("\nSTEP 2 - Train all 5 models")
target_names = ['Low', 'Medium', 'High'] # Classes encoded as Low=0, Medium=1, High=2
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gaussian NB': GaussianNB(),
    'SVC': SVC(random_state=42)
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    trained_models[name] = model
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n--- {name} ---")
    print(f"Accuracy Score: {acc:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

print("\nSTEP 3 - Confusion matrices")
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for idx, (name, model) in enumerate(trained_models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=target_names, yticklabels=target_names)
    axes[idx].set_title(name)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, 'confusion_matrices.png'), dpi=150)
plt.close()
print("Saved confusion_matrices.png")

print("\nSTEP 4 - GridSearchCV on Random Forest")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 4],
    'class_weight': ['balanced']
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_estimator = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

print("\nSTEP 5 - 10-fold cross-validation on the best model")
cv_scores = cross_val_score(best_estimator, X_train, y_train, cv=10, scoring='f1_weighted')
print(f"CV F1 Scores Mean: {cv_scores.mean():.4f}")
print(f"CV F1 Scores Std: {cv_scores.std():.4f}")

print("\nSTEP 6 - Final evaluation of best Random Forest")
y_pred_best = best_estimator.predict(X_test)
acc_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best, average='weighted')

print(f"Accuracy: {acc_best:.4f}")
print(f"Weighted F1: {f1_best:.4f}")
print("Classification Report:")
report = classification_report(y_test, y_pred_best, target_names=target_names, output_dict=True)
print(classification_report(y_test, y_pred_best, target_names=target_names))

recall_high = report['High']['recall']
print(f"-> Recall for class 2 (High Risk): {recall_high:.4f}")

print("\nSTEP 7 - Feature importance chart")
features = ['attendance_pct', 'assignment_marks', 'internal_score', 'participation', 'prev_grade', 'study_hours', 'api']
importances = best_estimator.feature_importances_

# Sort features by importance
sorted_indices = np.argsort(importances)[::-1]
sorted_features = [features[i] for i in sorted_indices]
sorted_importances = importances[sorted_indices]

plt.figure(figsize=(10, 6))
# Remove palette warning by using hue and legend=False
sns.barplot(x=sorted_importances, y=sorted_features, hue=sorted_features, palette="viridis", legend=False)
plt.title("Feature Importances (Best Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, 'feature_importance.png'), dpi=150)
plt.close()
print("Saved feature_importance.png")

print("\nSTEP 8 - Save the final model")
model_path = os.path.join(models_dir, 'random_forest_final.pkl')
joblib.dump(best_estimator, model_path)
print("Best model saved to models/random_forest_final.pkl")
