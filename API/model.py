import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

import pickle


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler
import random as random 
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import math

df = pd.read_csv("df2.csv")

def num_cat(df): 
    cat_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

df_encoding = num_cat(df)

df_encoding = df_encoding.drop(columns=['ApplicantIncome', 'CoapplicantIncome'])

X = df_encoding.drop('Loan_Status_Y', axis=1)
y = df_encoding['Loan_Status_Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=123)
rf.fit(X_train, y_train)
y_proba_rf = rf.predict_proba(X_test)[:,1]
y_pred_rf = (y_proba_rf >= 0.5).astype(int)
print('------------------Random Forest Classifier------------------')
print("\nConfusion matrix :\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification report :\n", classification_report(y_test, y_pred_rf))
print(f"\nAUC Score : {roc_auc_score(y_test, y_pred_rf):.5f}")


param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['auto', 'sqrt', 'log2', None],
}


random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=50, cv=5, scoring='roc_auc', n_jobs=-1,   random_state=123 )
random_search.fit(X_train, y_train)

print("Best parameters :", random_search.best_params_)
print("Best AUC score :", random_search.best_score_)

best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)  


y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)



print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
print("\nRapport de classification :\n", classification_report(y_test, y_pred))
print("\nAUC Score :", roc_auc_score(y_test, y_proba))

plt.figure(figsize=(10,6))
importances = best_model.feature_importances_
feature_names = X_train.columns

feat_importances = pd.Series(importances, index=feature_names)
top_features = feat_importances.sort_values(ascending=False).head(5).index
print("Top features :", list(top_features))

feat_importances_sorted = feat_importances.sort_values(ascending=False)
X_train_reduced = X_train[top_features]
X_test_reduced = X_test[top_features]


feat_importances_sorted.plot(kind='bar', color='red')

plt.title("Feature Importances")
plt.ylabel("Importance")
plt.xlabel("Features")
plt.tight_layout()
plt.show()

reduced_rf = RandomForestClassifier(
    max_depth=best_model.max_depth,
    max_features=best_model.max_features,
    min_samples_leaf=best_model.min_samples_leaf,
    min_samples_split=best_model.min_samples_split,
    n_estimators=best_model.n_estimators,
    class_weight='balanced',
    random_state=123
)

reduced_rf.fit(X_train_reduced, y_train)


with open("model.pkl", "wb") as f:
    pickle.dump(reduced_rf, f)


# # Cible binaire
# df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# # Nettoyage
# df = df.dropna(subset=["age", "Gender", "TotalIncome", "Loan_Status"])

# # Encoder "H" et "F" en "Homme" et "Femme"
# df['Gender'] = df['Gender'].replace({'Male': 'Homme', 'Female': 'Femme'})

# X = df[["age", "Gender", "TotalIncome"]]
# y = df["Loan_Status"]

# preprocessor = ColumnTransformer([
#     ("Gender", OneHotEncoder(drop="if_binary"), ["Gender"]),
#     ("num", StandardScaler(), ["age", "TotalIncome"])
# ])

# pipeline = make_pipeline(preprocessor, LogisticRegression(max_iter=1000, class_weight="balanced"))

# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=123)
# pipeline.fit(X_train, y_train)
# df_input = pd.DataFrame([{
#         "age": 85,
#         "Gender": "Femme",
#         "TotalIncome": 10
#     }])

# print(pipeline.predict(df_input)[0])
# # Sauvegarde
# with open("model.pkl", "wb") as f:
#     pickle.dump(pipeline, f)

