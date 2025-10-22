import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df = pd.read_csv("D:/sem5/stroke_data.csv")

print(df.isnull().sum())

df['bmi'] = df['bmi'].fillna(df['bmi'].median())


le = LabelEncoder()
if 'gender' in df.columns:
    gender_col = 'gender'
elif 'sex' in df.columns:
    gender_col = 'sex'
for col in ['sex', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    df[col] = le.fit_transform(df[col])

corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

sns.countplot(x='stroke', data=df)
plt.title("Stroke Class Distribution")
plt.show()

sns.boxplot(x='stroke', y='age', data=df)
plt.title("Age vs Stroke")
plt.show()

X = df.drop('stroke', axis=1)
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Evaluation
print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr))
print("ROC AUC:", roc_auc_score(y_test, y_pred_lr))

print("\n=== Naive Bayes ===")
print(classification_report(y_test, y_pred_nb))
print("ROC AUC:", roc_auc_score(y_test, y_pred_nb))
