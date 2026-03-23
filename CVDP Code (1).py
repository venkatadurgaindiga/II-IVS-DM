# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd

df = pd.read_csv("datasetCVDP.csv")

print("Original Shape:", df.shape)
df.head()
# Data Cleaning 
print("\nMissing Values:")
print(df.isnull().sum())
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        df[column] = df[column].fillna(df[column].mean())
print("\nAfter Cleaning Missing Values:")
print(df.isnull().sum())
# Removing Duplicates & Outliers
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)
# Select numerical columns
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]
print("After Removing Outliers:", df.shape)
# Data Integration
# Example: Suppose we create another dataset for demonstration
df_extra = df.copy()
df_extra['extra_info'] = np.random.randint(0, 2, size=len(df_extra))
df_integrated = pd.merge(df, df_extra[['extra_info']], 
                         left_index=True, right_index=True)
print("\nIntegrated Dataset Shape:", df_integrated.shape)
# Data Transformation
le = LabelEncoder()
for column in df_integrated.columns:
    if df_integrated[column].dtype == 'object':
        df_integrated[column] = le.fit_transform(df_integrated[column])
#Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_integrated)
df_scaled = pd.DataFrame(scaled_features, 
                         columns=df_integrated.columns)
print("\nScaled Data Sample:")
print(df_scaled.head())
# Data Reduction
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
df = pd.read_csv("datasetCVDP.csv")
df = df.fillna(df.mean(numeric_only=True))
df = pd.get_dummies(df, drop_first=True)
selector = VarianceThreshold(threshold=0.01)
reduced_data = selector.fit_transform(df)
print("Original Shape:", df.shape)
print("Reduced Shape:", reduced_data.shape)
# LogisticRegression Algorithm 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("datasetCVDP.csv")

# Convert target variable
df['Heart_Disease'] = df['Heart_Disease'].map({'Yes': 1, 'No': 0})

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop('Heart_Disease', axis=1)
y = df['Heart_Disease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling (IMPORTANT for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)

# Train
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
#ROC-curve 
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# ROC calculations
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (AUC = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.show()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix - Cardiovascular Disease Prediction")
plt.show()
