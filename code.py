#1. Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snsn 

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load Dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
 
# Convert to DataFrame
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)

# Display data
print(df.head())

# 3. Data Visualization
snsn.pairplot(df, hue='species')
plt.show()

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Model Training & Evaluation

def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
evaluate_model(knn, "K-Nearest Neighbors")

# SVM
svm = SVC(kernel='linear')
evaluate_model(svm, "Support Vector Machine")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_model(rf, "Random Forest")

# 7. Feature Importance (Random Forest)
importances = rf.feature_importances_
feature_names = iris.feature_names

plt.figure(figsize=(8, 5))
snsn.barplot(x=importances, y=feature_names)
plt.title("Feature Importance - Random Forest")
plt.show()
import numpy as np 
arr=np.array([1,2,3])
print(arr)
#creating an array from tuple
arr=np.array((1,3,2))
print(arr)
