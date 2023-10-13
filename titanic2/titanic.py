# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Data preprocessing
def preprocess_data(data):
    # Fill missing values
    data["Age"].fillna(data["Age"].median(), inplace=True)
    data["Fare"].fillna(data["Fare"].median(), inplace=True)
    data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)
    
    # Encode categorical features
    label_encoders = {}
    for feature in ["Sex", "Embarked"]:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])
        label_encoders[feature] = le
    
    return data, label_encoders

train_data, label_encoders = preprocess_data(train_data)
test_data, _ = preprocess_data(test_data)

# Select features and target variable
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = train_data[features]
y = train_data["Survived"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model
model = XGBClassifier(n_estimators=1000, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the validation set
val_predictions = model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, val_predictions)
print("Validation Accuracy:", accuracy)

# Make predictions on the test set
test_features = test_data[features]
test_predictions = model.predict(test_features)

# Prepare the submission file
submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": test_predictions})
submission.to_csv("submission.csv", index=False)
