
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data.csv')

features = ['Methane (%)', 'Carbon Monoxide (ppm)', 'Carbon Dioxide (%)', 
            'Hydrogen Sulfide (ppm)', 'Nitrogen (%)', 'Oxygen (%)', 
            'Sulfur Dioxide (ppm)', 'Humidity (%)', 'Temperature (°C)']


target = 'Classification'


X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)


clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')


print("\nSample Predictions:")
print("Actual Danger Level vs Predicted Danger Level")
for actual, predicted in zip(y_test[:10], y_pred[:10]):
    print(f"Actual: {actual}, Predicted: {predicted}")
