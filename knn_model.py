
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv('data.csv')


features = ['Methane (%)', 'Carbon Monoxide (ppm)', 'Carbon Dioxide (%)', 
            'Hydrogen Sulfide (ppm)', 'Nitrogen (%)', 'Oxygen (%)', 
            'Sulfur Dioxide (ppm)', 'Humidity (%)', 'Temperature (°C)']

target = 'Classification'


X = df[features]
y = df[target]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


print("\nSample Predictions:")

print("Actual Danger Level vs Predicted Danger Level")

for actual, predicted in zip(y_test[:10], y_pred[:10]):
    print(f"Actual: {actual}, Predicted: {predicted}")
