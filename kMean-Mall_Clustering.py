import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df=pd.read_csv('Mall_Customers.csv')

#print(df.head())
#print(df.shape)
#print(df.dtypes)
#print(df.describe())

x=df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(x)

plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title("Einkommen vs. Spending Score")
plt.show()

print(X_scaled[:5])
print("Mittelwert:", X_scaled.mean(axis=0))  # sollte ~0 sein
print("Std:", X_scaled.std(axis=0))           # sollte ~1 sein

# Berechne WCSS für K = 1 bis 10
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # Tipp: kmeans hat ein Attribut inertia_



# Trainiere K-Means mit K=5
kmeans=KMeans(n_clusters=5, random_state=42)

# fit_predict macht zwei Dinge auf einmal:
# 1. trainiert das Modell
# 2. gibt für jeden Punkt das Cluster-Label zurück
labels= kmeans.fit_predict(X_scaled)

# Füge die Labels als neue Spalte zum DataFrame hinzu
df["Cluster"]=labels

#zum Überprüfen:
print(df.head(10))
print(df['Cluster'].value_counts())  # wie viele Kunden pro Cluster?



# ---- Plot 1: Elbow Method ----
plt.plot(k_range, wcss, marker='o')
plt.xlabel('Anzahl Cluster (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()  # <-- hier abschließen

# ---- Plot 2: Cluster Scatter ----
plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df["Cluster"],
    cmap='tab10',
    s=60,
    alpha=0.8
)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Kundensegmente')
plt.colorbar(label='Cluster')
plt.show()  # <-- hier abschließen
