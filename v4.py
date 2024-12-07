import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

# Veri setini yükleme
data = pd.read_csv('C:\\Users\\LENOVO\\Desktop\\TDD and BDD Comparison\\tdd_bdd_comparison_dataset.csv')

# İlk 5 satırı görüntüleme
print("Veri Seti İlk 5 Satır:\n", data.head())

# Veri türlerini kontrol etme
print("\nVeri Türleri:\n", data.dtypes)


# Boxplot için Geliştirme Süresi (Development Time) ve Yöntem Karşılaştırması
sns.boxplot(x='Methodology', y='Development_Time_Hours', data=data)
plt.title("Development Time Comparison by Methodology")
plt.savefig("development_time_comparison.png")
plt.show()

# Boxplot için Test Coverage ve Yöntem Karşılaştırması
sns.boxplot(x='Methodology', y='Test_Coverage_Percentage', data=data)
plt.title("Test Coverage Comparison by Methodology")
plt.savefig("test_coverage_comparison.png")
plt.show()
# Boxplot için Gelişim Aşamasındaki ve Üretim Aşamasındaki Hatalar
plt.figure(figsize=(10,6))

# Gelişimdeki hatalar
plt.subplot(1, 2, 1)
sns.boxplot(x='Methodology', y='Bugs_Detected_Development', data=data)
plt.title("Bugs Detected During Development")

# Üretimdeki hatalar
plt.subplot(1, 2, 2)
sns.boxplot(x='Methodology', y='Bugs_Detected_Production', data=data)
plt.title("Bugs Detected in Production")

plt.tight_layout()
plt.savefig("bugs_comparison.png")
plt.show()


# 1. Korelasyon Analizi
print("\n### Korelasyon Analizi ###")
correlation_matrix = data.corr()  # Sayısal kolonların korelasyon matrisi
print(correlation_matrix)

# Korelasyon Matrisi Görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Korelasyon Matrisi")
plt.savefig('correlation_matrix.png')  # Görseli kaydetme
plt.show()

# 2. Segmentasyon için Ön İşleme
print("\n### Segmentasyon için Ön İşleme ###")
selected_features = ['Maintainability Index', 'Test Coverage', 'Code Complexity', 'Development Time']  # Önemli özellikler
segment_data = data[selected_features]

# Standartlaştırma
scaler = StandardScaler()
scaled_data = scaler.fit_transform(segment_data)

# PCA ile Boyut Azaltma (2 Boyuta İndirme)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

print(f"PCA ile boyut azaltma sonucu: İlk iki bileşen toplam varyansın %{pca.explained_variance_ratio_.sum()*100:.2f}'ini açıklıyor.")

# 3. Kümeleme (K-Means)
print("\n### Kümeleme (K-Means) ###")
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 küme
clusters = kmeans.fit_predict(pca_data)

# Küme Etiketlerini Veriye Ekleme
data['Cluster'] = clusters

# Kümeleme Sonuçlarını Görselleştirme
plt.figure(figsize=(10, 8))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=clusters, palette='Set2', s=100)
plt.title("K-Means Kümeleme (PCA ile 2D Görselleştirme)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend(title="Cluster")
plt.savefig('kmeans_clustering.png')  # Görseli kaydetme
plt.show()

# Küme Özelliklerinin Analizi
print("\nKümeleme Sonuçları:")
print(data.groupby('Cluster').mean())

# Sonuçları Kaydetme
data.to_csv('tdd_bdd_clustered_data.csv', index=False)
print("\nKüme etiketli veri başarıyla 'tdd_bdd_clustered_data.csv' dosyasına kaydedildi.")


# 1. Veri Görselleştirme
print("\n### Veri Görselleştirme ###")

# Histogram
data.hist(bins=10, figsize=(12, 10))
plt.tight_layout()
plt.savefig("histogram.png")
plt.show()

# Pairplot
sns.pairplot(data, hue='Methodology')
plt.savefig("pairplot.png")
plt.show()

# Boxplot (Özellik bazında)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Methodology', y='Development Time', data=data)
plt.title("Development Time by Methodology")
plt.savefig("development_time_boxplot.png")
plt.show()

# 2. Makine Öğrenimi Analizleri
print("\n### Makine Öğrenimi Analizleri ###")

# Sınıflandırma: Methodology tahmini
print("\n### Sınıflandırma ###")
X_classification = data[['Maintainability Index', 'Test Coverage', 'Code Complexity', 'Development Time']]
y_classification = data['Methodology'].apply(lambda x: 1 if x == 'BDD' else 0)  # BDD=1, TDD=0

# Veriyi bölme
X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_class = rf_classifier.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred_class))
print("Accuracy:", accuracy_score(y_test, y_pred_class))

# Feature Importance Görselleştirme
importance = rf_classifier.feature_importances_
features = X_classification.columns
plt.figure(figsize=(8, 6))
sns.barplot(x=importance, y=features)
plt.title("Feature Importance (Classification)")
plt.savefig("feature_importance_classification.png")
plt.show()

# Regresyon: Development Time tahmini
print("\n### Regresyon ###")
X_regression = data[['Maintainability Index', 'Test Coverage', 'Code Complexity', 'Team Size']]
y_regression = data['Development Time']

# Veriyi bölme
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

# Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = rf_regressor.predict(X_test_reg)

print("Mean Squared Error:", mean_squared_error(y_test_reg, y_pred_reg))
print("Feature Importance (Regressor):", rf_regressor.feature_importances_)

# Scatter Plot: Tahmin vs Gerçek Değerler
plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.7)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
plt.title("Regression: Predicted vs Actual")
plt.xlabel("Actual Development Time")
plt.ylabel("Predicted Development Time")
plt.savefig("regression_scatter.png")
plt.show()

# 3. Model Karşılaştırma
print("\n### Model Karşılaştırma ###")
models = {
    "Random Forest Classifier": RandomForestClassifier(random_state=42),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "SVM Classifier": SVC(kernel='linear', random_state=42),
    "Linear Regression": LinearRegression()
}

# Sınıflandırma modellerini karşılaştırma
for name, model in models.items():
    if "Classifier" in name:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.2f}")
    else:
        model.fit(X_train_reg, y_train_reg)
        y_pred = model.predict(X_test_reg)
        mse = mean_squared_error(y_test_reg, y_pred)
        print(f"{name} Mean Squared Error: {mse:.2f}")
