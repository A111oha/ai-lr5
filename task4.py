from sklearn.utils import shuffle  # Correct shuffle import
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split

# Завантаження даних із цінами на нерухомість (California housing dataset)
#Набір Бостон був видалений із sklearn
housing_data = datasets.fetch_california_housing()

# Переміщаємо дані, щоби підвищити об'єктивність нашого аналізу
X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

# Розбиття даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Модель на основі регресора AdaBoost
regressor = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4),
    n_estimators=400, random_state=7
)
regressor.fit(X_train, y_train)

# Оцінка ефективності регресора
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

# Вилучення важливості ознак
feature_importances = regressor.feature_importances_
feature_names = housing_data.feature_names

# Нормалізація значень важливості ознак
feature_importances = 100.0 * (feature_importances / max(feature_importances))

# Сортування та перестановка значень
index_sorted = np.flipud(np.argsort(feature_importances))

# Розміщення міток уздовж осі X
pos = np.arange(index_sorted.shape[0]) + 0.5

# Побудова стовпчастої діаграми
plt.figure(figsize=(10, 6))  # Adjust the figure size
plt.bar(pos, feature_importances[index_sorted], align='center')

# Correctly set the x-ticks
plt.xticks(pos, [feature_names[i] for i in index_sorted], rotation=90)

# Adding labels and title
plt.ylabel('Relative Importance')
plt.title('Оцінка важливості признаков с использованием регрессора AdaBoost')

# Adjust layout to prevent clipping
plt.tight_layout()

plt.show()



