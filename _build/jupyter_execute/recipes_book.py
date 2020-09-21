# Сборник полезных рецептов

Эта тетрадка - набор "рецептов" в виде кода, который свободно можно использовать в своих проектах. Необязательно делать так же - но в случае затруднений сюда можно подглядывать. Кроме того, эта тетрадка организована блоками по порядку исследований, не каждый блок кода будет нужен в каждом исследовании, но какие-то могут быть полезны.

%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## 1. Загрузка данных

dataframe = pd.read_csv(
    'data/synthetic_data.csv',
    sep=';', # разделитель колонок
    decimal=',', # разделитель дробей у чисел
    parse_dates=['date'], # у какой колонки разбирать даты
    dayfirst=True, # первым в дате идёт номер дня
    header=0 # первая строка - заголовки
)

dataframe.head()

dataframe.dtypes

Аналогично можно поступить и с Excel-файлами (при наличии некоторых предустановленных библиотек по типу `xlrd` - если их нет, будет ошибка).

dataframe = pd.read_excel('data/synthetic_data.xlsx', sheet_name='data')
dataframe.head()

dataframe.info()

Чтение текстового файла чуть-чуть посложнее.

with open('data/synthetic_data.csv', 'r', encoding='utf-8') as fd:
    content = fd.read()

content.split()[:4]

## 2. Разведочный анализ

# описательные статистики
dataframe.describe(include='all')

# если колонок много, эту таблицу удобно транспонировать
# без include='all' будут показаны только числовые колонки
dataframe.describe().T

# выбор уникальных значений
pd.unique(dataframe.group)

# подсчет количества уникальных
pd.value_counts(dataframe.group)

Основное что не следует забывать в разведочном анализе - это отсмотреть распределение как минимум целевой величины.

sns.distplot(
    dataframe.value,
    label='VALUE'
)

sns.distplot(
    dataframe.target,
    hist=True, # отображать столбчатую диаграмму
    kde=True, # отображать оценку плотности линией
    label='TARGET',
    color='red'
)

# точка с запятой в конце вывода графики нужна, чтобы 
# не отображать объект диаграммы в выводе ячейки
plt.legend(loc='best');

plt.title('Точечная диаграмма')
sns.scatterplot(x="value", y="target", hue="group", data=dataframe);

plt.title('Диаграмма по времени')
sns.lineplot(x="date", y="value", data=dataframe, label="value")
sns.lineplot(x="date", y="target", data=dataframe, label="target")

# метод форматирования отображения дат для get current figure (gcf)
plt.gcf().autofmt_xdate();

# ну и конечно же попарные диаграммы
sns.pairplot(
    hue='group',
    data=dataframe,
    # выбрать только конкретные колонки
    vars=['value', 'target']
);

## 3. Подготовка данных

# заполнение пропусков в колонке средним значением
dataframe['group'] = dataframe['group'].fillna('mean')

# заполнение пропусков в колонке нулем
dataframe['group'] = dataframe['group'].fillna(0)

# удаление всех строк с пропущенными значениями
dataframe = dataframe.dropna()

# удаление всех колонок с пропущенными значениями
dataframe = dataframe.dropna(axis='columns')

# случайная выборка из набора данных
dataframe.sample(2)

# добавление новой колонки как функции двух других
dataframe['function'] = dataframe['target'] * dataframe['group']
dataframe['function'].describe()

# удаление колонки
dataframe = dataframe.drop('function', axis='columns')
dataframe.columns

Перейдем к `numpy`-массивам.

X = dataframe[['number', 'value']].values
y = dataframe['target'].values
z = dataframe['group'].values

print('X shape', X.shape)
print('y shape', y.shape)
print('z shape', z.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, z_train, z_test, indices_train, indices_test = train_test_split(
    X, y, z, dataframe.index, # z, index, и даже y - необязательно, то есть можно X_train, X_test = train_test_split(X, random_state=...)
    random_state=1, # зафиксируем генератор случайных чисел для воспроизводимости
    test_size=0.2, # 20% тестовое множество
)

print('X train shape:', X_train.shape, 'X test shape:', X_test.shape)
print('y train shape:', y_train.shape, 'y test shape:', y_test.shape)
print('z train shape:', z_train.shape, 'z test shape:', z_test.shape)

# кодировщик, который кодирует категориальные переменные векторами из 1 и 0
from sklearn.preprocessing import OneHotEncoder

# поскольку категории заранее известны, подгонять можно на всём множестве
encoder = OneHotEncoder(
    sparse=False # скажем не использовать разреженные матрицы
).fit(z.reshape(-1, 1))
encoder.transform(z.reshape(-1, 1))[0:2]

# кодировщик, который для числовых переменных вычитает среднее и делит на разброс
from sklearn.preprocessing import StandardScaler  

# числовые кодировщики следует настраивать на тренировочном множестве
scaler = StandardScaler().fit(X_train)
scaler.transform(X_test)[0:2]

# так можно делать "пайплайны" - цепочки преобразований

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=2, random_state=1)
)

pipeline.fit(X_train)
pipeline.transform(X_test[0:2])

# при желании, можно добавить колонку
# и подогнать и преобразовать сразу

transformed = pipeline.fit_transform(np.column_stack([
    X_train, encoder.transform(z_train.reshape(-1, 1))
]))

transformed[0:2]

# есть и другие интересные препроцессоры
from sklearn.preprocessing import PolynomialFeatures

pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=True, interaction_only=False),
    StandardScaler()
)

pipeline.fit(X_train)

## 4. Построение моделей

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix, silhouette_score

Для `scikit-learn` всё достаточно просто. **fit-predict-score** :)

linear = LinearRegression().fit(
    pipeline.transform(X_train), y_train
)

"R2 test score %.3f" % linear.score(pipeline.transform(X_test), y_test)

"MAE test %.3f" % mean_absolute_error(y_test, linear.predict(pipeline.transform(X_test)))

logistic = LogisticRegression(random_state=1).fit(
    X_train, z_train
)

"Test set accuracy %.3f" % logistic.score(X_test, z_test)

confusion_matrix(logistic.predict(X_test), z_test)

clusterer = KMeans(n_clusters=2).fit(pipeline.transform(X))
"silhouette score %.3f" % silhouette_score(
    pipeline.transform(X),
    clusterer.labels_
)

## 5. Визуализация прогнозов

plt.title('Сравнение распределений')
sns.distplot(y_test, hist=False)
sns.distplot(linear.predict(pipeline.transform(X_test)), hist=False);

# у пайплайна возможно перечислить все его компоненты
featurer = pipeline.steps[0][1]

plt.figure(figsize=(10, 4))
plt.title('Влияние нормированных признаков на отклик')
plt.bar(featurer.get_feature_names(), linear.coef_);

# еще барчарт
plt.bar(pd.unique(clusterer.labels_).astype(str), pd.value_counts(clusterer.labels_), color=['steelblue', 'lightgray']);

plt.title('Красивая матрица несоответствий')
sns.heatmap(confusion_matrix(logistic.predict(X_test), z_test), annot=True);

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('True groups')
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=z_test
)
plt.subplot(1, 2, 2)
plt.title('Predicted groups')
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=logistic.predict(X_test)
);

plt.title('Clustered groups')

plt.plot(
    dataframe.date,
    dataframe.target,
    ls='--'
)
plt.scatter(
    dataframe.date,
    dataframe.target,
    c=[
        'red' if cluster == 0 else 'green' \
        for cluster in clusterer.labels_
    ]
)
plt.gcf().autofmt_xdate()

# очень полезная команда - сжимает отступы у графиков
plt.tight_layout();

---
**Удачи с проектами!**