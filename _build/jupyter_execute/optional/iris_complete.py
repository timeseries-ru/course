# Пример исследования данных

Эта тетрадка - пример завершенного исследования датасета iris. Будет рассмотрено три задачи машинного обучения - регрессия (обучение с учителем), классификация (обучение с учителем), кластеризация (обучение без учителя).

Это не призыв всё делать так же. Это пример как можно сделать некоторый результат - причем сразу в трех вариантах. Впрочем, достаточно уже вступлений, давайте начнем.

# импортируем библиотеки для работы с данными
import numpy as np # это библиотека для работы с массивами чисел
import pandas as pd # это библиотека для работы с датасетами
import seaborn as sns # это библиотека для визуализаций

# следующая магическая команда позволяет отображать диаграммы в тетрадке
%matplotlib inline

Мы возьмем уже готовый датасет iris из библиотеки sklearn, сначала сохраним его как csv-файл, а затем обратно считаем - в учебных целях.

# это подготовительная клетка. 

from sklearn.datasets import load_iris

# получим данные
data = load_iris()

# посмотрим что в словаре данных есть
data.keys()

print(data.DESCR) # распечатаем описание датасета

Наши данные представляют собой 150 записей о цветках ириса, которые представлены 4 признаками - sepal width/length, petal width/lengh (ширина и длина цветка и листов). Каждому цветку соответствует один из трех сортов: Setosa, Versicolour, Virginica.

# загрузим наши данные в датафрейм и тем самым создадим датасет
dataframe = pd.DataFrame({
    'sepal length': data.data[:, 0], # data.data - это numpy-массив признаков. 
    'sepal width': data.data[:, 1], # обозначением :, 1 - мы индексируем все строки и только ВТОРУЮ колонку
    'petal length': data.data[:, 2], 
    'petal width': data.data[:, 3],
    'target': data.target, # target содержит только порядковый номер сорта 
    'sorts': [data.target_names[sort] for sort in data.target] # поэтому мы обходим массив номеров
    # и получаем строки названий
})

# сохраним данные в csv
dataframe.to_csv('iris.csv', index=False)
# сохраним вместе с названиями столбцов
# и без порядкового номера

# перечитаем обратно, задав для примера некоторые параметры
# разделитель - запятая, десятичная точка - точка, кодировка utf-8, в первой строке - заголовки
dataframe = pd.read_csv('iris.csv', sep=',', decimal='.', encoding='utf-8')

# посмотрим на наши данные таблично
dataframe

## Разведочный анализ данных

# посмотрим информацию о данных
dataframe.describe()

Видим максимальные и минимальные значения для признаков, их среднее тоже видим, и меру разброса - стандартное отклонение. Надо сделать визуализацию, чтобы понять с чем мы имеем дело.

# построим точечную диаграмму, задав оси и цвет
sns.scatterplot(x='sepal width', y='petal length', hue='sorts', data=dataframe);

Уже сейчас видно, что на выбранных осях сорта "кучкуются" по группам. Посмотрим как выглядит эта картина по всем сочетаниям осей.

features = [
    'sepal width', 'sepal length',
    'petal width', 'petal length'
]

sns.pairplot(
    data=dataframe[features + ['sorts']], # возьмем только признаки и название сорта
    hue='sorts', # по какой колонке подсвечиваем
); # точка с запятой нужна, чтобы не выводить результат запуска команды

На диагонали у нас стоят гистограммы распределения каждого признака: то есть как часто встречаются те значения, которые отложены по оси Х. Вне диагонали - точечные диаграммы того, как располагаются характеристики конкретных наших 150 цветков ириса по выбранным осям.

Видно, что имеет место некоторое "кучкование" для всех сочетаний признаков, но классы versicolor и virginica ближе друг к другу и слегка так перемешаны.

У нас достаточно хорошие данные, уже почищенные. Если бы в них были пропущенные значения, такие строки можно было бы например удалить.

## Задача регрессии

Попробуем смоделировать зависимость petal width от всех остальных параметров.

# импортируем линейную модель
from sklearn.linear_model import LinearRegression

# импортируем градиентный бустинг над деревьями решений
from sklearn.ensemble import GradientBoostingRegressor

# импортируем методы работы с разбиениями и метриками
from sklearn.model_selection import train_test_split

regression_features = ['petal length', 'sepal length', 'sepal width']

# разобьем наше множество цветков на тренировочное и отложенный тест (25%)
train_X, test_X, train_y, test_y = train_test_split(
    dataframe[regression_features], # у нас будет три признака
    dataframe['petal width'], # один отклик
    test_size=0.25, # и одно разбиение на два множества - по 75% и 25% от всех 150 цветков
    shuffle=True # обязательно перемешаем все записи о цветках
)

# обучим модели
linear_model = LinearRegression().fit(train_X, train_y)
boosting_model = GradientBoostingRegressor().fit(train_X, train_y)

# импортируем методы подсчета метрик регрессии
from sklearn.metrics import r2_score, mean_absolute_error

# и сделаем функцию вывода в тетрадку этих метрик
def print_metrics(regressor):
    print('На тестовом множестве')
    test_predictions = regressor.predict(test_X)
    print('Мера объяснения изменений в данных', r2_score(test_y, test_predictions))
    print('Средняя абсолютная ошибка (в см)', mean_absolute_error(test_y, test_predictions))

print_metrics(linear_model)

print_metrics(boosting_model)

Интересно, какие признаки были более важными c точки зрения градиентного бустинга.

for index, importance in enumerate(boosting_model.feature_importances_):
    print('важность', regression_features[index], '%.3f' % importance)

# с точки зрения линейной регрессии чем больше по модулю коэффициент - тем больше его влияние
for index, importance in enumerate(linear_model.coef_):
    print('коэффицент', regression_features[index], '%.3f' % importance)

Сразу видно, какой признак дал наибольший вклад в предсказание целевой величины.

## Классификация

# снова сделаем разбиение, но уже с другими признаками и другим откликом (индексом сорта)
train_X, test_X, train_y, test_y = train_test_split(
    dataframe[features],
    dataframe['target'],
    test_size=0.25,
    shuffle=True
)

# импортируем модель логистической регрессии
from sklearn.linear_model import LogisticRegression

# импортируем классификатор на основе случайного леса
from sklearn.ensemble import RandomForestClassifier

# обучим их на тренировочном множестве
logreg_model = LogisticRegression(solver='liblinear', multi_class='auto').fit(train_X, train_y)
forest_model = RandomForestClassifier(n_estimators=15).fit(train_X, train_y)

# давайте посмотрим на матрицы несоответствий на тестовом множестве
from sklearn.metrics import confusion_matrix

def print_matrix(model):
    print('Матрица для тестового множества')
    predictions = model.predict(test_X)
    print(confusion_matrix(test_y, predictions))

print_matrix(logreg_model)

print_matrix(forest_model)

Как мы видим, случайный лес здесь ошибся на 1 раз больше, из всего 12 + 9 + 15 + 2 = 38 случаев.

## Кластеризация

Попробуем посмотреть, не зная ничего о сортах, разбиваются ли наши данные на разное количество групп.

# импортируем метрику качества кластеризации
from sklearn.metrics import silhouette_score

# чем больше величина silhouette - тем лучше точки принадлежат своим кластерам. то есть тем лучше

%%time

# магическая команда %%time в начале клетки позволяет узнать сколько времени она выполнялась

from sklearn.cluster import KMeans 

# переберем разное число кластеров и выберем лучший вариант по нашей метрике

best_score = -np.inf # инициализируем лучшую метрику минус бесконечностью
best_model = None

for clusters_number in range(2, 8): # от 2 до 7 включительно
    candidate_model = KMeans(clusters_number).fit(dataframe[features])
    score = silhouette_score(dataframe[features], candidate_model.labels_)
    if score > best_score:
        # запомним модель
        best_model = candidate_model
        best_score = score

print('наилучшая метрика', best_score, 'для', best_model.n_clusters, 'кластеров')

Как видим, KMeans по умолчанию нам дал два кластера. Посмотрим визуально.

dataframe['cluster'] = best_model.predict(dataframe[features])

sns.scatterplot(
    x='petal width',
    y='sepal width',
    hue='cluster',
    data=dataframe
);

Вот как ни странно, таким неочевидным образом, KMeans разбил на кластеры. То есть система отнесения цветов ириса к трем сортам, не самая совершенная. Можно относить их только к двум, у двух сортов слишком похожие характеристики, и они слабо отличаются.

## Интерактивная презентация

Мы сделаем из этой тетрадки интерактивную веб-страничку с помощью сервиса `voila`. Делается веб-страница одной командой, однако чтобы добавить интерактивности у нас будет немного кода.

# импортируем библиотеку, отвечающую за отображение элементов управления интерфейса
import ipywidgets

# сделаем классификатор только от двух признаков
dataset = dataframe[['petal width', 'sepal length', 'target']]

# обучим на всем множестве классификатор для целей презентации
classifier = RandomForestClassifier(n_estimators=40).fit(dataset[
    ['petal width', 'sepal length']
], dataset['target'])

# выведем долю точных ответов
'точность', "%.2f" % classifier.score(dataframe[['petal width', 'sepal length']], dataset['target'])

import matplotlib.pyplot as plt

sorts_names = {
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
}

def plot_chart(petal_width, sepal_length):
    """
        Эта функция, принимая на вход признаки цветка неизвестного сорта,
        относит его к определенному сорту благодаря классификации.
        В итоге она строит диаграмму с этой точкой.
    """
    sort = classifier.predict([[petal_width, sepal_length]])[0]
    plot_data = dataset.copy() # скопируем датасет, так как будем менять
    plot_data['sort'] = [sorts_names[number] for number in plot_data.target]
    plot_data = plot_data.append({
        'petal width': petal_width,
        'sepal length': sepal_length,
        'sort': sorts_names[sort]
    }, ignore_index=True)
    sns.scatterplot(x='petal width', y='sepal length', hue='sort', data=plot_data)
    plt.annotate(
        sorts_names[sort],
        [petal_width, sepal_length],
        [petal_width + 0.3, sepal_length + 0.3],
        arrowprops={'arrowstyle': '->'}
    ); # добавим точке надпись

# сделаем интерактивной нашу диаграмму!
ipywidgets.interact(plot_chart, petal_width=1.5, sepal_length=7.5);

Запустите следующую команду из командной строки, она откроет новую страницу в браузере с этой тетрадкой без кода

```
python -m voila iris_complete.ipynb
```

Перейдите в конец странице, там вы увидите два слайдера, которыми вы сможете управлять координатами точки.