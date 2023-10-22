![](https://github.com/plugg1N/gms-module/blob/main/images/chart1.png?raw=true)
*Схема 1: Основной рабочий процесс GMS*

# Краткое описание

**<ins>General Model Selection Module</ins>** *(далее: GMS-Module)* - это простой, но точный инструмент выбора модели, который поможет разработчикам машинного обучения подобрать наиболее эффективную модель/пайалайн для конкретной задачи.

Пользователю достаточно передать:
- *Модели И/ИЛИ пайплайны* по своему выбору
- *Метрики* для оценки
- *Pivot (основную метрику)* , если определенная метрика важнее остальных
- *Данные* для обучения и оценки

Модуль будет автоматически производить оценки, хранить их и давать подробное описание работы каждой модели!

# Установка

Для установки GMSModule убедитесь, что установлены python3 и pip. В терминале просто наберите:
`pip install gms` ИЛИ `pip3 install gms`.


```python
pip3 install gms
```

# Как использовать?

1. Убедитесь, что все переменные подготовлены для использования GMS:
	- `mode`: Строка вашей ML-задачи: `regression` ИЛИ `classification`.
	- `include`: Список моделей-объектов по вашему выбору: `[LinearRegression(), SVR()]`.
	- `metrics`: Список строк для оценки: 
		- `классификация = ['accuracy', 'f1-score', 'precision', 'recall', 'roc-auc']`.
		- `регрессия = ['mae', 'mape', 'mse', 'rmse', 'r2-score']`.
	- `data`: Список данных для обучения/проверки: 
		 `[X_train, X_test, y_train, y_test]`.
	- `pivot`: *если необходимо*: Строка одной из представленных метрик: `'accuracy'` (pivot - метрика, наиболее важная для оценки).

2. Импортируйте модуль GMSModule в свой проект:


```python
from gms.GMSModule import GMSModule
```


2. Создайте объект **GMSModule** с вашими данными:

```python
GMSPipe = GMSModule(mode="classification",
	pivot='f1-score',
	metrics=['accuracy', 'f1-score'],
	include=[LogisticRegression(), RandomForestClassifier()],
	data=[X_train, X_test, y_train, y_test])
```

3. Используйте любой из предложенных методов:

```python
best_model, _ = GMSPipe.best_model()
print(best_model)
```


```python
RandomForestClassifier()
```


# Зачем нужен этот модуль?

Каждому разработчику Machine Learning, особенно после тщательного анализа данных, приходится выбирать **наиболее точную модель Machine Learning**. Некоторые инженеры уже знают, какая модель подойдет идеально, в силу простоты поставленной задачи или очевидности ML-модели.

> **Но некоторые инженеры могут столкнуться с проблемой "Слепого" выбора
> между десятками, если не сотнями, построенных ими ML-моделей/пайплайнов. Вот здесь-то и может помочь GMS Module!

Пользователю не нужно создавать пользовательскую функцию, которая бы оценивала каждую модель по отдельности по их метрикам. **Пользователю достаточно передать каждую модель и название метрики по своему выбору** и *вуаля!*

Затем пользователь может посмотреть `GMSModule.description()` и получить подробную информацию об оценках моделей, а также увидеть, какие модели лучше других.

Пользователь также может получить свои данные в переменные для дальнейшего использования, как в этом <ins>примере</ins>:


```python
# Получение предсказаний лучшей модели из списка
_, preds = GMSModule.best_model()

# Создание информации, передаваемой в Датафрейм
data = {
	'id': range(25000),
	'value': preds
}

# Создание DataFrame и передача в него информации
df = pd.DataFrame(data)
df.to_csv('submit.csv', index=False)
```


# История проекта:


Этот проект был создан как **забавный побочный проект для меня**, чтобы поэкспериментировать с инструментами scikit-learn. Проект помог мне стать более сфокусированным на программировании в целом и научил меня писать свой собственный модуль PYPI для использования другими!

> Идея родилась `16.10.2023`, и первая версия проекта была настолько неэффективной, что мне пришлось переписать почти все.

Модуль использовался для повторной оценки каждый раз, когда я пытался получить оценки каждой модели. В оценках использовались "if-условия", что выглядело отвратительно и непрофессионально.

В 5-й версии, выполненной `20.10.2023`, все было изменено. Исправлена проблема с переоценкой, модуль может отлавливать наиболее очевидные исключения, вызванные пользователем, а "if-высказывания" заменены на аккуратные словари.

`22.10.2023`, я создаю первую версию этого Markdown (README.md) файла. Проект отшлифован. Все, что мне нужно, это:

- Создать файл модуля (`*.py`)
- Написать кучу документации: эту доку на русском, прогон кода, основные сценарии использования и многое другое!
- Получить лицензию
- Разместить этот модуль на PYPI

`21:54. 22.10.2023` я разместил свою работу на PYPI, все работает замечательно. Хорошего пользования моей библиотекой :)


# My Socials:

- Полное имя:  Жамков Никита Дмитриевич
- Страна, город:  Россия, Санкт-Петербург
- Номер телефона: +79119109210
- Электронная почта: nikitazhamkov@gmail.com
- GitHub: https://github.com/plugg1N
- Telegram: https://t.me/jeberkarawita