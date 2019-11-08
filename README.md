# uma-ml-bot
Uma.Challenge ml section task

# Подготовка данных
Изначальные данные были разбиты на тренировочную выборку и валидационную выборку с помощью [скрипта](make_data.py)



# Модель
![model](https://i.imgur.com/EfX6JBP.png)

Модель была обучена на [Google.Colab](https://colab.research.google.com), процесс обучения представлен в [ноутбуке](uma_model.ipynb)

Оценка модели после 100 эпох обучения:

Loss | Accuracy
------------ | -------------
0.24400990454355875 | 0.946666667620341

# Размещение
Бот был размещен на [Heroku](https://heroku.com), а в телеграме его можно найти как [@UmaML_bot](https://t.me/UmaML_bot)
