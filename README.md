# End-to-end ml project
В данном репозитории лежит простой пример end-to-end проекта машинного обучения

Проект содержит как ml часть, то есть решение задачи предсказывания цены,
так и ci cd часть для тестирования, упавковки в контейнер и последующего деплоя модели.


--------
## Порядок выполнения задач:
- Работа в Jupyter Notebook (подготовка данных, препроцессинг, EDA, обучение модели, сохрание файла модели в формате .pkl)
- Создание github repo, клонирование репозитория в локальную папку
- Копирование файлов ml проекта (файлы .ipynb и .pkl) в папку, куда мы ранее клонировали репозиторий; открытие всей папки в IDE (PyCharm, например)
- Создание виртуальной среды (в том числе создание файла requirements.txt)

      Сам процесс создания виртуальной среды выглядел следующим образом:
      - Скачать библиотеку virtualenv:
           pip install virtualenv
      - Создаем виртуальную среду
           virtualenv -p python3.8 <_название виртуальной среды_>
      - Запускаем среду:
           source <_название виртуальной среды_>/bin/activate
      - Выйти из среды можно командой
           deactivate


      Установка модулей из файла requirements.txt делается командой:
           pip install -r requirements.txt

- Smth else...g