pipeline {
    agent any
    stages {
        stage('Копирование')
        steps {
            git branch: 'main', url: 'https://github.com/Ygorik/AMO2'
        }

        stage('Установка зависимостей')
        steps {
            pip install -r requirements.txt
        }

        stage('Создание данных')
        steps {
            bat 'python data_creation.py'
        }

        stage('Обучение')
        steps {
            bat 'python model_preprocessing.py'
        }

        stage('Предсказание')
        steps {
            bat 'python model_preparation.py'
        }

        stage('Тестирование')
        steps {
            bat 'python model_testing.py'
        }
    }
}