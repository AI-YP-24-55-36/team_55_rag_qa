**## Описание работы класса, выполняющего бенчмарк видов поиска**

файл  `read_data_from_csv.py`  содержит функцию  чтения датасета из файла и подготовка данные для посылки в бенчмарк

файл `bench.py` содержит класс `QdrantAlgorithmBenchmark`, в котром есть методы:
- add_search_algorithm - метод для добавления поискового алгоритма
- add_model - метод для добавления модели для генерации эмбеддингов
- upload_data - загрузка датасета
- run_benchmark - выполнение сравнени по времени
- visualize_results - отрисовка графиков сравнения алгоритмов по времени и метрикам
- clear_collection - очищение коллекции перед созданием

  
файл `client.py` модержит функцию main (принимает на вход модель для создания эмбеддингов), которая вызывает  `QdrantAlgorithmBenchmark`. В функции проводится эксперимент по оптимизации алгоритма поиска и повторный тест. 

Результат работы выглядит так:


<img width="1185" alt="image" src="https://github.com/user-attachments/assets/07ba3fc7-3149-4924-8d19-38e6675c31ac" />


<img width="1184" alt="image 4" src="https://github.com/user-attachments/assets/23187896-5c86-463f-bb1a-0b12d81945c9" />


<img width="1187" alt="image 5" src="https://github.com/user-attachments/assets/8137358b-3785-476b-b4c3-bbb05df6d3e6" />



<img width="780" alt="image 2" src="https://github.com/user-attachments/assets/5793603f-365b-473a-a998-d4d8929a2da8" />


<img width="511" alt="image 3" src="https://github.com/user-attachments/assets/41412831-1406-437f-988c-6f2a539542f0" />



Класс планирует доработать до более гибкого варианта, в котором будет возможно сравнить несколько моделей токенизации. Также  функция будет доработана для возможности подачи гиперпараметров поиска, размера эмбеддинга и т.д.

