# test_yolov5_detect_image

Детекция картинок в документах. 

Использовалась технология yolov5 (модель yolov5x6, инференс которой заявлен в документации на CPU 0,1-0,2sec).

Была получена и обработана разметка в исходных данных excel таблицы, приведена к формату координат yolo.

Обучение происходило на 150 эпохах. 

Для запуска необходимо:

При первом запуске 
1. создать любую директорию, куда будем клонировать репозиторий. -> git clone https://github.com/Alexandr1997ag/test_yolov5_detect_image.git
2. cd test_yolov5_detect_image
3. sudo docker pull ultralytics/yolov5:latest

   
Далее
1. sudo docker run --ipc=host -it -v "$(pwd)":/usr/src/inference ultralytics/yolov5:latest
2. python /usr/src/inference/args_test.py  (запустит инференс для тестового фото, расположенного в контейнере в /usr/src/inference/123.png).
Чтобы протестировать для новой фотографии, необходимо выйти из контейнера exit и добавить в исходный каталог (локальный) фото для тестирования. Далее - повторить пункт 1 и пункт 2 с параметром --image_path /usr/src/inference/NAME_YOUR_TEST_PHOTO. Результат - координаты или отсутствие распознанных предметов. 
