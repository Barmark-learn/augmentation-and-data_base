# Модули работы с изображениями
from PIL import Image, ImageDraw
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
# визуализация
from matplotlib import pyplot as plt
# библиотека numpy
import numpy as np



# Функция рандома
def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def augmentation_2(data):
    ''' Функция случайной аугментации данных
        Args:
            data - изображения
        Return:
            аугментированные изображение и bounding_box

        '''

    # Словарь с параметрами аугментации
    params = {
        'jitter': .3,
        'hue': .1,
        'sat': 1.5,
        'val': 1.5
    }

    # Сплитим входную строку словаря
    data = data.split()

    # Открываем изображение самолета
    image = Image.open(data[0])

    # Получаем ширину и высоту оригинального изображения
    width_i, height_i = image.size

    # Получаем ширину и высоту входного изображения для модели RetinaNet
    widht_shape, height_shape = input_shape[:2]

    # Получаем координаты ограничивающей рамки и находим колличество box_plot
    box = np.array([np.array(list(map(lambda x: int(float(x)), box.split(',')))) for box in data[1:]])
    count_box = len(box)

    # Случайным образом масштабируем изображение
    new_ar = widht_shape / height_shape * rand(1 - params['jitter'], 1 + params['jitter']) / rand(1 - params['jitter'],
                                                                                                  1 + params['jitter'])
    scale = rand(.65, 2)
    if new_ar < 1:
        nh = int(scale * height_shape)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * widht_shape)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # Преобразуем картинку к input_shape и размещаем случайным образом
    dx = int(rand(0, widht_shape - nw))
    dy = int(rand(0, height_shape - nh))
    new_image = Image.new('RGB', (widht_shape, height_shape), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # С вероятностью 50% отображаем по горизонтале
    flip = rand() < .5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Случайным образом меняем освещенность, экспозицию, гамму изображения
    hue1 = rand(-params['hue'], params['hue'])
    sat1 = rand(1, params['sat']) if rand() < .5 else 1 / rand(1, params['sat'])
    val1 = rand(1, params['val']) if rand() < .5 else 1 / rand(1, params['val'])
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue1
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat1
    x[..., 2] *= val1
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # Получаем окончательный массив

    max_boxes = count_box  # Устанавливаем максимальное количество рамок на изображении

    # Корректируем параметры ограничивающей рамки в соответсвии с проведенными выше преобразованиями
    box_data = np.zeros((max_boxes, 5))  # Создаем массив из нулей размерностью (max_boxes, 5)

    if len(box) > 0:
        # Ресайзим и перемещаем
        box[:, [0, 2]] = box[:, [0, 2]] * nw / width_i + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / height_i + dy

        # Отражаем по горизонтале
        if flip: box[:, [0, 2]] = widht_shape - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        # Ограничиваем, если вышли за пределы input_shape
        box[:, 2][box[:, 2] > widht_shape] = widht_shape
        box[:, 3][box[:, 3] > height_shape] = height_shape
        # Считаем высоту и ширину рамок и оставляем только те, значения которых больше 1
        box_w = box[:, 2] - box[:, 0]  # xRight - xLeft
        box_h = box[:, 3] - box[:, 1]  # yBottom - yTop
        box = box[np.logical_and(box_w > 1, box_h > 1)]

        if len(box) > max_boxes:  # Оставляем только max_boxes рамок
            box = box[:max_boxes]
        box_data[:len(box)] = box  # Записываем данные в box_data

    return image_data, box_data, max_boxes  # Возвращаем аугментированные изображение и bounding_box и кол-во боксов


batch_size = 3
input_shape = (416, 416)

All_data_2 = np.load('oil_label_pycharm.npy')

All_data_3 = []
for i in range(3):
    buf_data = All_data_2[i].split(' ')
    file_name = buf_data[0]
    # Загружаем изображение сегментированной модели
    image = Image.open(file_name)
    X_img, Y_img = image.size
    All_data_3.append(file_name)
    for k in range(1, len(buf_data)):
        minX, minY, width, high, _ = buf_data[k].split(',')
        maxY = int(((int(minY) + int(high)) * Y_img) / 100)
        maxX = int(((int(minX) + int(width)) * X_img) / 100)
        minX = int((int(minX) * X_img) / 100)
        minY = int((int(minY) * Y_img) / 100)
        All_data_3[i] += ' ' + str(minX) + ',' + str(minY) + ',' + str(maxX) + ',' + str(maxY) + ',' + '0'
All_data_2 = np.array(All_data_3)

# Здесь храним все наши данные с именем и координатами, которые потом передадим в нампи массив
lst_box = []
# Цикл для применения 10 раз аугментации и отрисовка
for i in range(len(All_data_2)):
    # Колличество агументаций с каждой картинкой
    for j in range(10):
        image_data, box_data, box_count = augmentation_2(All_data_2[i])
        count = 0
        zeros = np.zeros((1, 5))
        # Создаем экземпляр изображения
        data_img = Image.fromarray(np.uint8(image_data * 255))
        # Сохраняем аугментированное изображение
        data_img.save('C:/Users/antoh/OneDrive/Рабочий стол/Airplanes_double_and_oil/Oil_augmentation'
                      '/augmentation_oil_{}_{}.jpg'.format(i, j))
        # Смотрим какие баундбоксы равны нулю и считаем колличество ненулевых строк с баундбоксами
        img1 = ImageDraw.Draw(data_img)
        # Создаем строку со всеми данными об агументированной картинке
        str_box = 'Airplane/augmentation_oil_{}_{}.jpg '.format(i, j)
        for r in range(box_count):
            for c in range(5):
                a = int(box_data[r, c])
                str_box += str(a) + ','
            str_box = str_box.rstrip(',') + ' '
            img1.rectangle([int(box_data[r][0]), int(box_data[r][1]), int(box_data[r][2]), int(box_data[r][3])],
                           outline='green', width=4)
        # Удаляем лишний пробел
        str_box = str_box.rstrip()
        lst_box.append(str_box)
        #plt.imshow(data_img)
        # Рисуем изображение
        #plt.show()

# Создаём нампи массив с нашими аугментированными элементами
arr_data = np.array(lst_box)
print(arr_data)
# Сохраняем файл нампи
np.save('C:/Users/antoh/OneDrive/Рабочий стол/Airplanes_double_and_oil/Oil_augmentation/oil_label_pycharm_augmentation',
        arr_data)
