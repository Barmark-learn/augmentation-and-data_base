import pandas as pd
import numpy as np

lst_data = []
str_data = ''

# Считываем csv файл в дф
df = pd.read_csv('oil_label.csv')

for i in range(df.shape[0]):
    # Переводим в список наши строки с координатами, разделенные нужным знаком
    s_lst = df['label'][i][1:-2].split('}, ')
    # Находим последний и первый индекс элемента строки с координатами якорного бокса
    s_lst_2 = []
    for j in range(len(s_lst)):
        # Находим последний и первый индекс элемента строки с координатами якорного бокса
        last_index = s_lst[j].find(', "rotation"')
        first_index = s_lst[j].find('{')
        s_lst_2.append(s_lst[j][first_index + 1:last_index].split(','))
    # Преобразуем каждый элемент строки в числа: [x, y, width, height] и заносим в список new_lst_2
    new_lst_2 = []
    for r in range(len(s_lst_2)):
        new_lst = []
        for c in range(len(s_lst_2[r])):
            # Тут преобразование типов, в конечном итоге получаем целые числа, преобразованные в строку
            new_lst.append(str(int(float(s_lst_2[r][c][s_lst_2[r][c].find(': ') + 1:]))) + ',')
        new_lst_2.append(new_lst)

    # Заносим наши данные в список
    str_data += df['image'][i]
    for row in range(len(new_lst_2)):
        str_data += ' ' + new_lst_2[row][0] + new_lst_2[row][1] + new_lst_2[row][2] + new_lst_2[row][3] + '0'
    lst_data.append(str_data)
    str_data = ''
# Создаём нампи массив с нашими элементами
arr_data = np.array(lst_data)

# Приводим нампи массив в формат 'Airplane/airplane_'...
#airplane = 'Airplane/'
airplane = 'D:/pythonProject1/'
lst_for_arr = []
for i in range(len(arr_data)):
    arr_data[i] = arr_data[i][24:]
    airplane += arr_data[i]
    lst_for_arr.append(airplane)
    airplane = 'D:/pythonProject1/'

arr_airplane = np.array(lst_for_arr)

print(arr_airplane)

# Сохраняем файл нампи
np.save('oil_label_pycharm', arr_airplane)

