# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt

LEARN_COEFFICIENT = 0.001
LOOP_MAX = 1000


# 未知データの推測
def predict(weight_vec, data_vec, b):
    sum = b
    for i, weight in enumerate(weight_vec):
        sum += weight * data_vec[i]
    if sum >= 0:
        label = 1

    else:
        label = -1
    return label


# １つの学習データに対する学習
def train(weight_vec, data_vec, y, b):
    label = predict(weight_vec, data_vec, b)
    is_updated = False
    if label * y < 0:  # 誤識別した場合は重みを更新
        is_updated = True
        # 重みの更新
        for i, weight in enumerate(weight_vec):
            weight_vec[i] = weight + LEARN_COEFFICIENT * y * data_vec[i]
            # バイアス項の更新
            # max_element = max(data_vec)
            # b = b + LEARN_COEFFICIENT * y * max_element * max_element
    result = {
        'weight_vec': weight_vec,
        'b': b,
        'is_updated': is_updated
    }
    return result


# 学習データの取得
def get_training_data():
    data = csv.reader(file('iris.csv'), delimiter=',')
    training_data = []
    training_data_label = []
    for line in data:
        training_data_row = []
        training_data_row.append(float(line[0]))
        training_data_row.append(float(line[1]))
        training_data.append(training_data_row)
        training_data_label.append(int(line[4]))
    print training_data
    print training_data_label
    return training_data, training_data_label


if __name__ == '__main__':
    # 学習データ読み込み
    (training_data, training_data_label) = get_training_data();

    # 重みの初期値
    weight_vec = [0, 1]
    b = 1

    # 学習
    is_train_completed = False
    loop = 0
    while True:
        loop += 1
        if is_train_completed:
            print 'training finished! (converge)'
            break
        if loop > LOOP_MAX:
            print 'training finished! (not converge)'
            break

        updated_count = 0
        for i, training_data_vec in enumerate(training_data):
            result = train(weight_vec, training_data_vec, training_data_label[i], b)
            # print result
            if (result['is_updated']):
                weight_vec = result['weight_vec']
                b = result['b']
                updated_count += 1
                # print updated_count

        print 'loop', loop, 'finished...', 'updated_count: ', updated_count
        is_train_completed = (updated_count == 0)

    # 学習結果
    print 'weight : ', weight_vec
    print 'b : ', b

    # テスト
    print '[5.0, 2.5]:', predict(weight_vec, [5.0, 2.5], b)
    print '[4.5, 4.8]:', predict(weight_vec, [4.5, 4.8], b)

    # 座標データの作成
    x_1 = []
    y_1 = []
    x_2 = []
    y_2 = []
    for i, training_data_vec in enumerate(training_data):
        if (training_data_label[i] == 1):
            x_1.append(training_data_vec[0])
            y_1.append(training_data_vec[1])
        else:
            x_2.append(training_data_vec[0])
            y_2.append(training_data_vec[1])
    plt.scatter(x_1, y_1, c='g')
    plt.scatter(x_2, y_2, c='r')

    # 識別面
    x_fig = range(3, 9)
    y_fig = [-(weight_vec[0] / weight_vec[1]) * xi - (b / weight_vec[1]) for xi in x_fig]
    plt.plot(x_fig, y_fig)

    plt.show()
