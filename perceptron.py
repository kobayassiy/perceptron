# -*- coding: utf-8 -*-

__author__ = 'koji_ono'

import csv
import matplotlib.pyplot as plt

LEARN_COEFFICIENT = 0.001
#initial_weight_vec = [0,0,1]
#initial_b = 3
#training_data = [[1,1,1],[1,3,4],[-3,-2,-5]]
#training_data_classes = [-1,1,-1]

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
    if label * y < 0: # 誤識別した場合は重みを更新
        is_updated = True
        # 重みの更新
        for i, weight in enumerate(weight_vec):
            weight_vec[i] = weight + LEARN_COEFFICIENT * y * data_vec[i]
        # バイアス項の更新
        max_element = max(data_vec)
        #b = b + LEARN_COEFFICIENT * y * max_element * max_element
    result = {
        'weight_vec' : weight_vec,
        'b' : b,
        'is_updated' : is_updated
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

if __name__=='__main__':
    # 学習データ読み込み
    (training_data, training_data_label) = get_training_data();

    # 重みの初期値
    weight_vec = [0,1]
    b = 1

    is_train_completed = False
    loop = 0
    while True:
        loop += 1
        if is_train_completed:
            print 'training finished! (converge)'
            break
        if loop > 1000:
            print 'training finished! (not converge)'
            break

        updated_count = 0
        for i, training_data_vec in enumerate(training_data):
            result = train(weight_vec, training_data_vec, training_data_label[i], b)
            # print result
            if(result['is_updated']):
                weight_vec = result['weight_vec']
                b = result['b']
                updated_count += 1

        print 'loop', loop, 'finished...', 'updated_count: ', updated_count
        is_train_completed = (updated_count == 0)
    plt.scatter([1,2],[2,3],marker='o',color='g',s=2)
    plt.show


