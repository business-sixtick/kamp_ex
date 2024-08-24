import tensorflow as tf
models = tf.keras.models
layers = tf.keras.layers


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(720, 480, 1))
    ,layers.MaxPooling2D((2, 2))
    ,layers.Conv2D(64, (3, 3), activation='relu')
    ,layers.MaxPooling2D((2, 2))
    ,layers.Conv2D(128, (3, 3), activation='relu')
    ,layers.MaxPooling2D((2, 2))
    ,layers.Conv2D(128, (3, 3), activation='relu')
    ,layers.MaxPooling2D((2, 2))
    ,layers.Flatten()
    ,layers.Dense(720, activation='relu')
    ,layers.Dense(1, activation='sigmoid')  # 이진 분류이므로 sigmoid 사용
])
# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

import sys
sys.path.append('D:/sixtick')
import sixtick.python.easy as s
def get_train_data_kemp() :
    image_path_arr = ['./image/KEMP_IMG_DATA_1.png', './image/KEMP_IMG_DATA_Error_2.png', './image/KEMP_IMG_DATA_Error_12.png']
    image_arr_weight = [8,1,1]
    path = s.sample(image_path_arr, 1, counts=image_arr_weight)[0]
    x_train = s.cv.imread(path, s.cv.IMREAD_GRAYSCALE)
    y_train = not 'Error' in path
    return x_train.reshape(-1, 720, 480), s.np.array([y_train]) 

from IPython.display import clear_output
import pandas as pd
import matplotlib.pyplot as plt
history_df = pd.DataFrame()


plt.ion()  # Interactive 모드를 켬 (실시간 업데이트)
fig, ax = plt.subplots()

for i in range(20):
    x_train, y_train = get_train_data_kemp()
    x_val, y_val = get_train_data_kemp()
    # print(len(history))
    if len(history_df) == 50:
        history_df.drop(0, inplace=True)
        history_df.reset_index(drop=True, inplace=True)
    history = model.fit(
        x_train,  # 입력 데이터 (훈련 데이터)
        y_train,  # 출력 데이터 (훈련 데이터의 레이블)
        # steps_per_epoch=100,
        epochs=1,
        validation_data=(x_val, y_val)  # 검증 데이터와 검증 레이블
        # validation_steps=50
        ,verbose=0
    )
    if len(history_df) == 0 :
        history_df = pd.DataFrame(history.history)
    else :
        history_df = pd.concat([history_df,pd.DataFrame(history.history)], axis=0)
        history_df.reset_index(drop=True, inplace=True)

    ax.clear()  # 기존 플롯을 지움
    ax.scatter(range(len(history_df)), history_df['val_accuracy'])
    ax.scatter(range(len(history_df)), history_df['val_loss'])
    plt.draw()
    plt.pause(0.1)
    # plt.show()
    print(i+1)
    # clear_output(wait=True)


plt.ioff()  # Interactive 모드 끔
plt.show()  # 마지막 플롯을 유지