# import tensorflow as tf
# models = tf.keras.models
# layers = tf.keras.layers


# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(720, 480, 1))
#     ,layers.MaxPooling2D((2, 2))
#     ,layers.Conv2D(64, (3, 3), activation='relu')
#     ,layers.MaxPooling2D((2, 2))
#     ,layers.Conv2D(128, (3, 3), activation='relu')
#     ,layers.MaxPooling2D((2, 2))
#     ,layers.Conv2D(128, (3, 3), activation='relu')
#     ,layers.MaxPooling2D((2, 2))
#     ,layers.Flatten()
#     ,layers.Dense(720, activation='relu')
#     ,layers.Dense(1, activation='sigmoid')  # 이진 분류이므로 sigmoid 사용
# ])
# # 모델 컴파일
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# import sys
# sys.path.append('D:/sixtick')
# import sixtick.python.easy as s
# def get_train_data_kemp(to_gray = True) :
#     image_path_arr = ['./image/KEMP_IMG_DATA_1.png', './image/KEMP_IMG_DATA_Error_2.png', './image/KEMP_IMG_DATA_Error_12.png']
#     image_arr_weight = [6,2,2]
#     # path = s.sample(image_path_arr, 1, counts=image_arr_weight)[0]  #choices
#     path = s.choices(image_path_arr, k=1, weights=image_arr_weight)[0]  #choices
#     if not to_gray : 
#         x_train = s.cv.imread(path)

#         height, width = x_train.shape[:2]
#         angle = s.random() * 10 - 5
#         rotation_matrix = s.cv.getRotationMatrix2D((width / 2, height / 2), angle, 1)
#         rotation_matrix[0][2] += s.random() * 72 - 36 # 세로 축 이동
#         rotation_matrix[1][2] += s.random() * 48 - 24 # 가로 축 이동
#         # print(rotation_matrix, type(rotation_matrix), rotation_matrix[0][2])
#         x_train = s.cv.warpAffine(x_train, rotation_matrix, (width, height))

#         y_train = not 'Error' in path
#         return x_train, s.np.array([y_train])
#     else : 
#         x_train = s.cv.imread(path, s.cv.IMREAD_GRAYSCALE)

#         height, width = x_train.shape
#         angle = s.random() * 10 - 5
#         rotation_matrix = s.cv.getRotationMatrix2D((width / 2, height / 2), angle, 1)
#         rotation_matrix[0][2] += s.random() * 72 - 36 # 세로 축 이동
#         rotation_matrix[1][2] += s.random() * 48 - 24 # 가로 축 이동
#         # print(rotation_matrix, type(rotation_matrix), rotation_matrix[0][2])
#         x_train = s.cv.warpAffine(x_train, rotation_matrix, (width, height))

#         y_train = not 'Error' in path
#         return x_train.reshape(-1, 720, 480), s.np.array([y_train]) #s.np.array([int(y_train)])


# from IPython.display import clear_output
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rc('font', family='gulim')
# history_df = pd.DataFrame()

# # plt.ion()  # Interactive 모드를 켬 (실시간 업데이트)
# # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# repeat = 50

# for i in range(repeat):
#     x_train, y_train = get_train_data_kemp()
#     x_val, y_val = get_train_data_kemp()
#     # print(len(history))
#     if len(history_df) == 50:
#         history_df.drop(0, inplace=True)
#         history_df.reset_index(drop=True, inplace=True)
#     history = model.fit(
#         x_train,  # 입력 데이터 (훈련 데이터)
#         y_train,  # 출력 데이터 (훈련 데이터의 레이블)
#         # steps_per_epoch=100,
#         epochs=1,
#         validation_data=(x_val, y_val)  # 검증 데이터와 검증 레이블
#         # validation_steps=50
#         ,verbose=0
#     )
#     if len(history_df) == 0 :
#         history_df = pd.DataFrame(history.history)
#         history_df['val_accuracy_mean'] = history_df['val_accuracy']
#         history_df['val_loss_mean'] = history_df['val_loss']
#     else :
#         history_df = pd.concat([history_df, pd.DataFrame(history.history)], axis=0)
#         history_df.reset_index(drop=True, inplace=True)
#         history_df.loc[i, 'val_accuracy_mean'] = history_df['val_accuracy'].mean()
#         history_df.loc[i, 'val_loss_mean'] = history_df['val_loss'].mean()
    
#     # ax.clear()  # 기존 플롯을 지움
#     plt.scatter(range(len(history_df)), history_df['val_accuracy'])
#     plt.scatter(range(len(history_df)), history_df['val_loss'])
#     plt.plot(range(len(history_df)), history_df['val_accuracy_mean'], label=f'val_accuracy_mean {history_df.loc[i, "val_accuracy_mean"]:.2f}')
#     plt.plot(range(len(history_df)), history_df['val_loss_mean'], label=f'val_loss_mean {history_df.loc[i, "val_loss_mean"]:.2f}')
#     plt.xlabel('COUNT')

#     # plt.draw()
#     # plt.pause(0.1)
#     plt.title('learning images')
#     plt.legend()
#     plt.show()

#     print(i+1)
#     clear_output(wait=True)

# # plt.ioff()  # Interactive 모드 끔
# # plt.show()

# # C:\Users\hungh\AppData\Local\Programs\Python\Python37\py3_7_9_tfgpu\Scripts\activate
# # python C:\source\kamp_ex\learning.py
