# 이미지 띄우면서 검증하기
import numpy as np
import cv2 as cv
import sixtick.python.easy as s
import tensorflow as tf
models = tf.keras.models
layers = tf.keras.layers

model = models.load_model('TDOD 모델 생성후 넣기')

def get_train_data_kemp(to_gray = False) :
    image_path_arr = ['../image/KEMP_IMG_DATA_1.png', '../image/KEMP_IMG_DATA_Error_2.png', '../image/KEMP_IMG_DATA_Error_12.png']
    image_arr_weight = [8,1,1]
    path = s.sample(image_path_arr, 1, counts=image_arr_weight)[0]
    if to_gray : 
        x_train = s.cv.imread(path)
        y_train = not 'Error' in path
        return x_train, s.np.array([y_train])
    else : 
        x_train = s.cv.imread(path, s.cv.IMREAD_GRAYSCALE)
        y_train = not 'Error' in path
        return x_train.reshape(-1, 720, 480), s.np.array([y_train]) #s.np.array([int(y_train)])


loss_arr = []
accuracy_arr = []
count = 100
for i in range(count):
    x_org, y = get_train_data_kemp(True)
    
    image_name = f'img_OK {i}'
    if not y :
        image_name = f'img_BAD {i}'
    #cv.imshow(image_name, x)
    s.image_center_show(image_name, x_org)

    x = s.cv.cvtColor(x_org, s.cv.COLOR_BGR2GRAY).reshape(-1, 720, 480)
    print(x.shape)
    loss, accuracy = model.evaluate(x, y)
    if accuracy == 1 and y == True :
        # cv.drawMarker(x_org, (20,20), color=(0,0,255), markerType=cv.marker, markerSize=20)
        cv.circle(x_org, (480//2,720//2), radius=150, thickness=20, color=(255,0,0)) # 원
    elif accuracy == 1 and y == False: 
        cv.drawMarker(x_org, (480//2,720//2), color=(0,0,255), markerType=cv.MARKER_TILTED_CROSS, markerSize=300, thickness=20)
    else : 
        cv.drawMarker(x_org, (480//2,720//2), color=(0,255,0), markerType=cv.MARKER_STAR, markerSize=300, thickness=20)
    s.image_center_show(image_name, x_org)
    loss_arr.append(round(loss, 1))
    accuracy_arr.append(round(accuracy, 1))
    cv.waitKey(500)



cv.destroyAllWindows()
print(np.mean(loss_arr), np.mean(accuracy_arr))