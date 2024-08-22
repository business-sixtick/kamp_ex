import cv2 as cv
import os
import numpy as np
import sys

def get_center_by_image(image):
    # 윈도우 화면 해상도 가져오기
    from win32api import GetSystemMetrics
    screen_width = GetSystemMetrics(0)
    screen_height = GetSystemMetrics(1)
    image_height, image_width = image.shape[:2]
    
    # 창의 중앙 위치 계산
    x = int((screen_width - image_width) / 2)
    y = int((screen_height - image_height) / 2)
    return x, y
