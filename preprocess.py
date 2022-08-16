import cv2
import numpy as np


def main():
    image_darked_dir = "./images_darked/"
    file_name = "13.jpg"
    img = cv2.imread(image_darked_dir + file_name)
    img = threshold_filtering(img,50,150,150)
    img = binarizer(img)

    print(img)
    image_shower(img)


def rectangulizer(img): # 外接矩形
    rect_list = []
    imgEdge, contours, hierarchy = cv2.findContours(1, 1, 2)
    return img,rect_list


def image_shower(image):
    cv2.imshow("window name dayo",image)
    cv2.waitKey(0)


def threshold_filtering(img,b,g,r):
    th, img_th = cv2.threshold(img, b, 255, cv2.THRESH_BINARY)
    return img_th


def binarizer(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



if __name__ == "__main__":
    main()
