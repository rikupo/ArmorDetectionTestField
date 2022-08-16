import lwpm_tools as lwpm
import cv2
import numpy as np
import os

fold = "../"
file = "image11d.jpg"
enemy_color = "blue"

img = cv2.imread(file)
print(img.shape)
target = lwpm.determinate(img, enemy_color)
print(target)

target = np.array(target)
print(target)
print(target[:,0])

if False:
    target_center = target[0]
    left_upper = np.array(target[2])
    wh = np.array(target[1])
    print(f"{target_center}, {left_upper}, {wh}, {(left_upper + wh)}")


    cv2.rectangle(img, left_upper, (left_upper + wh), (0, 0, 200), 3, cv2.LINE_AA)

    print(f"WIDTH ::{target[0][0]} + {target[1]}")
    print(f"HEIGHT ::{target[0][1]} + {target[2]}")

    cv2.circle(img, target_center, 3, (0, 255, 0), 3)
    cv2.imshow(file, np.zeros([1,1], np.uint8))
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.imshow("img_show", img_show)
    cv2.waitKey(0)

    cv2.destroyAllWindows()