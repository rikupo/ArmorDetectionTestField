import numpy as np
import cv2
import os


def main():
    b = [[[10, 10], [30, 40], [50, 60], [70, 80]], [[25, 30], [31, 41], [51, 61], [71, 81]]]
    a = np.array(b)
    print(a)

    print(a[:, 0])
    bef = [20, 15]

    centers = a[:, 0]
    print(f"A {np.sqrt((10 - 20)^2 + (10 - 15)^2)}")
    print(f"B {np.sqrt((25-20)^2 + (30-15)^2)}")

    # euqlid_distance = list(map(lambda x: np.sqrt((x[0]-bef[0])^2 + (x[1]-bef[0])^2), centers))
    euqlid_distance = list(map(lambda x: np.sqrt((x[0]-bef[0])^2 + (x[1]-bef[1])^2), centers))
    print(f"DISTANCE : {euqlid_distance}")
    print(f"LIST No. {euqlid_distance.index(min(euqlid_distance))}")
    print(f"Return : {b[euqlid_distance.index(min(euqlid_distance))]}")

if __name__ == "__main__":
    main()