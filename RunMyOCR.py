from train import train
from test import test
import pickle


def RunMyOCR(imagedir, input_file):
    data = train(imagedir, False)
    print("Recognition rate: " + str(test(data[0], data[1], data[2], data[3], input_file, False)))
    # maxI = 0
    # maxP = float('-inf')
    # for i in range(1, 11):
    #         print("Testing i = " + str(i))
    #         data = train(imagedir, False, i)
    #         x = test(data[0], data[1], data[2], data[3], input_file, False)
    #         if x > maxP:
    #             maxP = x
    #             maxI = i
    # print("max correct: " + str(maxP) + " r: " + str(maxI))
    #


if __name__ == "__main__":
    RunMyOCR("./images", "./test.bmp")
