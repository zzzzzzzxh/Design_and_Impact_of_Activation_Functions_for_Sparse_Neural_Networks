import matplotlib.pyplot as plt
import numpy as np
import os
from utils.result_utils import *

import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
import os
import math

def get_res(r1,r2,r3):



    avg = []
    for i in range(len(r1)):
        avg.append(round((r1[i] + r2[i] + r3[i]) / 3, 3))


    var = []
    for i in range(len(r1)):
        var.append(((r1[i] - avg[i]) ** 2 + (r2[i] - avg[i]) ** 2 + (r3[i] - avg[i]) ** 2) / 3)


    std = []
    for i in range(len(r1)):
        std.append(math.sqrt(var[i]))


    r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))
    r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))
    return avg,r1,r2




def PlotActiFunc(x, y, title):
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    plt.plot(x, y)
    plt.title(title)
    plt.show()


def PlotMultiFunc(x, y):
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    plt.plot(x, y)

def PlotMultiFunc_var(x, y,r1,r2):
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    plt.plot(x, y)
    plt.fill_between(x, r1, r2,
                     color=cm.viridis(0.5), alpha=0.2)


if __name__ == '__main__':
    x = np.arange(0, 500, 1)


    Lrelu_res_1 = read_dic('sparse_level_result/madalon_Lrelu_0.5_500epoch_epsilon40.txt')
    Lrelu_res_2 = read_dic('sparse_level_result/madalon_Lrelu_0.5_500epoch_epsilon40_3.txt')
    Lrelu_res_3 = read_dic('sparse_level_result/madalon_Lrelu_0.5_500epoch_epsilon40_2.txt')



    relu_res_1 = read_dic('sparse_level_result/madalon_relu_500epoch_epsilon40.txt')
    relu_res_2 = read_dic('sparse_level_result/madalon_relu_500epoch_epsilon40_2.txt')
    relu_res_3 = read_dic('sparse_level_result/madalon_relu_500epoch_epsilon40_3.txt')

    allrelu_res_1 = read_dic('sparse_level_result/madalon_allrelu_0.5_500epoch_epsilon40.txt')
    allrelu_res_2 = read_dic('sparse_level_result/madalon_allrelu_0.5_500epoch_epsilon40_3.txt')
    allrelu_res_3 = read_dic('sparse_level_result/madalon_allrelu_0.5_500epoch_epsilon40_2.txt')

    grelu_res_1 = read_dic('sparse_level_result/madalon_grelu_0.5_500epoch_epsilon40.txt')
    grelu_res_2 = read_dic('sparse_level_result/madalon_grelu_0.5_500epoch_epsilon40_2.txt')
    grelu_res_3 = read_dic('sparse_level_result/madalon_grelu_0.5_500epoch_epsilon40_3.txt')


    #Lrelu acc
    Lrelu_acc_1 = Lrelu_res_1['acc']
    Lrelu_acc_2 = Lrelu_res_2['acc']
    Lrelu_acc_3 = Lrelu_res_3['acc']
    avg,r1,r2, = get_res(Lrelu_acc_1,Lrelu_acc_2,Lrelu_acc_3)
    avg = [elem * 100 for elem in avg]
    r1 = [elem * 100 for elem in r1]
    r2 = [elem * 100 for elem in r2]
    PlotMultiFunc_var(x,avg,r1,r2)
    # relu acc
    relu_acc_1 = relu_res_1['acc']
    relu_acc_2 = relu_res_2['acc']
    relu_acc_3 = relu_res_3['acc']
    avg, r1, r2, = get_res(relu_acc_1, relu_acc_2, relu_acc_3)
    avg = [elem * 100 for elem in avg]
    r1 = [elem * 100 for elem in r1]
    r2 = [elem * 100 for elem in r2]
    PlotMultiFunc_var(x, avg, r1, r2)
    # allrelu acc
    allrelu_acc_1 = allrelu_res_1['acc']
    allrelu_acc_2 = allrelu_res_2['acc']
    allrelu_acc_3 = allrelu_res_3['acc']
    avg, r1, r2, = get_res(allrelu_acc_1, allrelu_acc_2, allrelu_acc_3)
    avg = [elem * 100 for elem in avg]
    r1 = [elem * 100 for elem in r1]
    r2 = [elem * 100 for elem in r2]
    PlotMultiFunc_var(x, avg, r1, r2)
    # grelu acc
    grelu_acc_1 = grelu_res_1['acc']
    grelu_acc_2 = grelu_res_2['acc']
    grelu_acc_3 = grelu_res_3['acc']
    avg, r1, r2, = get_res(grelu_acc_1, grelu_acc_2, grelu_acc_3)
    avg = [elem * 100 for elem in avg]
    r1 = [elem * 100 for elem in r1]
    r2 = [elem * 100 for elem in r2]
    PlotMultiFunc_var(x, avg, r1, r2)


    # plt.legend(['ReLU', 'All-ReLU', 'GAReLU'])
    plt.legend(['Leaky ReLU', 'ReLU', 'All-ReLU', 'GAReLU'])
    plt.xlabel("Epochs[#]")
    plt.ylabel("Accuracy on the test set[%]")
    plt.title('Performance on Madelon')
    # plt.ylim([50, 95])
    plt.figure(1)
    plt.show()


    #Lrelu EGF
    Lrelu_EGF_1 = Lrelu_res_1['EGF']
    Lrelu_EGF_2 = Lrelu_res_2['EGF']
    Lrelu_EGF_3 = Lrelu_res_3['EGF']
    avg, r1, r2, = get_res(Lrelu_EGF_1, Lrelu_EGF_2, Lrelu_EGF_3)
    PlotMultiFunc_var(x, avg, r1, r2)
    #relu EGF
    relu_EGF_1 = relu_res_1['EGF']
    relu_EGF_2 = relu_res_2['EGF']
    relu_EGF_3 = relu_res_3['EGF']
    avg, r1, r2, = get_res(relu_EGF_1, relu_EGF_2, relu_EGF_3)
    PlotMultiFunc_var(x, avg, r1, r2)
    #allrelu EGF
    allrelu_EGF_1 = allrelu_res_1['EGF']
    allrelu_EGF_2 = allrelu_res_2['EGF']
    allrelu_EGF_3 = allrelu_res_3['EGF']
    avg, r1, r2, = get_res(allrelu_EGF_1, allrelu_EGF_2, allrelu_EGF_3)
    PlotMultiFunc_var(x, avg, r1, r2)
    #grelu EGF
    grelu_EGF_1 = grelu_res_1['EGF']
    grelu_EGF_2 = grelu_res_2['EGF']
    grelu_EGF_3 = grelu_res_3['EGF']
    avg, r1, r2, = get_res(grelu_EGF_1, grelu_EGF_2, grelu_EGF_3)
    PlotMultiFunc_var(x, avg, r1, r2)
    plt.legend(['Leaky ReLU', 'ReLU', 'All-ReLU', 'GAReLU'])
    plt.xlabel("Epochs[#]")
    plt.ylabel("Effective Gradient Flow")
    plt.title('Effective Gradient Flow on Madelon')
    # plt.ylim([50, 90])
    plt.figure(2)
    plt.show()



