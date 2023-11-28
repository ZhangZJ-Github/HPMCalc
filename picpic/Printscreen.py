from Datapro import DataProcessing
import fld_parser
import par_parser
import grd_parser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
import pyautogui
import os

if __name__ == '__main__':
    ################################################
    #依次读取文件夹中的.toc文件（二选一）

    # folderpath = r"F:\11-11\1" #需要修改的文件所在的路径
    # savepath=r"E:\11-18\savepng" #保存路径
    # csvname = 'index.csv'
    # DP=DataProcessing()
    # DP.foldersave(folderpath, savepath, csvname)


    ##################################################
    #读取csv中的文件（二选一）

    csvpath = r"e:\generatoraccelerator\genac\optmz\genac20g100kev\RoughMesh\genac20g100kev.m2d.log.csv"
    savepath=r"e:\generatoraccelerator\genac\optmz\genac20g100kev\RoughMesh\ScreenShot"
    csvname = 'index.csv'
    DP = DataProcessing()
    DP.csvsave(csvpath,savepath,csvname )






