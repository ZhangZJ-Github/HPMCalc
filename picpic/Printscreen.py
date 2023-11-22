from Datapro import DataProcessing
import os   #导入os模块


if __name__ == '__main__':
    savepath=r"E:\11-18\3" #保存路径
    path = r"E:\11-18\3" #需要修改的文件所在的路径
    original_name = os.listdir(path)        #读取文件初始的名字
    #print(original_name)
    DP=DataProcessing()
    for i in original_name:                 #遍历全部文件
        if os.path.splitext(i)[-1]=='.toc':
            DP.modelsavepng(path+'\\'+i,savepath)   #保存模型图为.png
