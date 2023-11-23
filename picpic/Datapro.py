import time
from PIL import Image
import pyautogui
import os



class DataProcessing(object):
    def __init__(self):
        pass






    def modelsavepng(self,path,savepath,n=8):
        #分辨率1920*1080或2560*1440，100%缩放 横向
        if (pyautogui.size().height == 1080 and pyautogui.size().width == 1920) or \
        (pyautogui.size().height == 1440 and pyautogui.size().width == 2560):
            pyautogui.PAUSE = 1
            pyautogui.hotkey('win', 'r')
            path = path
            Review_SNG_path = 'D:\magic\Review_SNG.exe'  # Review_SNG.exe路径
            pyautogui.typewrite(Review_SNG_path + ' ' + path)
            pyautogui.press('\n')
            # print('------------------------------------------------------------')
            pyautogui.press('n')
            pyautogui.press('n')
            pyautogui.press('n')
            time.sleep(n)  #等待n秒打开toc文件 取决于电脑运行速度
            #若不适配可手调下列click参数
            if pyautogui.size().height == 1080 and pyautogui.size().width == 1920:
                pyautogui.click(1658, 66)
                pyautogui.click(462, 55)
                pyautogui.click(441, 57)
            else:
                pyautogui.click(2239, 117)
                pyautogui.click(464, 58)
                pyautogui.click(442, 57)
            directory = savepath  # 保存的目录
            # filename = os.path.basename(path)
            dirStr, ext = os.path.splitext(path)
            file = dirStr.split("\\")[-1]
            path1 = directory + '\\' + file + '.png'  # 保存路径
            im = pyautogui.screenshot()
            try:
                im.save(path1)  # quality为图片质量，65为最低，95为最高
                print(file + '.png图片保存成功，保存在' + directory + "\n")
            except:
                print('图片保存失败')
            time.sleep(1)
            pyautogui.click(pyautogui.size().width-10, 7)

        else:
            print("分辨率错误，请调整至1920*1080或2560*1440")
