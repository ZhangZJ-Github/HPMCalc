import time
from PIL import Image
import pyautogui
import os



class DataProcessing(object):
    def __init__(self):
        pass






    def modelsavepng(self,path,savepath):
        if pyautogui.size().height == 1080 and pyautogui.size().width == 1920:
            pyautogui.PAUSE = 1
            pyautogui.hotkey('win', 'r')
            path = path
            Review_SNG_path='D:\magic\Review_SNG.exe' #Review_SNG.exe路径
            pyautogui.typewrite(Review_SNG_path + ' ' + path) 
            pyautogui.press('\n')
            # print('------------------------------------------------------------')
            pyautogui.press('n')
            pyautogui.press('n')
            pyautogui.press('n')
            time.sleep(7)
            pyautogui.click(1658, 66)
            pyautogui.click(462, 55)
            pyautogui.click(441, 57)
            directory = savepath  # 保存的目录
            # filename = os.path.basename(path)
            dirStr, ext = os.path.splitext(path)
            file = dirStr.split("\\")[-1]
            path1 = directory + '\\' + file + '.png'  # 保存路径
            im = pyautogui.screenshot()
            try:
                im.save(path1)  # quality为图片质量，65为最低，95为最高
                print('图片保存成功，保存在' + directory + "\n")
            except:
                print('图片保存失败')
            time.sleep(1)
            pyautogui.click(1910, 7)

        else:
            print("分辨率错误，请调整至1920*1080")




if __name__ == '__main__':
    # pyautogui.size()
    # pyautogui.position()
    # sizex,sizey=pyautogui.size()
    # pyautogui.moveTo(sizex / 2, sizey / 2, duration=1)
    # pyautogui.moveRel(100, -200, duration=0.5)
    #pyautogui.click(sizex / 2, sizey / 2, duration=0.5)
    #pyautogui.click(button='right')
    # pyautogui.click(100, 100, clicks=3, interval=0.1, duration=0.5)
    # pyautogui.click(100, 100, clicks=2, interval=0.5, button='right', duration=0.2)
    # # pyautogui.doubleClick(10, 10)  # 指定位置，双击左键
    # # pyautogui.rightClick(10, 10)  # 指定位置，双击右键
    # # pyautogui.middleClick(10, 10)  # 指定位置，双击中键
    # # 鼠标位置不动，向上回滚2个单位，项目文档对滚动量参数说明不详
    # pyautogui.scroll(2)
    #
    # # 鼠标移动至(1000,700)，前下滚动10个单位
    # # 运行发现鼠标并没有动
    # pyautogui.scroll(-10, 1000, 700)
    #
    # # 将鼠标从当前位置拖至屏幕中心，默认左键
    # pyautogui.dragTo(sizex / 2, sizey / 2)
    #
    # # 将鼠标从当前位置向左100像素、向右200像素拖动，过渡时间0.5秒，指定右键
    # pyautogui.dragRel(-100, 200, duration=0.5, button='right')
    # pyautogui.mouseDown()  # 鼠标按下
    # pyautogui.mouseUp()  # 鼠标释放
    # pyautogui.scroll(300)  # 向下滚动300个单位；
    # # 拖动到指定位置
    # # 将鼠标拖动到指定的坐标；duration 的作用是设置移动时间，所有的gui函数都有这个参数，而且都是可选参数
    # pyautogui.dragTo(100, 300, duration=1)
    #
    # # 按方向拖动
    # # 向右拖动100px，向下拖动500px, 这个过程持续 1 秒钟
    # pyautogui.dragRel(100, 500, duration=4)  # 第一个参数是左右移动像素值，第二个是上下

    # pyautogui.keyDown() ： 模拟按键按下；
    # pyautogui.keyUp() ： 模拟按键释放；
    # pyautogui.press() ：  # 就是调用keyDown() & keyUp(),模拟一次按键；
    # pyautogui.typewrite('this', 0.5) ： 第一参数是输入内容，第二个参数是每个字符间的间隔时间；
    # pyautogui.typewrite(['T', 'h', 'i', 's'])：typewrite

    # pyautogui.keyDown('shift')  # 按下shift
    # pyautogui.press('4')  # 按下 4
    # pyautogui.keyUp('shift')  # 释放 shift
    # pyautogui.typewrite('$*……%……￥', 0.5)
    #
    # #CTRL+c
    # pyautogui.keyDown('ctrl')
    # pyautogui.keyDown('c')
    # pyautogui.keyUp('c')
    # pyautogui.keyUp('ctrl')
    #
    # pyautogui.hotkey('ctrl', 'c')
    #
    #
    # im = pyautogui.screenshot()  # 返回屏幕的截图，是一个Pillow的image对象
    # im.getpixel((500, 500))  # 返回im对象上，（500，500）这一点像素的颜色，是一个RGB元组
    # pyautogui.pixelMatchesColor(500, 500, (12, 120, 400))  # 是一个对比函数，对比的是屏幕上（500，500）这一点像素的颜色，与所给的元素是否相同；

    if pyautogui.size().height==1080 and pyautogui.size().width==1920:
        pyautogui.PAUSE = 1
        pyautogui.hotkey('win', 'r')
        path=r"E:\11-18\3\Genac10G50keV-1.toc"
        pyautogui.typewrite('D:\magic\Review_SNG.exe '+path)
        pyautogui.press('\n')
        #print('------------------------------------------------------------')
        pyautogui.press('n')
        pyautogui.press('n')
        pyautogui.press('n')
        time.sleep(7)
        pyautogui.click(1658,66)
        pyautogui.click(462, 55)
        pyautogui.click(441, 57)
        directory = os.path.dirname(path) #保存的目录
        # filename = os.path.basename(path)
        dirStr, ext = os.path.splitext(path)
        file = dirStr.split("\\")[-1]
        path1 = directory +'\\'+ file + '.png'  # 保存路径
        im = pyautogui.screenshot()
        try:
            im.save(path1)  # quality为图片质量，65为最低，95为最高
            print('图片保存成功，保存在' + directory + "\n")
        except:
            print('图片保存失败')
        time.sleep(1)
        pyautogui.click(1910, 7)

    else:
        print("分辨率错误，请调整至1920*1080")












