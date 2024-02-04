# HPMCalc: 高功率微波源设计计算器
# 安装
## 依赖

- [ ] MAGIC: 一种Particle-In-Cell模拟软件，Ref: Goplen B, Ludeking L, Smith D, et al. User-configurable MAGIC for electromagnetic PIC calculations\[J\]. Computer Physics Communications, 1995, 87(1-2): 54-86.
- [ ] [pyMagicTools](https://gitee.com/zhangzj-gitee/MagicTools): 用于解析MAGIC结果文件的Python库。

以下依赖软件、库主要用于粒子加速器的设计，若仅考虑微波源设计则暂时不用：
- [ ] General Particle Tracer (简称：GPT) ：一种粒子追踪框架。
- [ ] [pygpt](https://gitlab.com/sgeer/pygpt): Python parser for GPT output files (.gdf). This code is developed by Bas van der Geer, the author of General Particle Tracer.
- [ ] CST
## 配置
运行[simulation/conf/config.py](simulation/conf/config.py)生成本地配置（./simulation/conf/config.json）。
此操作会覆盖目录下现有的config.json，因此建议操作前先复制或重命名该文件。

# 快速入门
在[simulation/template](simulation/template)文件夹下有许多之前的优化任务相关文件，可作为示例参考。
其中的大部分任务是可以直接运行的，或只需将相关的文件路径改为自己计算机上存在的路径。
更妥当的方法是，根据git记录，找到此文件夹下最新的main.py文件开始运行。

一个优化任务通常包含以下几个要素：

- [ ] 输入文件模板，如**XXX.m2d**

    在MAGIC输入文件的基础上略做修改得到。通常是把其中待优化的参数用百分号（%）包裹，例如，原始输入文件中内容如下
    ```
    sws1.N =   3; 
    sws1.p =   4 mm;
    ```
    而我们希望将这两个参数作为待优化变量，那么，只需复制原始输入文件，并将这两行修改为
    ```
    sws1.N =   %sws1.N%;
    sws1.p =   %sws1.p% mm;
    ```
    这代表我们设置了两个待优化变量：`%sws1.p%`和`%sws1.p%`
    这些变量名一般无需再手动输入到其他地方，详见[初始化表](##初始化表)。
    
- [ ] 优化任务入口（**main.py）**，用于指定优化任务的细节，HPM源相关的优化任务一般包括以下内容：
    - [ ]  输入文件模板的路径
    - [ ]  大量输入文件和结果文件的保存位置；
    - [ ]  如何从生成的结果文件中提取我们关心的中间结果，如平均功率、频率等（通过重写TaskBase.get_res方法）；
    - [ ]  如何根据这些指标进一步计算出优化器关心的目标函数值（通过重写TaskBase.evaluate方法）；
    - [ ]  如何实现参数检查（通过重写TaskBase.param_check），即遇到怎样的参数时，不需执行模拟，直接返回低分结果；
    - [ ]  其他优化器细节，如并行进程数目、每代个体数目、学习率等。
- [ ] 初始值表（通常命名为**initialize.csv**）    
    指定优化任务的初始值、精度、边界等。手动构建这个表是很繁琐的，但我们提供了许多函数来简化这些步骤，详见[初始化表](##初始化表)。


# 运行
## 初始化表
如前所述，输入模板文件中，待优化的参数用百分号（%）包裹。

优化框架中有能够识别按这种方式命名的变量函数，用户可以灵活地调用它们，以自动生成一个初始化表。实现这一功能的最短代码示例位于[simulation/task_manager/test/test_initializer/test_initializer.py](simulation/task_manager/test/test_initializer/test_initializer.py)。运行此文件后，程序会调用系统默认的软件（如Excel或WPS）打开initial.csv供用户编辑。可以新增多行初始值，他们会在优化开始后全部被执行。因此，这也可以作为参数扫描的配置文件。
并非所有都是必填项。非必填项留空时，会采用默认值。上界和下界默认是最后一组初始值的$\pm10\%$，精度默认是0（这表示两个参数的数值必须精确相等，才会认为是重复的），最后两列是为一些目前不完全支持的优化器准备的，暂时用不到。

![4876881d38128230af36407ca3ccea20.png](.md_attachments/4876881d38128230af36407ca3ccea20.png)

# 联系
在配置、运行本套代码时，遇到任何问题，欢迎提issue或联系作者(zijingzhang@mail.ustc.edu.cn)。