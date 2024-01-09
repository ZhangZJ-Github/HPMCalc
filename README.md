# 高功率微波源设计计算器
## 安装
### 依赖

Magic: 一种PIC 模拟软件

[MagicTools](https://gitee.com/zhangzj-gitee/MagicTools): 用于解析Magic产生的结果文件的Python库

General Particle Tracer
### 配置
运行./simulation/conf/config.py生成本地配置。
## 运行
### 运行一个新的优化任务
在simulation/templates文件夹下新建文件夹，用于存放此优化相关的模板（XXX.m2d）、数据处理代码（main.py）、初始值表（initialize.csv）。
#### main.py
其中，main.py中新建一个类，继承HPMSimWithInitializer，重写以下方法：

get_res # 告诉程序如何从.grd .fld .par等结果文件中获取所需的数据

evaluate # 告诉程序如何将get_res获得的数据转化为评分

params_check # 告诉程序如何检查一组参数是否可用


