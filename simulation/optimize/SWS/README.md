本模块用于自动生成慢波结构，以匹配给定的相速度。

# 设计阶段的草稿
## UniformSWSDesigner
用于设计均匀的慢波结构
### 输入：
desired_vph: 预期的相速度

f_desired: 预期的频率

param_names_to_adjust_mode: 通过调整哪些参数可以改变结构的运行模式？

param_name_of_Nslots: 哪个参数用于调整槽的个数？

get_vph(grd): 函数，用于从给定的grd中获取慢波结构内的相速度




