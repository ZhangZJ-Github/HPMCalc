# -*- coding: utf-8 -*-
# @Time    : 2023/6/13 14:26
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _test.py
# @Software: PyCharm
from myPySldWrap import sw_tools
from pathlib import Path
sw_tools.connect_sw('2020')
with sw_tools.EditPart( Path(r"D:\MagicFiles\HPM\12.5GHz\HPM-全金属外壳.SLDPRT")) as model:
    sw_tools.edit_dimension_sketch(model, '草图2', "NP_SWS1", 3)
    # sw_tools.edit_dimension_sketch(model, '草图2', 'D17', 3)
