# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/10 22:58
@Auth ： Zi-Jing Zhang (张子靖)
@File ：test_EMcomplex.py
@IDE ：PyCharm
"""
import fastdesign.microwave_source.EMcomplex as EMC
def test_MagicContourRebuilder():
    rebuilder  = EMC.MagicContourRebuilder(r"E:\BigFiles\GENAC\GENACX50kV\手动\GenacX50kV_tmplt_20240210_051249_02\GenacX50kV_tmplt_20240210_051249_02 周期内密集采样4.toc")
    rebuilder.reconstruct_from_contours(' FIELD_POWER S.DA @COUPLER.PORT,FFT-#5.1', interested_frequency=9e9)

if __name__ == '__main__':
    test_MagicContourRebuilder()
