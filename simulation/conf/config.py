# -*- coding: utf-8 -*-
# @Time    : 2023/11/6 13:31
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : config.py
# @Software: PyCharm
"""
给新用户提示，以便其创建专属的配置文件；
解析配置文件。
"""
import json
import os
import typing
from enum import Enum, auto


class Config:
    DEFAULT_PATH = os.path.join(os.path.split(__file__)[0], 'config.json')
    DEFAULT_ENCODING = 'utf-8'

    class ItemNames(Enum):
        Magic_executable_path = auto()
        General_Particle_Tracer_bin_dir = auto()
        General_Particle_Tracer_license = auto()

    str_itemnames: typing.List[str] = [itemname.name for itemname in ItemNames]

    item_prompt = {ItemNames.Magic_executable_path.name: 'Magic可执行文件的路径',
                   ItemNames.General_Particle_Tracer_bin_dir.name: r'General Particle Tracer的二进制文件位置，如"G:\Program Files\General Particle Tracer\bin"'}
    default_items = {
        ItemNames.Magic_executable_path.name: r"G:\Program Files\Magic Tools\magic2d_Sng.exe",
        ItemNames.General_Particle_Tracer_bin_dir.name: r"G:\Program Files\General Particle Tracer\bin",
        ItemNames.General_Particle_Tracer_license .name : r"123456789",

    }

    @staticmethod
    def is_valid_items(items: typing.Dict[str, typing.Any]) -> bool:
        if set(items.keys()).issubset(set(Config.str_itemnames)):
            return True
        return False

    def __init__(self, items: dict):
        self.items = Config.default_items
        if not Config.is_valid_items(items):
            raise RuntimeError("配置无效：\n%s" % items)
        self.items.update(items)

    @staticmethod
    def about(item_name: str):
        return Config.item_prompt.get(item_name, '')

    def to_json_file(self, filepath=DEFAULT_PATH):
        with open(filepath, 'w', encoding=Config.DEFAULT_ENCODING) as f:
            json.dump(self.items, f, separators=(',\n',':'),)

    @staticmethod
    def read_json_file(filepath=DEFAULT_PATH):
        with open(filepath, 'r', encoding=Config.DEFAULT_ENCODING) as f:
            _items = json.load(f, )
        return Config(_items)

    def get_value(self, itemname: typing.Union[str, ItemNames]):
        if isinstance(itemname, Config.ItemNames):
            itemname = itemname.name
        return self.items[itemname]


if __name__ == '__main__':
    Config({}).to_json_file() # 生成默认的配置文件
    # cfg = Config.read_json_file()
