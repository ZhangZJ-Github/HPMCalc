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

from _logging import logger


class ConfigBase:
    DEFAULT_PATH = os.path.join(os.path.split(__file__)[0], 'config.json')
    DEFAULT_ENCODING = 'utf-8'

    class ItemNames(Enum):
        Magic_executable_path = auto()
        General_Particle_Tracer_bin_dir = auto()
        General_Particle_Tracer_license = auto()

    str_itemnames: typing.List[str] = [itemname.name for itemname in ItemNames]

    default_items = {
        ItemNames.Magic_executable_path.name: r"G:\Program Files\Magic Tools\magic2d_Sng.exe",
        ItemNames.General_Particle_Tracer_bin_dir.name: r"G:\Program Files\General Particle Tracer\bin",
        ItemNames.General_Particle_Tracer_license.name: r"1234567890",

    }
    meaning = {
        ItemNames.Magic_executable_path.name: r"Magic可执行文件的路径",
        ItemNames.General_Particle_Tracer_bin_dir.name: r"General Particle Tracer的二进制可执行文件所在的文件夹路径",
        ItemNames.General_Particle_Tracer_license.name: r"General Particle Tracer的License号码（默认值仅供示例，并非有效的license）",
    }


class Config(ConfigBase):

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

        s = '含义：%s, 默认值："%s"' % (
        Config.meaning.get(item_name, "（开发者暂未填写该项含义）"), Config.default_items.get(item_name, "(无默认值)"))
        return s

    def to_json_file(self, filepath=ConfigBase.DEFAULT_PATH):
        with open(filepath, 'w', encoding=Config.DEFAULT_ENCODING) as f:
            json.dump(self.items, f, separators=(',\n', ':'), )

        logger.info('已将配置文件写入"%s"' % (os.path.abspath(filepath)))

    @staticmethod
    def read_json_file(filepath=ConfigBase.DEFAULT_PATH):
        with open(filepath, 'r', encoding=Config.DEFAULT_ENCODING) as f:
            _items = json.load(f, )
        return Config(_items)

    def get_value(self, itemname: typing.Union[str, ConfigBase.ItemNames]):
        if isinstance(itemname, Config.ItemNames):
            itemname = itemname.name
        return self.items[itemname]

    @staticmethod
    def generate_configuration_interactively():
        """
        交互式地生成配置
        :return:
        """
        cfg = Config({})
        for item in cfg.items:
            user_defined_value = input('设置配置文件中"%s"的值，\n%s\n（若直接按下回车键，表示采用默认值）' % (item, cfg.about(item),)
                                       )
            if len(user_defined_value) == 0:
                user_defined_value = cfg.default_items[item]
            cfg.items[item] = user_defined_value
        return cfg


if __name__ == '__main__':
    cfg = Config.generate_configuration_interactively()
    cfg.to_json_file()  # 生成默认的配置文件

    # cfg = Config.read_json_file()
