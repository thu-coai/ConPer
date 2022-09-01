import os
import sys


class Config:

    def __init__(self, args=None, file=None):
        self.py_name, _ = os.path.splitext(file)

        if args is not None:
            for k, v in args.__dict__.items():
                self.__setattr__(k, v)

    def show(self):
        for name, value in vars(self).items():
            print(f"{name}={value}")

    def add_display(self, name):
        if hasattr(self, name):
            return f'_{name}{getattr(self, name)}'
        else:
            return ''

    def get_generate_out_file_name(self):
        res = self.py_name
        names = ['version_num', 'sent']
        for name in names:
            res += self.add_display(name)
        return res


if __name__ == '__main__':
    print(__file__)
    config = Config(file=__file__)
