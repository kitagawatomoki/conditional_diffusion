from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import glob2
import re

# htmlコードから漢字のみを抜き取る処理
data = []
with open('sample.txt') as f:
        for line in f:
            x = re.findall('html">.*</a></li>', line)
            if len(x) !=0:
                x = x[0].replace('html">', '')
                # x = x.replace('"', '')
                # x = x.replace('>', '')
                x = x.replace('</a></li>', '\n')
                data.append(x)

# テキストファイルにリストの値を書き込む処理
with open('jis2.txt', 'w', encoding='utf-8', newline='\n') as f:
    f.writelines(data)