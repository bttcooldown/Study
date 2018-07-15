# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:55:02 2018

@author: Administrator
"""

    
with open('Blowing in the wind.txt', 'r+') as f:
    lines = f.readlines()
    lines.insert(0, "Blowin' in the wind\n")
    lines.insert(1, "Bob Dylan\n")
    lines.append("1962 by Warner Bros. Inc.")
    string = ''.join(lines)
    print(string)
    f.seek(0)
    f.write(string)

    