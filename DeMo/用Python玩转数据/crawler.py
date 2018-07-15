# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:32:15 2018

@author: Administrator
"""

import requests
from bs4 import BeautifulSoup
import re
sum = 0
r = requests.get('https://book.douban.com/subject/26264642/')
#soup = BeautifulSoup(r.text,'lxml')
#pattern = soup.find_all('p','comment-content')
#for item in pattern:
 #   print(item.string)
pattern_s = re.compile('span class="user-stars allstar(.*?) rating"')
p = re.findall(pattern_s,r.text)
for star in p:
    sum += int(star)
print(sum)