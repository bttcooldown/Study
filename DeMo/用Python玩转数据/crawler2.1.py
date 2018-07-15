# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:32:15 2018

@author: Administrator
"""

import requests, re, time
from bs4 import BeautifulSoup

count = 0
i = 0
sum, count_s = 0, 0
while(count < 10):
    try:
        r = requests.get('https://book.douban.com/subject/26264642/comments/hot?p=' + str(i+1))
    except Exception as err:
        print(err)
        break
    soup = BeautifulSoup(r.text, 'lxml')
    comments = soup.find_all('p', 'comment-content')
    for item in comments:
        count = count + 1
        print(count, item.string)
        if count == 10:
            break  
    pattern = re.compile('<span class="user-stars allstar(.*?) rating"')
    p = re.findall(pattern, r.text)
    for star in p:
        count_s = count_s + 1
        sum += int(star)
        if count_s == 10:
            print(sum/count_s)
            break
    time.sleep(5)    # delay request from douban's robots.txt
    i += 1
