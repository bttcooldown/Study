# -*- coding: utf-8 -*-
"""
Created on Sat May 19 21:19:41 2018

@author: Administrator
"""
import requests
from urllib.parse import urlencode
from requests.exceptions import RequestException
import json
from bs4 import BeautifulSoup

def get_page_index(offset,keyword):
    data={
        'offset':offset,
        'format':'json',
        'keyword':'keyword',
        'autoload':'true',
        'count':'20',
        'cur_tab':1
          }
    url = 'https://www.toutiao.com/search_content/?' + urlencode(data)
    try:
        response = requests.get(url)
        if response.status_code ==200:
            return response.text
        return None
    except RequestException:
        print('失败')
        return None

def parse_page_index(html):
    data = json.loads(html)        
    if data and 'data' in data.keys():
        for item in data.get('data'):
            yield item.get('article_url')
        
def get_page_detail(url):
    try:
        response = requests.get(url)
        if response.status_code ==200:
            return response.text
        return None
    except RequestException:
        print('请求详情页失败',url)
        return None
            
def parse_page_detail(html):
    soup = BeautifulSoup(html,'lxml')
    title = soup.select('title')[0].get_text()
            
        
def main():
    html = get_page_index(0,'街拍')
    for url in parse_page_index(html):
        html = get_page_detail(url)

if __name__ == '__main__':
    main()    
