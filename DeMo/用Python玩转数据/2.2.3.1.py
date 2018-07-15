# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def countchar(s):
    lst = [0]*26
    for i in range(len(s)):
        if s[i]>='a' and s[i]<='z':
            lst[ord(s[i])-ord('a')] += 1
    print(lst)

#print('The string is:{0,c}'.format(ss=input()))
ss = input("The string is:")
#ss = "Hope is a good thing."
ss = ss.lower()
countchar(ss)


