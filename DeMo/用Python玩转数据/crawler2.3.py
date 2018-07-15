# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:00:25 2018

@author: Administrator
"""

import requests
import re
r = requests.get('http://italy2014.fivb.org/en/competition/results_and_statistics')
r.encoding = r.apparent_encoding
pattern = re.compile('<td><a id="wcbody_0_wcgridpade50e7ca82ec64ee2b91ea4cc6c4e00c6_1_PlayerStatisticsTable_BestScorers_Name_.*?" href="/en/competition\/teams\/.*?\/players/.*?id=.*?">(.*?)</a></td>\s+<td id="wcbody_0_wcgridpade50e7ca82ec64ee2b91ea4cc6c4e00c6_1_PlayerStatisticsTable_BestScorers_TeamCell_.*?"><a id="wcbody_0_wcgridpade50e7ca82ec64ee2b91ea4cc6c4e00c6_1_PlayerStatisticsTable_BestScorers_Team_.*?" href="/en/competition/teams/.*?">(.*?)</a></td>\s+<td>(.*?)</td>\s+<td>(.*?)</td>\s+<td>(.*?)</td>\s+<td>(.*?)</td>',flags=re.M)
p = re.findall(pattern, r.text)
print(p)