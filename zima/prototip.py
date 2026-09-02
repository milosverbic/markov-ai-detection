# ovo je prvi model koji sam ikad napravio i bukvalno je samo punio matrice brojem tranzicija iz reci u rec nije ih nikako skalirao

import time
t0 = time.time()
import pandas as pd

def euklidska(m1, m2, m, n):
    global reci50

    t = 0
    for i in reci50:
        for j in reci50:
            t += (m1[i][j] - m2[i][j])**2
    return t**0.5



N = 7321

reci100 = ['the', 'and', 'of', 'a', 'to', 'in', 'for', 'that', 'with', 'is', 'on', 'as', 'are', 'from', 'by', 'has', 'this', 'was',
 'it', 'at', 'new', 'his', 'their', 'have', 'be', 'an', 'not', 'but', 'or', 'more', 'he', 'about', 'who', 'i', 'will', 'we',
   'can', 'its', 'you', 'her', 'our', 'they', 'also', 'been', 'which', 'these', 'into', 'york', 'were', 'one', 'said', 'us', '—',
     'over', 'your', 'like', 'health', 'all', 'she', 'just', 'many', 'what', 'county', 'had', 'state', 'data', 'while', 'how',
       'cases', 'up', 'than', '-', 'people', 'election', 'world', 'trump', 'out', 'where', 'when', 'times', 'my', 'most', 'would',
         'time', 'through', 'some', 'only', 'may', 'could', 'other', 'community', 'both', 'public', 'covid-19', 'significant',
           'first', 'those', 'republican', 'if', 'so']
reciClean = [rec for rec in reci100 if rec not in ("york", "—", "health", "county", "state", "data", "cases", "-", "election", 
                                                   "trump", "covid-19", "republican")]
reci50 = [reciClean[i] for i in range(len(reciClean)) if i<50]

#df = pd.read_csv("train.csv")
#print(df)
#print(df.columns)

#ljudski = df["Human_story"]
#llama = df["llama-8B"]
#gpt = df["GPT_4-o"]

#ljudski.to_csv("ljudski.csv")
#llama.to_csv("llama.csv")
#gpt.to_csv("gpt.csv")


ljudski = pd.read_csv("ljudski.csv")["Human_story"]
llama = pd.read_csv("llama.csv")["llama-8B"]
gpt = pd.read_csv("gpt.csv")["GPT_4-o"]

"""
reci: dict = {}
for i in range(N):
    s = ljudski.iloc[i]
    #f = open("test.txt", "r")
    #s = f.read()
    #print(type(s), i)
    if type(s)==str:
        for rec in s.split(" "):
            rec2 = ""
            for c in rec:
                if c not in (",", ".", "?", "!"): rec2+=c.lower()
            if rec2 == "": continue
            try:
                reci[rec2] += 1
            except:
                reci[rec2] = 1
    #f.close()
    s = llama.iloc[i]
    #print(type(s), i)
    if type(s)==str:
        for rec in s.split(" "):
            rec2 = ""
            for c in rec:
                if c not in (",", ".", "?", "!"): rec2+=c.lower()
            if rec2 == "": continue
            try:
                reci[rec2] += 1
            except:
                reci[rec2] = 1
    s = gpt.iloc[i]
    #print(type(s), i)
    if type(s)==str:
        for rec in s.split(" "):
            rec2 = ""
            for c in rec:
                if c not in (",", ".", "?", "!"): rec2+=c.lower()
            if rec2 == "": continue
            try:
                reci[rec2] += 1
            except:
                reci[rec2] = 1
            
top100 = sorted(reci.items(), key=lambda item: item[1], reverse=True)[:100]
print(top100)
"""

ljM = {r: {r2: 0 for r2 in reci50} for r in reci50}
llM = {r: {r2: 0 for r2 in reci50} for r in reci50}
gM = {r: {r2: 0 for r2 in reci50} for r in reci50}

TN = 5000


for i in range(TN):
    p = ""
    s = ljudski.iloc[i]
    if type(s)==str:
        for rec in s.split(" "):
            rec2 = ""
            for c in rec:
                if c not in (",", ".", "?", "!"): rec2+=c.lower()
            if rec2 not in reci50:
                p = ""
                continue
            if p == "":
                p = rec2
                continue
            ljM[p][rec2] += 1/TN

for i in range(TN):
    p = ""
    s = gpt.iloc[i]
    if type(s)==str:
        for rec in s.split(" "):
            rec2 = ""
            for c in rec:
                if c not in (",", ".", "?", "!"): rec2+=c.lower()
            if rec2 not in reci50:
                p = ""
                continue
            if p == "":
                p = rec2
                continue
            gM[p][rec2] += 1/TN
            

predikcije = {x: {y: 0 for y in ("LJ", "G", "0")} for x in ("LJ", "G")}



for i in range(N-TN):
    m = {r: {r2: 0 for r2 in reci50} for r in reci50}
    p = ""
    s = ljudski.iloc[i+TN]
    if type(s)==str:
        for rec in s.split(" "):
            rec2 = ""
            for c in rec:
                if c not in (",", ".", "?", "!"): rec2+=c.lower()
            if rec2 not in reci50:
                p = ""
                continue
            if p == "":
                p = rec2
                continue
            m[p][rec2] += 1
    ljD = euklidska(ljM, m, 50, 50)
    gD = euklidska(gM, m, 50, 50)
    if ljD<gD: predikcije["LJ"]["LJ"] += 1
    elif ljD>gD: predikcije["LJ"]["G"] += 1
    elif ljD==gD: predikcije["LJ"]["0"] += 1



for i in range(N-TN):
    m = {r: {r2: 0 for r2 in reci50} for r in reci50}
    p = ""
    s = gpt.iloc[i+TN]
    if type(s)==str:
        for rec in s.split(" "):
            rec2 = ""
            for c in rec:
                if c not in (",", ".", "?", "!"): rec2+=c.lower()
            if rec2 not in reci50:
                p = ""
                continue
            if p == "":
                p = rec2
                continue
            m[p][rec2] += 1
    ljD = euklidska(ljM, m, 50, 50)
    gD = euklidska(gM, m, 50, 50)
    if ljD<gD: predikcije["G"]["LJ"] += 1
    elif ljD>gD: predikcije["G"]["G"] += 1
    else: predikcije["G"]["0"] += 1
    

print(predikcije)

print(time.time()-t0)