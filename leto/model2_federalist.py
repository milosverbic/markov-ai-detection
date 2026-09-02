import time
t0 = time.time()
import pandas as pd
from copy import deepcopy
import random as R
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path


def euklidska(m1: dict, m2: dict):
    t = 0
    for i in m1.keys():
        for j in m1[i].keys():
            t += (m1[i][j] - m2[i][j])**2
    return t**0.5

def logLikelihood(text: str, m: dict):
    lll = 0
    p = ""
    for Rec in text.split():
        rec = ""
        for c in Rec:
            if c not in (",", ".", "?", "!", ";", ":"): rec+=c.lower()
        if rec not in RECI:
            if p!="":
                lll += np.log(m[p]["OTHER"])
            p = ""
            continue
        if p == "":
            p = rec
            continue
        lll += np.log(m[p][rec])
        p = rec
    return lll

def newChain(alpha=0):
    m = {r: {r2: alpha for r2 in RECI} for r in RECI}
    for r in RECI: m[r]["OTHER"] = alpha
    return m


def buildChain(text: str, M): # zapravo dodaje broj parova na onaj koji je vec u matrici (0 za novu matricu); matrica koja nastane nije normalizovana
    p = ""
    nReci = 0
    for Rec in text.split():
        nReci += 1
        rec = ""
        for c in Rec:
            if c not in (",", ".", "?", "!", ";", ":"): rec+=c.lower()
        if rec not in RECI:
            if p!="":
                M[p]["OTHER"] += 1
            p = ""
            continue
        if p == "":
            p = rec
            continue
        M[p][rec] += 1
        p = rec
    return nReci


def trainFromDF(sample: set | range, data: pd.DataFrame, M):
    for i in sample:
        s = data.iloc[i]
        if type(s)==str:
            buildChain(s, M)

def trainFromFolder(folder_path, M, exclude_path=""):
    # Get full Path objects
    full_paths = [
        f.resolve() for f in folder_path.rglob('*') if f.is_file()
    ]
    for path in full_paths:
        if path != exclude_path:
            with open(path, 'r') as f:
                text = f.read()
                buildChain(text, M)


def testOnDF(sample: set | range, data: pd.DataFrame, lanci: list, label=""):
    R = {x:0 for x in LABELI}
    R['-'] = 0
    for i in sample:
        s = data.iloc[i]
        if type(s)!=str:
            R["-"] += 1
            continue
        author = testOnText(s, lanci, label)
        R[author] += 1
    return R

def testOnFolder(folder_path, lanci, label=""):
    R = {x:0 for x in LABELI}
    # Get full Path objects
    full_paths = [
        f.resolve() for f in folder_path.rglob('*') if f.is_file()
    ]
    for path in full_paths:
        # id = str(path)[-6:-4]
        with open(path, 'r') as f:
            text = f.read()
            author = testOnText(text, lanci, label)
            R[author] += 1
    
    return R

def testOnText(text: str, lanci, label="", alg="euklidska"):
    if alg=="euklidska":
        m = newChain()
        nReci = buildChain(text, m)
        normalize(m)
        # mappc(scale, m, 1/nReci)
        distances = [euklidska(l, m) for l in lanci]
        minD = min(distances)
        predictionIndex = distances.index(minD)
    elif alg=="logLikelihood":
        probabilities = [logLikelihood(text, l) for l in lanci]  
        maxP = max(probabilities)
        predictionIndex = probabilities.index(maxP)

    prediction = LABELI[predictionIndex]

    if alg=="euklidska":
        print(distances, prediction)
    elif alg=="logLikelihood":
        print(probabilities, prediction)
    # dictToCsv(m, "leto/lanci/dispt/dispt"+id+"_"+prediction+".csv")
    return prediction



def mapp(f, M: dict, *args):   # mislim da mi ne treba za m-2?
    M2 = deepcopy(M)
    for rec in M.keys():
        for rec2 in M.keys():
            M2[rec][rec2] = f(M[rec][rec2], *args)
    return M2

def mappc(f, M: dict, *args):   # mislim da mi ne treba za m-2? (ovo je mapp() koji ne vraca novu matricu nego pravi promene na datoj)
    for rec in M.keys():
        for rec2 in M.keys():
            M[rec][rec2] = f(M[rec][rec2], *args)

def scale(x, s):
    return x*s

def addDicts(a: dict, b: dict):
    D = {}
    for k in a.keys():
        D[k] = a[k] + b[k]
    return D

def mergeChains(a: dict, b: dict, w1=0.5, w2=0.5):
    D = {}
    for k in a.keys():
        D[k] = {}
        for l in a[k].keys():
            D[k][l] = w1*a[k][l] + w2*b[k][l]
    return D

def normalize(M: dict):
    for k in M.keys():
        s = sum(M[k].values())
        if s>0:
            for k2 in M[k].keys():
                M[k][k2] *= 1/s

def dictToCsv(dict, filename):
    df = pd.DataFrame.from_dict(dict)
    df.to_csv(filename)


# reci100 = ['the', 'and', 'of', 'a', 'to', 'in', 'for', 'that', 'with', 'is', 'on', 'as', 'are', 'from', 'by', 'has', 'this', 'was',
#  'it', 'at', 'new', 'his', 'their', 'have', 'be', 'an', 'not', 'but', 'or', 'more', 'he', 'about', 'who', 'i', 'will', 'we',
#    'can', 'its', 'you', 'her', 'our', 'they', 'also', 'been', 'which', 'these', 'into', 'york', 'were', 'one', 'said', 'us', '—',
#      'over', 'your', 'like', 'health', 'all', 'she', 'just', 'many', 'what', 'county', 'had', 'state', 'data', 'while', 'how',
#        'cases', 'up', 'than', '-', 'people', 'election', 'world', 'trump', 'out', 'where', 'when', 'times', 'my', 'most', 'would',
#          'time', 'through', 'some', 'only', 'may', 'could', 'other', 'community', 'both', 'public', 'covid-19', 'significant',
#            'first', 'those', 'republican', 'if', 'so']
reci100 = ['the', 'of', 'to', 'and', 'in', 'a', 'be', 'that', 'it', 'is', 'which', 'as', 'by', 'this', 'would', 'have', 'or', 'for',
     'not', 'will', 'their', 'with', 'from', 'are', 'an', 'they', 'on', 'states', 'been', 'government', 'may', 'state', 'all', 'but',
       'its', 'power', 'has', 'other', 'if', 'at', 'more', 'than', 'them', 'any', 'one', 'people', 'no', 'those', 'there', 'we', 'constitution',
         'these', 'can', 'who', 'must', 'upon', 'such', 'so', 'union', 'most', 'his', 'national', 'should', 'i', 'same', 'was', 'new', 'might',
           'against', 'every', 'our', 'under', 'authority', 'into', 'federal', 'only', 'had', 'great', 'were', 'powers', 'public', 'general',
             'shall', 'executive', 'could', 'ought', 'between', 'united', 'some', 'time', 'us', 'men', 'what', 'body', 'part', 'he', 'each',
               'particular', 'less', 'members']
# reciClean = [rec for rec in reci100 if rec not in ("york", "—", "health", "county", "state", "data", "cases", "-", "election", 
#                                                    "trump", "covid-19", "republican")]
reciClean = reci100

NR = 50  # NR<=88 za stare reci, NR<=100 za nove reci
reciNR = [reciClean[i] for i in range(len(reciClean)) if i<NR]
RECI = reciNR

ALPHA = 0

LABELI = ["HAMILTON", "MADISON", "JAY", "DISPUTED"]

HM = newChain(alpha=ALPHA)
MM = newChain(alpha=ALPHA)
JM = newChain(alpha=ALPHA)
DM = newChain(alpha=ALPHA)

lanci = [HM, MM, JM, DM]


script_dir = Path(__file__).parent 
HAMILTON_PATH = script_dir / "FedPapersCorpus" / "hamilton"
MADISON_PATH = script_dir / "FedPapersCorpus" / "madison"
JAY_PATH = script_dir / "FedPapersCorpus" / "jay"
DISPUTED_PATH = script_dir / "FedPapersCorpus" / "dispt"

hamilton_file_paths = [f.resolve() for f in HAMILTON_PATH.rglob('*') if f.is_file()]
madison_file_paths = [f.resolve() for f in MADISON_PATH.rglob('*') if f.is_file()]
jay_file_paths = [f.resolve() for f in JAY_PATH.rglob('*') if f.is_file()]
dispt_file_paths = [f.resolve() for f in DISPUTED_PATH.rglob('*') if f.is_file()]

H = {x:0 for x in LABELI}
for fp in hamilton_file_paths:
    HM = newChain(alpha=ALPHA)
    MM = newChain(alpha=ALPHA)
    JM = newChain(alpha=ALPHA)
    DM = newChain(alpha=ALPHA)
    lanci = [HM, MM, JM, DM]
    trainFromFolder(HAMILTON_PATH, HM, exclude_path=fp)
    trainFromFolder(MADISON_PATH, MM)
    trainFromFolder(JAY_PATH, JM)
    trainFromFolder(DISPUTED_PATH, DM)
    normalize(HM)
    normalize(MM)
    normalize(JM)
    normalize(DM)
    with open(fp, 'r') as f:
        text = f.read()
        author = testOnText(text, lanci, alg="euklidska")
        H[author] += 1


M = {x:0 for x in LABELI}
for fp in madison_file_paths:
    HM = newChain(alpha=ALPHA)
    MM = newChain(alpha=ALPHA)
    JM = newChain(alpha=ALPHA)
    DM = newChain(alpha=ALPHA)
    lanci = [HM, MM, JM, DM]
    trainFromFolder(HAMILTON_PATH, HM)
    trainFromFolder(MADISON_PATH, MM, exclude_path=fp)
    trainFromFolder(JAY_PATH, JM)
    trainFromFolder(DISPUTED_PATH, DM)
    normalize(HM)
    normalize(MM)
    normalize(JM)
    normalize(DM)
    with open(fp, 'r') as f:
        text = f.read()
        author = testOnText(text, lanci, alg="euklidska")
        M[author] += 1


J = {x:0 for x in LABELI}
for fp in jay_file_paths:
    HM = newChain(alpha=ALPHA)
    MM = newChain(alpha=ALPHA)
    JM = newChain(alpha=ALPHA)
    DM = newChain(alpha=ALPHA)
    lanci = [HM, MM, JM, DM]
    trainFromFolder(HAMILTON_PATH, HM)
    trainFromFolder(MADISON_PATH, MM)
    trainFromFolder(JAY_PATH, JM, exclude_path=fp)
    trainFromFolder(DISPUTED_PATH, DM)
    normalize(HM)
    normalize(MM)
    normalize(JM)
    normalize(DM)
    with open(fp, 'r') as f:
        text = f.read()
        author = testOnText(text, lanci, alg="euklidska")
        J[author] += 1


D = {x:0 for x in LABELI}
for fp in dispt_file_paths:
    HM = newChain(alpha=ALPHA)
    MM = newChain(alpha=ALPHA)
    JM = newChain(alpha=ALPHA)
    DM = newChain(alpha=ALPHA)
    lanci = [HM, MM, JM, DM]
    trainFromFolder(HAMILTON_PATH, HM)
    trainFromFolder(MADISON_PATH, MM)
    trainFromFolder(JAY_PATH, JM)
    trainFromFolder(DISPUTED_PATH, DM, exclude_path=fp)
    normalize(HM)
    normalize(MM)
    normalize(JM)
    normalize(DM)
    with open(fp, 'r') as f:
        text = f.read()
        author = testOnText(text, lanci, alg="euklidska")
        D[author] += 1


print("NAPISAO HAMILTON, PREDICTOVAO HAMILTON:", H["HAMILTON"])
print("NAPISAO HAMILTON, PREDICTOVAO MADISON:", H["MADISON"])
print("NAPISAO HAMILTON, PREDICTOVAO JAY:", H["JAY"])
print("NAPISAO HAMILTON, PREDICTOVAO DISPUTED:", H["DISPUTED"])
print("NAPISAO MADISON, PREDICTOVAO HAMILTON:", M["HAMILTON"])
print("NAPISAO MADISON, PREDICTOVAO MADISON:", M["MADISON"])
print("NAPISAO MADISON, PREDICTOVAO JAY:", M["JAY"])
print("NAPISAO MADISON, PREDICTOVAO DISPUTED:", M["DISPUTED"])
print("NAPISAO JAY, PREDICTOVAO HAMILTON:", J["HAMILTON"])
print("NAPISAO JAY, PREDICTOVAO MADISON:", J["MADISON"])
print("NAPISAO JAY, PREDICTOVAO JAY:", J["JAY"])
print("NAPISAO JAY, PREDICTOVAO DISPUTED:", J["DISPUTED"])
print("NAPISAO DISPUTED, PREDICTOVAO HAMILTON:", D["HAMILTON"])
print("NAPISAO DISPUTED, PREDICTOVAO MADISON:", D["MADISON"])
print("NAPISAO DISPUTED, PREDICTOVAO JAY:", D["JAY"])
print("NAPISAO DISPUTED, PREDICTOVAO DISPUTED:", D["DISPUTED"])








print(round(time.time()-t0, 4), "s", sep='')