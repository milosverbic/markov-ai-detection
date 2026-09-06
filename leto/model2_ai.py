import time
t0 = time.time()
import pandas as pd
from copy import deepcopy
import random as R
import numpy as np
from scipy.spatial.distance import jensenshannon, cosine
from scipy.stats import entropy
from pathlib import Path
from top100_words import reci100


def euklidska(m1: dict, m2: dict):
    t = 0
    for i in m1.keys():
        for j in m1[i].keys():
            t += (m1[i][j] - m2[i][j])**2
    return t**0.5

def logLikelihood(text: str, m: dict):  # potreban laplace smoothing
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

def JS(m1: dict, m2: dict):
    l1 = [m1[i][j] for i in m1.keys() for j in m1[i].keys()]
    l2 = [m2[i][j] for i in m2.keys() for j in m2[i].keys()]
    return jensenshannon(l1, l2)

def KL(m1: dict, m2: dict):  # potreban laplace smoothing
    l1 = [m1[i][j] for i in m1.keys() for j in m1[i].keys()]
    l2 = [m2[i][j] for i in m2.keys() for j in m2[i].keys()]
    return entropy(l1, l2)

def cos(m1: dict, m2: dict):
    l1 = [m1[i][j] for i in m1.keys() for j in m1[i].keys()]
    l2 = [m2[i][j] for i in m2.keys() for j in m2[i].keys()]
    return cosine(l1, l2)

def newChain(alpha=0):
    m = {r: {r2: alpha for r2 in RECI} for r in RECI}
    for r in RECI: m[r]["OTHER"] = alpha
    return m


def buildChain(text: str, M): # zapravo dodaje broj tranzicija na onaj koji je vec u matrici (0 za novu matricu); matrica koja nastane nije normalizovana
    p = ""
    nReci = 0
    for Rec in text.split():
        nReci += 1
        rec = ""
        for c in Rec:
            if c not in (",", ".", "?", "!", ";", ":"): rec+=c.lower()
        if rec == "eof":
            p = ""
            continue
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

def trainFromFolder(folder_path, M, sample):
    for i in sample:
        path = folder_path / (str(i+1)+".txt")
        with open(path, 'r', encoding='utf8') as f:
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

def testOnFolder(folder_path, lanci, sample, alg="euklidska", label=""):
    R = {x:0 for x in LABELI}
    for i in sample:
        path = folder_path / (str(i+1)+".txt")
        with open(path, 'r', encoding='utf8') as f:
            text = f.read()
            author = testOnText(text, lanci, label, alg=alg)
            R[author] += 1
    return R

def testOnText(text: str, lanci, label="", alg="euklidska"):
    if alg in ("euklidska", "JS", "KL", "cos"):
        m = newChain(ALPHA)
        nReci = buildChain(text, m)
        normalize(m)
        # mappc(scale, m, 1/nReci)
        if alg=="euklidska": distances = [euklidska(l, m) for l in lanci]
        elif alg=="JS": distances = [JS(l, m) for l in lanci]
        elif alg=="KL": distances = [KL(m, l) for l in lanci]
        elif alg=="cos": distances = [cos(l, m) for l in lanci]
        minD = min(distances)
        predictionIndex = distances.index(minD)
    elif alg=="logLikelihood":
        probabilities = [logLikelihood(text, l) for l in lanci]  
        maxP = max(probabilities)
        predictionIndex = probabilities.index(maxP)

    prediction = LABELI[predictionIndex]

    if alg in ("euklidska", "JS", "KL", "cos") and PRINT:
        print(distances, prediction)
    elif alg=="logLikelihood" and PRINT:
        print(probabilities, prediction)
    # dictToCsv(m, "leto/lanci/dispt/dispt"+id+"_"+prediction+".csv")
    return prediction



def mapp(f, M: dict, *args):   # ne koristi se nigde
    M2 = deepcopy(M)
    for rec in M.keys():
        for rec2 in M.keys():
            M2[rec][rec2] = f(M[rec][rec2], *args)
    return M2

def mappc(f, M: dict, *args):   # ne koristi se nigde
    for rec in M.keys():
        for rec2 in M.keys():
            M[rec][rec2] = f(M[rec][rec2], *args)

def scale(x, s):   # ne koristi se nigde
    return x*s

def addDicts(a: dict, b: dict):
    D = {}
    for k in a.keys():
        D[k] = a[k] + b[k]
    return D

def mergeChains(a: dict, b: dict, w1=0.5, w2=0.5):   # ne koristi se nigde
    D = {}
    for k in a.keys():
        D[k] = {}
        for l in a[k].keys():
            D[k][l] = w1*a[k][l] + w2*b[k][l]
    return D

def normalize(M: dict):  # normalizuje matricu tako da zbir svih redova bude 1
    for k in M.keys():
        s = sum(M[k].values())
        if s>0:
            for k2 in M[k].keys():
                M[k][k2] *= 1/s

def dictToCsv(dict, filename):
    df = pd.DataFrame.from_dict(dict)
    df.to_csv(filename)



### odavde pa na dole menjam kod na osnovu toga sta hocu da testiram


reciClean = [r for r in reci100 if r not in ()]  # cistim listu reci od theme-specificnih reci

NR = 50  # NR<=100
reciNR = [reciClean[i] for i in range(len(reciClean)) if i<NR]
RECI = reciNR

script_dir = Path(__file__).parent 
ESSAY_PATH = script_dir / "ghostbuster-data" / "essay"
REUTER_PATH = script_dir / "ghostbuster-data" / "reuter"
WP_PATH = script_dir / "ghostbuster-data" / "wp"

ALG = "logLikelihood"
ALPHA = 0.1  # laplace smoothing, potreban za LLL i KL (inace treba da bude 0)
TRAIN_TEST_RATIO = 0.8

LABELI = ["human", "gpt"]

PRINT = False


n_texts = 1000  # treniramo na human-gpt parovima i testiramo na human-gpt parovima (ako trainujem na wp/human/58 moram i na wp/gpt/58)
                # takodje imam tri dataseta i na svakom trainujem na istom broju tekstova (800/800 i na wp i na essays i na reuter)
n_train = round(n_texts*TRAIN_TEST_RATIO)
n_test = n_texts - n_train


ACC = []
HUM = []
GPT = []

for i in range(2):

    samples = [R.sample(range(n_texts), n_texts) for i in range(3)]
    train = [s[:n_train] for s in samples]
    test = [s[n_train:] for s in samples]

    HM = newChain(ALPHA)
    GM = newChain(ALPHA)
    lanci = [HM, GM]

    trainFromFolder(ESSAY_PATH/"human", HM, train[0])
    trainFromFolder(REUTER_PATH/"human", HM, train[1])
    trainFromFolder(WP_PATH/"human", HM, train[2])
    normalize(HM)

    trainFromFolder(ESSAY_PATH/"gpt", GM, train[0])
    trainFromFolder(REUTER_PATH/"gpt", GM, train[1])
    trainFromFolder(WP_PATH/"gpt", GM, train[2])
    normalize(GM)

    HR = testOnFolder(ESSAY_PATH/"human", lanci, test[0], alg=ALG)
    HR = addDicts(HR, testOnFolder(REUTER_PATH/"human", lanci, test[1], alg=ALG))
    HR = addDicts(HR, testOnFolder(WP_PATH/"human", lanci, test[2], alg=ALG))

    GR = testOnFolder(ESSAY_PATH/"gpt", lanci, test[0], alg=ALG)
    GR = addDicts(GR, testOnFolder(REUTER_PATH/"gpt", lanci, test[1], alg=ALG))
    GR = addDicts(GR, testOnFolder(WP_PATH/"gpt", lanci, test[2], alg=ALG))

    print(HR)
    print(GR)


    # print("napisala sally, predictovao sally:", sallyR["SALLY"])
    # print("napisala sally, predictovao susan:", sallyR["SUSAN"])
    # print("napisala susan, predictovao sally:", susanR["SALLY"])
    # print("napisala susan, predictovao susan:", susanR["SUSAN"])

    acc = (HR["human"] + GR["gpt"]) / (n_test*3*2)
    human_acc = HR["human"] / (n_test*3)
    gpt_acc = GR["gpt"] / (n_test*3)
        
    # print("\naccuracy:", acc)
    ACC.append(acc)
    HUM.append(human_acc)
    GPT.append(gpt_acc)

    # print(results)

print(ACC)
print(HUM)
print(GPT)
print("prosek:", sum(ACC)/len(ACC))
print("prosek human:", sum(HUM)/len(HUM))
print("prosek gpt:", sum(GPT)/len(GPT))






print(round(time.time()-t0, 4), "s", sep='')