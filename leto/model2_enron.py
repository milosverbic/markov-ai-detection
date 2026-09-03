import time
t0 = time.time()
import pandas as pd
from copy import deepcopy
import random as R
import numpy as np
from scipy.spatial.distance import jensenshannon, cosine
from scipy.stats import entropy
from enron_emails.top_words import reci200


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

def trainFromSample(sample, text_split: list, M):
    for i in sample:
        text = " ".join(text_split[i*SEGMENT_LENGTH:(i+1)*SEGMENT_LENGTH])
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

def testOnSample(sample, text_split: list, lanci, alg="euklidska"):
    R = {x:0 for x in LABELI}
    for i in sample:
            text = " ".join(text_split[i*SEGMENT_LENGTH:(i+1)*SEGMENT_LENGTH])
            author = testOnText(text, lanci, alg=alg)
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

def addDicts(a: dict, b: dict):   # ne koristi se nigde
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


reciClean = [r for r in reci200 if r not in ("enron")]  # cistim listu reci od theme-specificnih reci

NR = 50  # NR<=199
reciNR = [reciClean[i] for i in range(len(reciClean)) if i<NR]
RECI = reciNR

ALG = "logLikelihood"
ALPHA = 0.1  # laplace smoothing, potreban za LLL i KL (inace treba da bude 0)
for SEGMENT_LENGTH in range(500, 1100, 5000):
    print("segment length:", SEGMENT_LENGTH)

    TRAIN_TEST_RATIO = 0.8

    LABELI = ["SALLY beck", "SUSAN scott", "perlingiere", "shackleton", "germany", "nemec", "taylor", "kaminski", "jones", "mann"]

    # SALLY_WORDS = 131270
    # SUSAN_WORDS = 92424

    words = [131270, 92424, 97765, 172199, 127410, 88182, 102111, 75481, 149057, 169624]

    PRINT = False

    word_limit = min(words)
    n_segments = word_limit // SEGMENT_LENGTH
    n_train = round(n_segments*TRAIN_TEST_RATIO)
    n_test = n_segments - n_train
    n_author = [w // SEGMENT_LENGTH for w in words]

    print(n_segments)
    print(n_train)
    print(n_test)

    # SALLY - 131270 reci (ukljucujuci EOF tokene)
    # SUSAN - 92424 reci (ukljucujuci EOF tokene)
    # perlingiere - 97765
    # shackleton - 172199
    # germany - 127410
    # nemec - 88182
    # taylor - 102111
    # kaminski - 75481
    # jones - 149057
    # mann - 169624

    filenames = ["sally", "susan", "debra_perlingiere", "sara_shackleton", "chris_germany", "gerald_nemec", "mark_taylor", "vince_kaminski",
                 "tana_jones", "kay_mann"]

    splits = []
    for fn in filenames:
        with open("leto/enron_emails/"+fn+".txt") as f:
            splits.append(f.read().split())

    ACC = []
    SAL = []
    SUS = []

    for i in range(1):

        samples = [R.sample(range(na), n_segments) for na in n_author]
        train = [s[:n_train] for s in samples]
        test = [s[n_train:] for s in samples]

        lanci = [newChain(ALPHA) for i in range(len(LABELI))]

        for i in range(len(LABELI)):
            trainFromSample(train[i], splits[i], lanci[i])
            normalize(lanci[i])

        results = []
        for i in range(len(LABELI)):
            results.append(testOnSample(test[i], splits[i], lanci, alg=ALG))

        # print("napisala sally, predictovao sally:", sallyR["SALLY"])
        # print("napisala sally, predictovao susan:", sallyR["SUSAN"])
        # print("napisala susan, predictovao sally:", susanR["SALLY"])
        # print("napisala susan, predictovao susan:", susanR["SUSAN"])

        # acc = (sallyR["SALLY"] + susanR["SUSAN"]) / (n_test*2)
        # sally_acc = sallyR["SALLY"] / n_test
        # susan_acc = susanR["SUSAN"] / n_test
            
        # print("\naccuracy:", acc)
        # ACC.append(acc)
        # SAL.append(sally_acc)
        # SUS.append(susan_acc)

        print(results)

    # print(ACC)
    # print(SAL)
    # print(SUS)
    # print("prosek:", sum(ACC)/len(ACC))
    # print("prosek sally:", sum(SAL)/len(SAL))
    # print("prosek susan:", sum(SUS)/len(SUS))






print(round(time.time()-t0, 4), "s", sep='')