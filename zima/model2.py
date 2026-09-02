# ovo je isti model koji trenutno koristim (as of 1.9.2026.) samo sto koristi euklidsku distancu da poredi matrice

import time
t0 = time.time()
import pandas as pd
from copy import deepcopy
import random as R
from matplotlib import pyplot as plt
import numpy as np

def euklidska(m1: dict, m2: dict):
    t = 0
    for i in m1.keys():
        for j in m1[i].keys():
            t += (m1[i][j] - m2[i][j])**2
    return t**0.5


def train(sample: set | range, data: pd.DataFrame, M, reci, brojReci):
    for i in sample:
        p = ""
        s = data.iloc[i]
        if type(s)==str:
            for Rec in s.split(" "):
                brojReci += 1
                rec = ""
                for c in Rec:
                    if c not in (",", ".", "?", "!"): rec+=c.lower()
                if rec not in reci:
                    if p!="":
                        M[p]["OTHER"] += 1
                    p = ""
                    continue
                if p == "":
                    p = rec
                    continue
                M[p][rec] += 1
                p = rec
    # mappc(scale, M, 1/brojReci)  # ovo ponekad treba u komentar # prototip: 1/TN (pojavljivanja po dokumentu), ovde pojavljivanja po broju reci (ukupnom)
    return brojReci
    

def test(sample: set | range, data: pd.DataFrame, reci, lanci: list, label: str):
    R = {x:0 for x in M}
    R['-'] = 0
    for i in sample:
        brojReci = 0
        m = {r: {r2: 0 for r2 in reci} for r in reci}
        for r in reci: m[r]["OTHER"] = 0
        p = ""
        s = data.iloc[i]
        if type(s)!=str:
            # print(type(s))
            # input()
            R["-"] += 1
            continue
        # if s[:39] == "Error: Error communicating with OpenAI:":
        #     R["-"] += 1
        #     continue
        for Rec in s.split(" "):
                brojReci += 1
                rec = ""
                for c in Rec:
                    if c not in (",", ".", "?", "!"): rec+=c.lower()
                if rec not in reci:
                    if p!="":
                        m[p]["OTHER"] += 1
                    p = ""
                    continue
                if p == "":
                    p = rec
                    continue
                m[p][rec] += 1
                p = rec
        # if brojReci<200:
        #     R['-'] += 1
        #     continue
        # mappc(scale, m, 1/brojReci) #1/1 za prototip, ovde 1/brojReci
        normalize(m)
        distance = [euklidska(l, m) for l in lanci]
        if label == 'LJ':
            distance_LJ_LJ.append(distance[0])
            distance_LJ_AI.append(distance[1])
            razlike_LJ.append(distance[0]-distance[1])
            # if distance[0]-distance[1] > 0.007:
            #     print(s)
            #     input()
        elif label == 'AI':
            distance_AI_LJ.append(distance[0])
            distance_AI_AI.append(distance[1])
            razlike_AI.append(distance[0]-distance[1])
            # if distance[0]-distance[1] < -0.00115:
            #     print(s)
            #     input()
        minD = min(distance)
        modelIndex = distance.index(minD)
        model = M[modelIndex]
        R[model] += 1
    return R

def mapp(f, M: dict, *args):
    M2 = deepcopy(M)
    for rec in M.keys():
        for rec2 in M.keys():
            M2[rec][rec2] = f(M[rec][rec2], *args)
    return M2

def mappc(f, M: dict, *args):
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
 




data = pd.read_csv("train.csv")

ljudskiV2 = pd.read_csv("train_v2_drcat_02_ljudski.csv")["text"] # 27371 tekstova
aiV2 = pd.read_csv("train_v2_drcat_02_ai.csv")["text"] # 17497 tekstova

ljudski = data["Human_story"]

N = 0
N_LJ = 27371
N_AI = 17497

TN = 0  # broj tekstova koji ide na training
TN_LJ = 0
TN_AI = 0

ljM1 = {r: {r2: 0 for r2 in reci50} for r in reci50}
for r in reci50: ljM1[r]["OTHER"] = 0
aiM1 = {r: {r2: 0 for r2 in reci50} for r in reci50}
for r in reci50: aiM1[r]["OTHER"] = 0

S = 7
M = ['LJ', 'AI']

# testSamples = [set(), set()]
# trainSamples = [set(), set()]
# testSamples[0] = range(0) #set(R.sample(range(N_LJ), N_LJ-TN_LJ))
# testSamples[1] = range(0) #set(R.sample(range(N_AI), N_AI-TN_AI))
# testSamples = [set(R.sample(range(N), N-TN)) for i in range(S)]
# trainSamples = [set(range(N)) - testSamples[i] for i in range(S)]
# trainSamples[0] = range(TN_LJ) #set(range(N_LJ)) - testSamples[0]
# trainSamples[1] = range(TN_AI) #set(range(N_AI)) - testSamples[1]


# TSDF = pd.DataFrame(testSample)
# TSDF.to_csv("testSampleRandomM1.csv")
# testSample = pd.read_csv("testSampleRandomM1.csv")['0'].tolist()

# testSamples = [range(TN, N) for i in range(S)]
# trainSamples = [range(TN) for i in range(S)]

# brojReciLJ = train(trainSamples[0], ljudski, ljM1, reci50, 0)
# # mappc(scale, ljM1, 1/brojReciLJ)
# normalize(ljM1)
# brojReciAI = train(trainSamples[1], data["gemma-2-9b"], aiM1, reci50, 0)
# brojReciAI = train(trainSamples[2], data["mistral-7B"], aiM1, reci50, brojReciAI)
# brojReciAI = train(trainSamples[3], data["qwen-2-72B"], aiM1, reci50, brojReciAI)
# brojReciAI = train(trainSamples[4], data["llama-8B"], aiM1, reci50, brojReciAI)
# brojReciAI = train(trainSamples[5], data["accounts/yi-01-ai/models/yi-large"], aiM1, reci50, brojReciAI)
# brojReciAI = train(trainSamples[6], data["GPT_4-o"], aiM1, reci50, brojReciAI)
# # mappc(scale, aiM1, 1/brojReciAI)
# normalize(aiM1)

ljM2 = {r: {r2: 0 for r2 in reci50} for r in reci50}
aiM2 = {r: {r2: 0 for r2 in reci50} for r in reci50}

testSamples2 = [set(), set()]
trainSamples2 = [set(), set()]
testSamples2[0] = set(R.sample(range(N_LJ), N_LJ-TN_LJ))
testSamples2[1] = set(R.sample(range(N_AI), N_AI-TN_AI))
# trainSamples2[0] = set(range(N_LJ)) - testSamples[0]
# trainSamples2[1] = set(range(N_AI)) - testSamples[1]
# trainSamples2[0] = range(TN_LJ)
# trainSamples2[1] = range(TN_AI)

# brojReciLJV2 = train(trainSamples2[0], ljudskiV2, ljM2, reci50, 0)
# mappc(scale, ljM2, 1/brojReciLJV2)
# brojReciAIV2 = train(trainSamples2[1], aiV2, aiM2, reci50, 0)
# mappc(scale, aiM2, 1/brojReciAIV2)


# ljM = mergeChains(ljM1, ljM2)
# aiM = mergeChains(aiM1, aiM2)


# train(trainSamples[1], gpt, gM, reci50)
 
ljM = pd.read_csv("ljudskiFullM2.csv", index_col=0).to_dict()
aiM = pd.read_csv("aiFullM2.csv", index_col=0).to_dict()

# ljDF = pd.DataFrame(mapp(scale, mapp(round, ljM, 8), 1000000), reci50)
# gDF = pd.DataFrame(mapp(scale, mapp(round, gM, 8), 1000000), reci50)
# ljDF = pd.DataFrame.from_dict(ljM1)
# aiDF = pd.DataFrame.from_dict(aiM1)
# ljDF.to_csv("ljudskiFullM2.csv")
# aiDF.to_csv("aiFullM2.csv")

# exit()

pred = {}


distance_LJ_LJ = []
distance_LJ_AI = []
distance_AI_LJ = []
distance_AI_AI = []

razlike_LJ = []
razlike_AI = []

# pred['LJ1'] = test(testSamples[0], ljudski, reci50, [ljM, aiM], 'LJ')

pred['LJ'] = test(testSamples2[0], ljudskiV2, reci50, [ljM, aiM], 'LJ')

# pred['AI1'] = test(testSamples[1], data["gemma-2-9b"], reci50, [ljM, aiM], 'AI')
# pred['AI1'] = addDicts(pred['AI1'], test(testSamples[2], data["mistral-7B"], reci50, [ljM, aiM], 'AI'))
# pred['AI1'] = addDicts(pred['AI1'], test(testSamples[3], data["qwen-2-72B"], reci50, [ljM, aiM], 'AI'))
# pred['AI1'] = addDicts(pred['AI1'], test(testSamples[4], data["llama-8B"], reci50, [ljM, aiM], 'AI'))
# pred['AI1'] = addDicts(pred['AI1'], test(testSamples[5], data["accounts/yi-01-ai/models/yi-large"], reci50, [ljM, aiM], 'AI'))
# pred['AI1'] = addDicts(pred['AI1'], test(testSamples[6], data["GPT_4-o"], reci50, [ljM, aiM], 'AI'))

pred['AI'] = test(testSamples2[1], aiV2, reci50, [ljM, aiM], 'AI')

# pred['G'] = test(testSamples[1], gpt, reci50, [ljM, gM], 'G')

# pred['LJ'] = test(testSamples[0], ljudski, reci50, [ljM, aiM], 'LJ')
# pred['AI'] = test(testSamples[1], ai, reci50, [ljM, aiM], 'AI')

print(pred)

# n1 = pred['AI1']['AI'] + pred['LJ1']['AI'] + pred['LJ1']['LJ'] + pred['AI1']['LJ']
# n2 = pred['AI2']['AI'] + pred['LJ2']['AI'] + pred['LJ2']['LJ'] + pred['AI2']['LJ']

tp = pred['AI']['AI'] #/n1 + pred['AI2']['AI']/n2
fp = pred['LJ']['AI'] #/n1 + pred['LJ2']['AI']/n2
tn = pred['LJ']['LJ'] #/n1 + pred['LJ2']['LJ']/n2
fn = pred['AI']['LJ'] #/n1 + pred['AI2']['LJ']/n2

acc = (tp + tn)/(tp + tn + fp + fn)
tpr = tp/(tp+fn)
fpr = fp/(fp+tn)
rocauc = (tpr-fpr+1)/2

print("Accuracy:", acc)
print("TPR:", tpr)
print("FPR:", fpr)
print("ROC-AUC:", rocauc)

# combined_data = np.concatenate([distance_AI_LJ, distance_AI_AI])
# bin_edges = np.linspace(min(combined_data), max(combined_data)/1.5, 100)

# plt.hist(distance_AI_LJ, bins=bin_edges, color="blue", alpha=0.5)
# plt.hist(distance_AI_AI, bins=bin_edges, color="red", alpha=0.5)

combined_data = np.concatenate([razlike_LJ, razlike_AI])
bin_edges = np.linspace(min(combined_data), max(combined_data), 100)

plt.hist(razlike_LJ, bins=bin_edges, color="blue", alpha=0.5, density=True)
plt.hist(razlike_AI, bins=bin_edges, color="red", alpha=0.5, density=True)
plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
plt.show()

print(round(time.time()-t0, 4), "s", sep='')