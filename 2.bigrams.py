# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Models del llenguatge amb bigrames

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Imports

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# %matplotlib inline

# %%
df = pd.read_csv("data/noms_net.csv", keep_default_na=False)
noms = df.Nom.tolist()
noms[:5]

# %% [markdown]
# Treballarem a nivell de caràcters, farem servir `#` per marcar inici i fi dels noms.
#
# Creem una llista amb els caràcters possibles i diccionaris per passar de caràcter a enter i al revés

# %%
chars = ["#"] + sorted(list(set("".join(noms))))
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for i, c in enumerate(chars)}

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Model amb recompte de bigrames

# %% [markdown]
# Agafant la llista de noms, fem una matriu amb totes les possiblitats de dos caràcters, per ara plena de zeros

# %%
N = torch.zeros((len(c2i), len(c2i)), dtype=torch.int32)
N.shape

# %% [markdown]
# Emplenem la matriu amb el nombre de vegades que apareix cada combinació i ho visualitzem

# %%
for nom in noms:
    chs = ["#"] + list(nom) + ["#"]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = c2i[ch1]
        ix2 = c2i[ch2]
        N[ix1, ix2] += 1

# %%
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap="Blues")
for i in range(len(i2c)):
    for j in range(len(i2c)):
        chstr = i2c[i] + i2c[j]
        plt.text(j, i, chstr, ha="center", va="bottom")
        plt.text(j, i, N[i, j].item(), ha="center", va="top")

# %% [markdown]
# Anem a generar nous noms a partir d'aquestes freqüències

# %%
P = N.float()
# P = (N + 1).float()  # smoothing
P /= P.sum(axis=1, keepdim=True)  # normalitzem

# %%
P.sum(axis=1, keepdim=True).shape  # vector columna

# %%
P.sum(axis=0, keepdim=True).shape  # vector fila

# %%
g = torch.Generator().manual_seed(42)

# %%
# P = torch.ones(32, 32) / 32  # comparació amb model aleatori 

# %%
for i in range(10):
    ix = 0
    nom = ""
    while True:
        ix = torch.multinomial(P[ix], 1, replacement=True, generator=g).item()
        if ix == 0:
            break
        nom += i2c[ix]
    print(nom)

# %% [markdown]
# Són molt dolents!
#
# Es veu que fer servir bigrames no ens dona resultats massa bons.
#
# Ara bé, si ho comparem amb una distribució totalment aleatòria, donant el mateix pes a totes les combinacions la cosa és horrible, així que estem fent algo bé.
# Comprova-ho descommentant la cel·la de dalt amb `torch.ones` i recorrent la generació

# %% [markdown]
# ## Ho podem fer millor?

# %% [markdown]
# Per ara hem considerat tots els noms de la mateixa manera. Podem incorporar la freqüència que apareix a les dades originals per fer que els bigrames més comuns apareguin més.

# %%
freqs = df.freq.to_numpy()
freqs

# %%
N2 = torch.zeros((len(c2i), len(c2i)), dtype=torch.int32)
for i, nom in enumerate(noms):
    chs = ["#"] + list(nom) + ["#"]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = c2i[ch1]
        ix2 = c2i[ch2]
        N2[ix1, ix2] += freqs[i]

# %%
P2 = N2.float()
P2 /= P2.sum(axis=1, keepdim=True)  # normalitzem

# %%
for i in range(10):
    ix = 0
    nom = ""
    while True:
        ix = torch.multinomial(P2[ix], 1, replacement=True, generator=g).item()
        if ix == 0:
            break
        nom += i2c[ix]
    print(nom)

# %% [markdown]
# Potser són una mica millors, però molts segueixen siguent dolents. Considerar només dos caràcters és molt limitant

# %% [markdown]
# ## Trigrames

# %% [markdown]
# El problema de pujar el nombre de caràcters que considerem és que el nombre de possibles combinacions creix ràpidament, i la gran majoria de combinacions no apareixeran mai o gairabé mai.
#
# En aquests casos podem fer servir una representació esparsa, on només guardarem els valor que no siguin zero.

# %%
N3 = torch.zeros((len(c2i), len(c2i), len(c2i)), dtype=torch.int32)

# %% [markdown]
# Fins ara hem fet els bigrames amb un simple zip de python.
#
# Ara passem a fer servir la funció `ngrams` de `nltk` per no complicar-nos la vida

# %%
from nltk import ngrams

# %%
for i, nom in enumerate(noms):
    chs = "#" + nom + "#"
    for ch1, ch2, ch3 in ngrams(chs, 3):
        ix1 = c2i[ch1]
        ix2 = c2i[ch2]
        N3[c2i[ch1], c2i[ch2], c2i[ch3]] += freqs[i]

# %%
P3 = N3.float()

# %%
P3.sum(axis=2, keepdim=True).shape

# %%
P3.sum(axis=2, keepdim=True) + 1

# %%
P3 /= P3.sum(axis=1, keepdim=True)  # normalitzem

# %% [markdown]
# Veiem que dels possibles 32_768 ($32^3$) trigrames només 7_110 apareixen

# %%
# N3 = N3.to_sparse()

# %%
N3

# %%

# %%
log_likelihood = 0.0
n = len(noms)

for w in noms:
    # for w in ["povw"]: # -> smoothing
    chs = ["#"] + list(w) + ["#"]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = c2i[ch1]
        ix2 = c2i[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        # print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f"{log_likelihood=}")
nll = -log_likelihood
print(f"{nll=}")
print(f"{nll/n=}")

# %% [markdown]
# ## Neural Network model

# %%
# Training set of bigrams

X_train, y_train = [], []
for w in noms:
    chs = ["#"] + list(w) + ["#"]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = c2i[ch1]
        ix2 = c2i[ch2]
        X_train.append(ix1)
        y_train.append(ix2)

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)

# %%
xs

# %% [markdown]
# ### One-hot encoding
#
# Convertim un número enter en un vector on tot són zeros excepte una dimensió, que és 1.
#
# Must be converted to float, one_hot does not take dtype input

# %%
noms[0]

# %%
xenc = F.one_hot(X_train[:15], num_classes=27).float()
plt.imshow(xenc)

# %% [markdown]
# Random initialization with gaussian distr

# %%
W = torch.randn((27, 27), requires_grad=True)

# %%
# Forward pass
xenc = F.one_hot(xs, num_classes=27).float()
logits = xenc @ W

# softmax
counts = logits.exp()
probs = counts / counts.sum(1, keepdims=True)

# %%
# Probabilitites we want to maximize (next character)
probs[torch.arange(5), ys]

# %%
loss = -probs[torch.arange(5), ys].log().mean()
loss.item()

# %%
# We can add a regularization to the loss, proportional to
(W**2).mean()

# This is analogous to adding counts (Laplace smoothing)

# %%
# backward pass
W.grad = None  # set gradient to zero
loss.backward()  # populates W.grad

# %%
W = torch.randn((27, 27), requires_grad=True)
# gradient descent
for i in range(10):
    # In this case we can just index
    # xenc = F.one_hot(xs, num_classes=27).float()
    # logits = xenc @ W
    logits = W[xs]
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)

    loss = -probs[torch.arange(5), ys].log().mean() + 0.01 * (W**2).mean()
    W.grad = None  # set gradient to zero
    loss.backward()  # populates W.grad
    print(loss.item())
    W.data += -20 * W.grad

# %%
(W[xs] == xenc @ W).all()
