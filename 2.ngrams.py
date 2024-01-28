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
# # Models del llenguatge amb n-grames 

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Imports

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['figure.dpi'] = 100
# %matplotlib inline

# %%
df = pd.read_csv("data/noms_net.csv", keep_default_na=False)
df["nom#"] = "#" + df["Nom"] + "#"
noms = df["nom#"].tolist()
noms[:5]

# %% [markdown]
# Treballarem a nivell de caràcters, farem servir `#` per marcar inici i fi dels noms.
#
# Creem una llista amb els caràcters possibles i diccionaris per passar de caràcter a enter i al revés

# %%
chars = ["#"] + sorted(list(set("".join(df.Nom))))
nchar = len(chars)
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for i, c in enumerate(chars)}
c2i

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Model amb recompte de bigrames

# %% [markdown]
# Podem fer servir zip per crear una llista de bigrames

# %%
noms[0], list(zip(noms[0], noms[0][1:]))

# %% [markdown]
# Agafant la llista de noms, fem una matriu amb totes les possiblitats de dos caràcters, per ara plena de zeros.
#
# Com que volem fer comptar el nombre d'aparicions de cada bigrama farem servir enters (per defecte np.zeros fa servir reals). Podríem fer `dtype=int` o `dtype=np.int64`, que és el mateix, però com que no ens calen nombres enormes, farem servir `dtype=int32`, que ocupa menys espai en memòria.
#
# Podeu veure els nombres mínim i màxim que pot representar un tipus de dades de numpy amb `np.iinfo` per enters o `np.finfo` per floats.

# %%
N = np.zeros((nchar, nchar), dtype=np.int32)
np.iinfo(np.int32)

# %% [markdown]
# Emplenem la matriu amb el nombre de vegades que apareix cada combinació i ho visualitzem

# %%
for nom in noms:
    for ch1, ch2 in zip(nom, nom[1:]):
        ix1 = c2i[ch1]
        ix2 = c2i[ch2]
        N[ix1, ix2] += 1

# %%
plt.figure(figsize=(18, 18))
plt.imshow(N, cmap="Blues")
for i in range(len(i2c)):
    for j in range(len(i2c)):
        chstr = i2c[i] + i2c[j]
        plt.text(j, i, chstr, ha="center", va="bottom")
        plt.text(j, i, N[i, j].item(), ha="center", va="top")

# %% [markdown]
# Anem a generar nous noms a partir d'aquestes freqüències.
#
# Per cada nom que generem, comencem amb '#' i mirem quina probabilitat hi ha per la següent caràcter, fins que ens surti un altre '#' final.
#
# N[0] (fila 0 de la matriu N, com podem veure a la figura superior) ens diu el recompte dels caràcters que segueixen '#'. Per exemple veiem que no va mai seguit d'un altre '#', espai o apòstrof.

# %%
N[0]

# %% [markdown]
# Si ens interessa, podem fer un gràfic de quins caràcters solen seguir quins altres, o predecedir-los (canvieu `N[i]` per `N[:, i]`)

# %%
c = "#"
i = c2i[c]
plt.bar(np.arange(nchar), N[i])
plt.title(f"{i2c[i]} va seguit de...")
for i, value in enumerate(N[i]):
    plt.text(i, value, i2c[i], ha='center', va='bottom')

# %% [markdown]
# El que volem és **normalitzar** aquest recompte per obtenir freqüències, que interpretarem com a probabilitats

# %%
N[0] / N[0].sum()

# %% [markdown]
# Com que haurem de fer aquesta operació per totes les possibles combinacions creem una nova matriu P amb les files normalitzades

# %%
N.sum(axis=1, keepdims=True).shape  # vector columna

# %%
P = N / N.sum(axis=1, keepdims=True)  # normalitzem

# %% [markdown]
# Comprovem que la fila és la mateixa

# %%
P[0]

# %%
# P = np.ones((nchar, nchar)) / nchar  # comparació amb model aleatori

# %%
for i in range(10):
    ix = 0
    nom = ""
    while True:
        ix = np.argmax(np.random.multinomial(1, P[ix]))
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
# Comprova-ho descomentant la cel·la de dalt amb `np.ones` i refent la generació

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Ho podem fer millor?

# %% [markdown]
# Per ara hem considerat tots els noms de la mateixa manera. Podem incorporar la freqüència que apareix a les dades originals per fer que els bigrames més comuns apareguin més.

# %%
freqs = df.freq.to_numpy()
freqs

# %%
N2 = np.zeros((nchar, nchar), dtype=np.int32)
for i, nom in enumerate(noms):
    for ch1, ch2 in zip(nom, nom[1:]):
        ix1 = c2i[ch1]
        ix2 = c2i[ch2]
        N2[ix1, ix2] += freqs[i]

# %%
P2 = N2 / N2.sum(axis=1, keepdims=True)

# %%
for i in range(10):
    ix = 0
    nom = ""
    while True:
        ix = np.argmax(np.random.multinomial(1, P[ix]))
        if ix == 0:
            break
        nom += i2c[ix]
    print(nom)

# %% [markdown]
# Potser són una mica millors, però molts segueixen siguent dolents. Considerar només dos caràcters és molt limitant

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Trigrames

# %% [markdown]
# Fins ara hem fet els bigrames amb un simple zip de python.
#
# Ara passem a fer servir la funció `ngrams` de `nltk` per no complicar-nos la vida

# %%
from nltk import ngrams

# %%
N3 = np.zeros((nchar, nchar, nchar), dtype=np.int32)
for i, nom in enumerate(noms):
    for ch1, ch2, ch3 in ngrams(nom, 3):
        N3[c2i[ch1], c2i[ch2], c2i[ch3]] += freqs[i]

# %% [markdown]
# Ara cal fixar-nos en els 2 caràcters anteriors per generar el següent. També farem ús del model de bigrames per generar el primer caràcter de tots.
#
# Hem de normalitzar el recompte de caràcters que segueixen qualsevols dos caràcters, per exemple "ma"

# %%
c2i["m"],  c2i["a"]

# %%
N3[15,3]

# %%
plt.bar(np.arange(nchar), N3[15, 3])
for i, value in enumerate(N3[15, 3]):
    plt.text(i, value, i2c[i], ha='center', va='bottom')

# %%
N3[15,3] / N3[15,3].sum()

# %% [markdown]
# Com haviem fet pels bigrames, fem totes les operacions de cop

# %%
N3.sum(axis=2, keepdims=True).shape

# %%
P3 = N3 / N3.sum(axis=2, keepdims=True)
P3[0,0]

# %% [markdown]
# Veiem que ens surt un error. És degut a una divisió entre 0, que dona `nan` com a resultat.
#
# Podríem no fer-ne cas, ja que precisament no farem servir les combinacions que mai apareixen. El que farem, però és fer servir `np.divide`, que ens permetrà obtenir zeros en comptes de `nan`.

# %%
N3_sum = N3.sum(axis=2, keepdims=True)
P3 = np.divide(N3, N3_sum, out=np.zeros(N3.shape, dtype=float), where=N3_sum!=0)
P3[0,0]

# %% [markdown]
# I comprovem que el resultat és el mateix que a dalt

# %%
P3[15, 3]

# %% [markdown]
# Fins ara hem generat resultats aleatoris no reproduïbles, si volem fer un experiment o obtenir dades per un article, és bona idea fixar una llavor (seed), que ens permeti reproduir els resultats.
#
# Segons d'on provingui l'aleatorietat caldrà veure com es fixa la llavor. Per numpy es fa amb `np.random.seed`

# %%
np.random.seed(42)

# %%
for i in range(10):
    # Primer caràcter a partir de bigrames
    ix1 = 0
    ix2 = np.argmax(np.random.multinomial(1, P2[ix1]))
    nom = i2c[ix2]
    
    # Resta amb trigrames
    while True:
        ix3 = np.argmax(np.random.multinomial(1, P3[ix1, ix2]))
        if ix3 == 0:
            break
        nom += i2c[ix3]
        ix1 = ix2
        ix2 = ix3
        
    print(nom)

# %% [markdown]
# ## Exercici 1: adapta el codi de generació de noms amb trigrames perquè generi noms d'home o de dona
#
# Crea una funció `genera_noms` que rebi un sol argument `sexe` i retorni un únic nom
#
# Deixarem les matrius de probabilitat com a variables globals.
#
# Et caldran dues matrius per homes (PH2, PH3) i dues per dones (PD2, PD3).
#

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Solució

# %%
noms_home = df[df.Sexe == "H"]["nom#"].to_list()
noms_dona = df[df.Sexe == "D"]["nom#"].to_list()


# %%
def get_P2(noms, c2i):
    nchar = len(c2i)
    N2 = np.zeros((nchar, nchar), dtype=np.int32)
    for i, nom in enumerate(noms):
        for ch1, ch2 in zip(nom, nom[1:]):
            N2[c2i[ch1], c2i[ch2]] += freqs[i]

    N2_sum = N2.sum(axis=1, keepdims=True)
    P2 = np.divide(N2, N2_sum, out=np.zeros(N2.shape, dtype=float), where=N2_sum!=0)
    return P2


# %%
def get_P3(noms, c2i):
    nchar = len(c2i)
    N3 = np.zeros((nchar, nchar, nchar), dtype=np.int32)
    for i, nom in enumerate(noms):
        for ch1, ch2, ch3 in ngrams(nom, 3):
            N3[c2i[ch1], c2i[ch2], c2i[ch3]] += freqs[i]

    N3_sum = N3.sum(axis=2, keepdims=True)
    P3 = np.divide(N3, N3_sum, out=np.zeros(N3.shape, dtype=float), where=N3_sum!=0)
    return P3


# %%
PH2 = get_P2(noms_home, c2i)
PD2 = get_P2(noms_dona, c2i)
PH3 = get_P3(noms_home, c2i)
PD3 = get_P3(noms_dona, c2i)


# %%
def genera_noms(sexe):
    if sexe == "H":
        P2 = PH2
        P3 = PH3
    elif sexe == "D":
        P2 = PD2
        P3 = PD3
    else:
        raise ValueError("sexe ha de ser H o D")

    ix1 = 0
    ix2 = np.argmax(np.random.multinomial(1, P2[ix1]))
    nom = i2c[ix2]

    while True:
        ix3 = np.argmax(np.random.multinomial(1, P3[ix1, ix2]))
        if ix3 == 0:
            break
        nom += i2c[ix3]
        ix1 = ix2
        ix2 = ix3

    return nom


# %%
for _ in range(10):
    print(genera_noms("H"))

# %%
for _ in range(10):
    print(genera_noms("D"))


# %% [markdown]
# ## Exercici 2: amplia la funció `genera_noms` perquè faci servir 4-grames

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Solució

# %%
def get_P4(noms, c2i):
    nchar = len(c2i)
    N4 = np.zeros((nchar,)*4, dtype=np.int32)
    for i, nom in enumerate(noms):
        for ch1, ch2, ch3, ch4 in ngrams(nom, 4):
            N4[c2i[ch1], c2i[ch2], c2i[ch3], c2i[ch4]] += freqs[i]

    N4_sum = N4.sum(axis=3, keepdims=True)
    P4 = np.divide(N4, N4_sum, out=np.zeros(N4.shape, dtype=float), where=N4_sum!=0)
    return P4


# %%
PH4 = get_P4(noms_home, c2i)
PD4 = get_P4(noms_dona, c2i)

# %% [markdown]
# Tot i ser només un cas, existeix el nom _E_, d'un sol caràcter. I la probabilitat que `ix3` sigui 0 no és nula. Per aixo tenim el primer `return nom`

# %%
P3[c2i["#"], c2i["e"], c2i["#"]]


# %%
def genera_noms(sexe):
    if sexe == "H":
        P2 = PH2
        P3 = PH3
        P4 = PH4
    elif sexe == "D":
        P2 = PD2
        P3 = PD3
        P4 = PD4
    else:
        raise ValueError("sexe ha de ser H o D")

    ix1 = 0
    ix2 = np.argmax(np.random.multinomial(1, P2[ix1]))
    nom = i2c[ix2]

    ix3 = np.argmax(np.random.multinomial(1, P3[ix1, ix2]))
    if ix3 == 0:
        return nom
    else:
        nom += i2c[ix3]

    while True:
        ix4 = np.argmax(np.random.multinomial(1, P4[ix1, ix2, ix3]))
        if ix4 == 0:
            return nom
        nom += i2c[ix4]
        ix1 = ix2
        ix2 = ix3
        ix3 = ix4


# %%
for _ in range(10):
    print(genera_noms("H"))

# %%
for _ in range(10):
    print(genera_noms("D"))

# %% [markdown]
# ## *Exercici 3: amplia la funció `genera_noms` perquè faci servir n-grames
#
# Com que generar les matrius de probabilitat pot ser un procés lent, busca com es fa servir el paquet `tqdm` per mostrar una barra amb el progrés.
#
# Fes el codi en general, i prova'l amb 5-grames.
#
# 1. Fes primer una funció `get_Pn` que retorni les matrius de probabilitat corresponents a n-grames, partint de `get_P4`.
# 2. Genera i guarda les matrius de P2 a P5 corresponents a homes i dones en un vector per cada sexe (PnsH, PnsD).
# 3. Crea la funció `genera_noms(Pns, n)`, que rebi com a input Pns i el nombre d'ngrames que farà servir per la generació. 
#  

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Pistes

# %% [markdown]
# 1. Per `genera_noms`, pots fer servir `deque` per guardar els indexs ix1, ix2... en un array que automàticament mantingui només `n-1` índexs.

# %%
from collections import deque

ixs = deque([0], maxlen=3)
ixs.append(1)
print(ixs)
ixs.append(2)
print(ixs)
ixs.append(15)
print(ixs)
ixs.append(3)
print(ixs)
ixs.append(14)
print(ixs)

# %% [markdown]
# 2. Pots indexar un array de numpy amb una tuple

# %%
PH4[tuple(ixs)]

# %% [markdown]
# 3. Fixa't en què es repeteix a `genera_noms` de l'exercici 2. Agrupa-ho dins un `while True` amb un únic `return nom`

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Solució

# %%
from tqdm.auto import tqdm


# %%
def get_Pn(noms, freqs, c2i, n):
    nchar = len(c2i)
    N = np.zeros((nchar,) * n, dtype=np.int32)
    for i, nom in enumerate(noms):
        freq = freqs[i]
        for chs in ngrams(nom, n):
            ixs = tuple(c2i[ch] for ch in chs)
            N[ixs] += freq

    N_sum = N.sum(axis=n-1, keepdims=True)
    P = np.divide(N, N_sum, out=np.zeros(N.shape, dtype=float), where=N_sum!=0)
    return P


# %%
def generar_Pns(noms, freqs, c2i, n):
    Pns = []
    for i in tqdm(range(2, n+1)):
        Pns.append(get_Pn(noms, freqs, c2i, i)) 
    return Pns


# %%
PnsH = generar_Pns(noms_home, freqs, c2i, 5)
PnsD = generar_Pns(noms_dona, freqs, c2i, 5)

# %%
from collections import deque

def genera_noms(Pns, n):
    assert n <= len(Pns) + 1
    ixs = deque([0], maxlen=n-1)
    nom = ""

    i = 0
    while True:
        ix = np.argmax(np.random.multinomial(1, Pns[i][tuple(ixs)]))
        if ix == 0:
            return nom
        else:
            ixs.append(ix)
            nom += i2c[ix]

        if i + 2 < n:
            i += 1


# %%
for _ in range(10):
    print(genera_noms(PnsH, n=5))

# %%
for _ in range(10):
    print(genera_noms(PnsD, n=5))
