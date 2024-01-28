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
# # Repàs pandas

# %% [markdown]
# ## Lectura, tipus de dades i duplicats

# %%
import matplotlib.pyplot as plt
import pandas as pd

# %matplotlib inline

# %% [markdown]
# Treballarem amb una llista de noms extrets de https://www.idescat.cat/noms/

# %%
df = pd.read_csv("data/noms.csv", sep=";", skiprows=7)
df

# %% [markdown]
# Renombrem les columnes

# %%
df = df.rename(columns={"Rànquing. Freqüència": "freq", "Rànquing. ‰": "permil"})

# %% [markdown]
# El primer que cal fer és mirar les dades a mà, per saber què contenen.

# %% [markdown]
# En quin format està llegint les dades pandas? (`object` ve a ser `string`)
#
# Veiem que el rànquing de freqüència és un nombre real en comptes d'un enter,  i el rànquing en tant per mil no el reconeix com a numèric.

# %%
df.info()

# %%
df["freq"] *= 1000
df = df.astype({"freq": int})
df["permil"] = pd.to_numeric(df["permil"].str.replace(",", "."))
df.head()

# %%
df.info()

# %%
df.describe()

# %% [markdown]
# Si ja sabem que totes les columnes tenen el mateix format, podem directament dir-li a pandas com es marquen els decimals i els milers

# %%
pd.read_csv("data/noms.csv", sep=";", skiprows=7, decimal=",", thousands=".").head()

# %% [markdown]
# Hi ha algun duplicat a les dades?

# %%
df[df.Nom.duplicated()]

# %% [markdown]
# Sembla que molts, però...

# %%
df[df[["Nom", "Sexe"]].duplicated()]

# %% [markdown]
# Era perquè eren tant per `Sexe` H com D. Realment només hi ha un duplicat

# %%
df.query("Nom == 'BEGOÑA'")

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Ens falta netejar la columna `Nom`

# %% [markdown]
# Veiem que hi ha un nom que es llegeix com a NaN (Not a Number). Si anem a veure les dades el nom és "NA", que pandas interpreta com a NaN.
# Ho podem solucionar passant l'argument `keep_default_na=False` a `pd.read_csv`

# %%
df[df.Nom.isna()]

# %%
df = pd.read_csv(
    "data/noms.csv", sep=";", skiprows=7, decimal=",", thousands=".", keep_default_na=False
)
df[df.Nom.isna()]

# %%
df.iloc[6334]

# %% [markdown]
# Molts noms tenen dues o més opcions d'escriptura, amb accent o sense; amb accent obert o tancat, etc.
#
# En podem veure la distribució i casos en particular:

# %%
plt.hist(df["Nom"].str.split("/").str.len())
plt.yscale("log")

# %%
df[df["Nom"].str.contains("/").fillna(False)]


# %% [markdown]
# Per ara optarem per quedar-nos sempre amb una sola opció. A més ho passem tot a minúscules i treiem els accents.

# %%
def treure_accents(s):
    s = (
        s.replace("à", "a")
        .replace("á", "a")
        .replace("è", "e")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ï", "i")
        .replace("ò", "o")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ü", "u")
    )
    return s


# %%
df["Nom"] = df["Nom"].str.split("/").str[0].str.lower().apply(treure_accents)
df.Nom

# %% [markdown]
# ## Posant-ho tot junt:

# %%
df = pd.read_csv(
    "data/noms.csv", sep=";", skiprows=7, decimal=",", thousands=".", keep_default_na=False
)
df = df.rename(columns={"Rànquing. Freqüència": "freq", "Rànquing. ‰": "permil"})
df["Nom"] = df["Nom"].str.split("/").str[0].str.lower().apply(treure_accents)
df

# %% [markdown]
# Podem guardar aquestes dades en un altre fitxer.
#
# `index=False` evita guardar la primera columna (índex)

# %%
df.to_csv("data/noms_net.csv", index=False)

# %% [markdown]
# I el podem rellegir i comprovar que tenim les mateixes dades

# %%
df2 = pd.read_csv("data/noms_net.csv", keep_default_na=False)
(df == df2).all()
