# DSLR – Data Science Linear Regression (42)

## Objectif du projet

Ce projet consiste à explorer un dataset d’élèves de Poudlard afin de :

1. Comprendre les données
2. Visualiser les relations entre variables
3. Sélectionner les features pertinentes
4. Préparer une régression logistique pour prédire la maison

---

# Structure du projet

```
.
├── data_analysis/
├── data_visualization/
├── files/
├── datasets/
└── Makefile
```

---

# PARTIE 1 — Data Analysis (`describe.py`)

## Objectif

Reproduire un équivalent de `pandas.describe()` **à la main**

## Statistiques

- Count
- Mean
- Std
- Min
- 25%
- 50%
- 75%
- Max

---

# PARTIE 2 — Data Visualization

## Histogram
→ distribution des données par maison

## Scatter plot
→ relation entre 2 variables

## Pair plot
→ vue globale du dataset

## Files
tous les histograms, scatter_plot et le pair_plot sont deja generes et dispo dans /files !!! les make permettent de les afficher un a un

---

# Commandes

```
make describe
make histogram -> affiche l'histogramme de la feature (modifier les features dans data_visualization/histogram.py)
make scatter -> affiche une comparaison de deux features (modifier les features dans data_visualization/scatter.py)
make pair -> genere le scatter plot matrix (point de vue global du dataset)
```

---

# Concepts clés

- Feature
- Distribution
- Corrélation
- Data cleaning

---

# Suite

→ Régression logistique