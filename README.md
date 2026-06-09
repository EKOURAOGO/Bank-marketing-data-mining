# Data Mining — Bank Marketing UCI

> Segmentation des profils clients et modélisation prédictive de la souscription à un dépôt à terme bancaire

**Auteur :** Emmanuel KOURAOGO 
**Encadrant :** Laurent MAGON

---

## Contexte

Dataset issu des campagnes de marketing direct d'une banque portugaise (appels téléphoniques).
**Objectif :** prédire si un client souscrira à un dépôt à terme (`souscription = yes/no`) avant tout contact.

- **45 211 observations**, 17 variables
- Déséquilibre marqué : **88,3% non** / **11,7% oui**
- Approche **ex ante** : exclusion de `duration` (data leakage)
- Source : [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

---

## Structure du projet

```
bank-marketing-data-mining/
├── SCRIPT_EMMANUEL.R                     # Pipeline complet R (1 400+ lignes)
├── PROJET_DATA_MINNING_EMMANUEL.docx     # Rapport complet
└── README.md
```

---

## Pipeline complet

### 1. Prétraitement & EDA (sur Train uniquement)

Split stratifié **70% / 30%** :

| Jeu | Observations | Proportion |
|-----|-------------|------------|
| Train | 31 647 | 70% |
| Test | 13 564 | 30% |

**Résultats EDA :**
- Effet saisonnier fort : mars, sept, oct, déc > 40–50% de souscription
- `poutcome = success` = variable la plus discriminante
- `students` et `retired` : taux de souscription nettement supérieurs
- Clients sans prêt immobilier ni personnel : plus susceptibles de souscrire
- `duration` exclue (observable seulement après l'appel → data leakage)

### 2. Feature engineering

- Création de variables de **saisonnalité** (regroupement des mois en saisons)
- **Tranches d'âge** pour capter les effets du cycle de vie
- Transformation de `pdays` : distinction clients jamais contactés vs historique existant
- Encodage OHE des catégorielles, normalisation des numériques
- Gestion des modalités "unknown" comme catégorie informative

### 3. Modèles supervisés — Benchmark 3 algorithmes

| Modèle | Framework |
|--------|-----------|
| Régression logistique | `tidymodels` + `glm` |
| Random Forest | `tidymodels` + `ranger` |
| XGBoost | `tidymodels` + `xgboost` |

- SMOTE via `themis` pour rééquilibrage des classes
- Optimisation par Grid Search + validation croisée
- Métriques : AUC-ROC, Recall, courbes ROC comparées
- Export des coefficients logistiques significatifs (Word via `flextable` + `officer`)

### 4. Analyse business — Simulation opérationnelle

- **Courbe de profitabilité commerciale** (XGBoost)
- Optimisation du seuil de décision pour maximiser le profit net
- Tableau de simulation par seuil (0.05 → 0.50) :
  - Nombre de clients à contacter
  - Souscriptions attendues
  - Coût total, gain total, profit net estimé

### 5. Clustering KMeans — Segmentation non supervisée

- Application sur données transformées (recette Train)
- Méthode du coude → **k = 4 clusters** optimal
- Analyse : effectif et taux de souscription par segment

---

## Installation

```r
install.packages(c(
  "tidyverse", "tidymodels", "xgboost", "ranger",
  "themis", "naniar", "foreign", "flextable",
  "officer", "doParallel", "rsample", "yardstick"
))

source("SCRIPT_EMMANUEL.R")
```

---

## Stack technique

![R](https://img.shields.io/badge/R-276DC3?style=flat-square&logo=r&logoColor=white)
![tidymodels](https://img.shields.io/badge/tidymodels-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-red?style=flat-square)
![ggplot2](https://img.shields.io/badge/ggplot2-visualization-green?style=flat-square)

---

## Auteur

**Emmanuel KOURAOGO** — M2 IMSD
[GitHub](https://github.com/EKOURAOGO) · [Email](mailto:ekouraogo73@gmail.com)
