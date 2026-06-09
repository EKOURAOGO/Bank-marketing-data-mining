#rm(list = ls())
# Affiche la version de R utilisée
R.version.string
# Met à jour tous les packages installés afin d'assurer la compatibilité avec la version actuelle de R
update.packages(ask = FALSE, checkBuilt = TRUE)
install.packages("installr")
# Chargement du package
library(installr)
# Tentative de mise à jour automatique de R (peut retourner FALSE si R est déjà à jour)
updateR()


###############################################################################
# 0. Installation des packages nécessaires au projet
###############################################################################
# Installation des packages principaux pour :
# - manipulation des données (tidyverse)
# - machine learning (tidymodels)
# - modèles de type gradient boosting (xgboost)
# - évaluation des performances (yardstick, rsample)
# - importation de données (readr)
install.packages(c(
  "tidyverse",
  "tidymodels",
  "xgboost",
  "yardstick",
  "rsample",
  "readr",
  "broom",
  "doParallel",
  "themis",
  "foreign",
  "ranger",
  "tibble"
))
install.packages("naniar")
install.packages(c("flextable", "officer"))

#  Chargement des bibliothèques
library(tidyverse)
library(tidymodels)
library(xgboost)
library(foreign)
library(dplyr)
library(ggplot2)
library(broom)
library(doParallel)
library(themis) 
library(yardstick)
library(ranger)
library(tibble)
library(naniar)
library(flextable)
library(officer)
###############################################################################
### Prédiction de la souscription à un dépôt à terme bancaire
###############################################################################

# Définir le répertoire de travail
setwd("C:/Users/Pc/OneDrive/Bureau/Dossier/IMSD/DATA MINNING/MINNING/PROJET_R")
getwd()
list.files()

# lire le fichier ARFF
df <- read.arff("phpkIxskf.arff")
df <- as_tibble(df)
class(df)

# Renommage des colonnes
colnames(df) <- c(
  "age",
  "job",
  "marital",
  "education",
  "default",
  "balance",
  "housing",
  "loan",
  "contact",
  "day",
  "month",
  "duration",
  "campaign",
  "pdays",
  "previous",
  "poutcome",
  "y"
)

head(df)
summary(df)

# Visualisation  de la structure des manquants
vis_miss(df)
# Vérifier la présence de "unknown" dans chaque colonne
sapply(df, function(x) sum(x == "unknown"))

# Recodage, passage en facteur et renommage de la colonne 'y'
df <- df %>%
  mutate(y = factor(recode(y, "1" = "no", "2" = "yes"), levels = c("no", "yes"))) %>%
  rename(souscription = y)

## Affichage
table(df$souscription)
prop.table(table(df$souscription))*100


##### Split TRAIN / TEST
set.seed(42)

split <- initial_split(
  df,
  prop = 0.7,
  strata = souscription
)

train_data <- training(split)
test_data  <- testing(split)

prop.table(table(train_data$souscription))
prop.table(table(test_data$souscription))

# Nombre total d'observations
n_total <- nrow(df)

# Nombre d'observations dans chaque jeu
n_train <- nrow(train_data)
n_test  <- nrow(test_data)

# Résumé
tibble(
  Jeu = c("Train", "Test", "Total"),
  Observations = c(n_train, n_test, n_total),
  Proportion = round(c(n_train, n_test, n_total) / n_total * 100, 1)
)


###############################################################################
## Analyse Exploratoire des Données (EDA)
###############################################################################

# « Toutes les décisions exploratoires sont prises exclusivement à partir du jeu d’apprentissage afin d’éviter toute fuite d’information vers le test. »


#### Types de variables (jeu d’apprentissage uniquement)
# Variables catégorielles
cat_vars <- names(train_data %>% select(where(is.factor)))
# Variables numériques
num_vars <- names(train_data %>% select(where(is.numeric)))
# Affichage
cat_vars
num_vars

#Toutes les variables servent à répondre à une seule question :
# Ce client a-t-il un profil qui le rend plus ou moins susceptible de souscrire si on l’appelle ?
# Le modèle combine ces signaux pour produire une probabilité.

### 1. Distribution de la variable cible y

# Calcul des proportions
dist_target <- train_data %>%
  count(souscription) %>%
  mutate(prop = n / sum(n))

# Création du graphique
ggplot(dist_target, aes(x = souscription, y = n, fill = souscription)) +
  geom_col(width = 0.6) +
  geom_text(
    aes(label = paste0(round(prop * 100, 1), " %")),
    vjust = -0.5,
    size = 5,
    fontface = "bold"
  ) +
  # Application des couleurs spécifiques : Rouge pour 'no', Vert pour 'yes'
  scale_fill_manual(values = c("no" = "#E41A1C", "yes" = "#4DAF4A")) +
  labs(
    title = "Distribution de la variable cible (Souscription)",
    x = "Souscription au dépôt à terme",
    y = "Nombre de clients"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(face = "bold"),
    legend.position = "none" # On cache la légende car les labels x suffisent
  )

#La figure représente la distribution de la variable cible correspondant à la souscription
# à un dépôt à terme bancaire. On observe un fort déséquilibre entre les deux classes. 
# La majorité des clients n’a pas souscrit au produit, tandis qu’une minorité seulement 
# a accepté l’offre.

# Cette répartition déséquilibrée est typique des campagnes de marketing bancaire, 
# où la souscription constitue un événement relativement rare. 
# Ce déséquilibre devra être pris en compte lors de la phase de modélisation, 
# notamment dans le choix des métriques d’évaluation et, éventuellement, des techniques de rééquilibrage.
# Cela signifie que l’accuracy seule serait trompeuse et qu’il faut privilégier des métriques adaptées comme l’AUC ou le recall.



#### 2. Analyse des variables catégorielles vs y
# Liste des variables catégorielles
cat_vars

# Tableau croisé : profession vs souscription (TRAIN uniquement)
tab_job_y <- train_data %>%
  count(job, souscription) %>%
  group_by(job) %>%
  mutate(pourcentage = n / sum(n) * 100) %>%
  ungroup() %>%
  select(-n) %>%
  tidyr::pivot_wider(
    names_from  = souscription,
    values_from = pourcentage
  )

tab_job_y

# L’analyse croisée entre la profession et la souscription met en évidence des différences marquées de comportement selon les catégories socioprofessionnelles. Les étudiants et les retraités présentent des taux de souscription largement supérieurs à la moyenne, tandis que les professions actives à revenu contraint, telles que les ouvriers ou les employés de services, affichent des probabilités de souscription nettement plus faibles. Ces résultats confirment le caractère fortement discriminant de la variable « profession » dans la modélisation de la souscription.

# Visualisations (barres empilées) - TRAIN uniquement
plot_cat_vs_y <- function(data, var) {
  
  data %>%
    count(.data[[var]], souscription) %>%
    group_by(.data[[var]]) %>%
    mutate(proportion = n / sum(n)) %>%
    ungroup() %>%
    ggplot(aes(
      x = .data[[var]],
      y = proportion,
      fill = souscription
    )) +
    geom_col(width = 0.7) +
    scale_fill_viridis_d(name = "Souscription") +
    labs(
      title = paste("Souscription par", var),
      x = var,
      y = "Proportion"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
}


plot_cat_vs_y(train_data, "job")
plot_cat_vs_y(train_data, "marital")
plot_cat_vs_y(train_data, "education")
plot_cat_vs_y(train_data, "default")
plot_cat_vs_y(train_data, "housing")
plot_cat_vs_y(train_data, "loan")
plot_cat_vs_y(train_data, "contact")
plot_cat_vs_y(train_data, "poutcome")
plot_cat_vs_y(train_data, "month")



# Souscription par mois (month)
# L’analyse par mois met en évidence un effet saisonnier très marqué dans la souscription au dépôt à terme.
# Les mois de mars, septembre, octobre et décembre présentent des proportions de souscription nettement plus élevées que la moyenne, avec des taux pouvant approcher ou dépasser 40–50 % dans certains cas.
# À l’inverse, les mois d’été (juin, juillet, août) ainsi que janvier et février affichent des taux de souscription plus faibles.
# Ces résultats suggèrent que la période de contact joue un rôle clé dans l’efficacité des campagnes marketing, possiblement en lien avec des comportements d’épargne saisonniers ou des périodes propices à la prise de décision financière.


# Souscription par résultat de la campagne précédente (poutcome)
# La variable poutcome apparaît comme l’une des plus discriminantes.
# Les clients pour lesquels la campagne précédente s’est soldée par un succès présentent un taux de souscription très élevé, largement supérieur à celui des autres modalités (plus de 60 %).
# À l’inverse, les clients associés à un échec ou à un résultat inconnu affichent des taux de souscription très faibles.
# Ce résultat confirme que l’historique des interactions passées constitue un déterminant majeur de la décision de souscription et justifie pleinement l’intégration de cette variable dans la modélisation.


# Souscription par type de contact (contact)
# Les clients contactés via un téléphone mobile (cellular) présentent une proportion de souscription légèrement supérieure à ceux contactés via un téléphone fixe.
# En revanche, la modalité unknown est associée à un taux de souscription très faible.
# Cela suggère que le canal de contact influence l’efficacité de la campagne, possiblement en raison d’une plus grande réactivité ou disponibilité des clients contactés par téléphone mobile.


# Souscription et prêt personnel (loan)
# Les clients sans prêt personnel présentent une probabilité de souscription sensiblement plus élevée que ceux disposant d’un prêt en cours.
# Ce résultat est cohérent avec l’intuition économique : un niveau d’endettement plus faible peut faciliter la décision d’épargne.
# La variable loan apparaît ainsi comme un indicateur pertinent de contrainte financière.


# Souscription et prêt immobilier (housing)
# De manière similaire, les clients ne disposant pas d’un prêt immobilier souscrivent davantage que ceux ayant un crédit logement.
# La présence d’un prêt immobilier semble réduire la capacité ou la volonté d’épargne à court terme, ce qui se traduit par une proportion de souscription plus faible.
# Cette variable reflète donc également le niveau d’engagement financier du client.


# Souscription et défaut de crédit (default)
# Les clients sans défaut de crédit présentent une probabilité de souscription plus élevée que ceux ayant connu un défaut.
# Bien que l’écart reste modéré, ce résultat est cohérent avec l’intuition économique : les clients financièrement plus stables sont plus enclins à souscrire à un produit d’épargne.


# Souscription par niveau d’éducation (education)
# Le taux de souscription augmente avec le niveau d’éducation.
# Les clients ayant un niveau tertiaire présentent une proportion de souscription plus élevée que ceux ayant un niveau primaire ou secondaire.
# Ce résultat suggère que le niveau de formation peut être associé à une meilleure compréhension des produits financiers ou à une capacité d’épargne plus importante.


# Souscription par statut marital (marital)
# Les clients célibataires affichent une proportion de souscription légèrement plus élevée que les clients mariés ou divorcés.
# Toutefois, les écarts observés restent relativement modérés, ce qui indique que le statut marital joue un rôle secondaire par rapport à d’autres variables plus discriminantes.


# Souscription par profession (job)
# L’analyse par profession confirme une forte hétérogénéité des comportements de souscription.
# Les étudiants et les retraités se distinguent par des taux de souscription nettement supérieurs à la moyenne, tandis que les ouvriers, employés de services et entrepreneurs présentent des taux plus faibles.
# Ces résultats soulignent l’importance du cycle de vie professionnel et du niveau de stabilité financière dans la décision de souscription.


#L’analyse des variables catégorielles met en évidence plusieurs facteurs fortement associés à la souscription à un dépôt à terme bancaire, en particulier l’historique des campagnes précédentes, la période de contact, le niveau d’endettement et la situation professionnelle. Ces résultats confirment la pertinence de ces variables pour la phase de modélisation et suggèrent l’existence de comportements différenciés selon le profil des clients.


#### Analyse des variables numériques selon la souscription
# Variables numériques à analyser
num_vars

# Boxplots par variable numérique (comparaison no vs yes)
for (var in num_vars) {
  
  p <- ggplot(train_data, aes(x = souscription, y = .data[[var]])) +
    geom_boxplot(fill = "steelblue", alpha = 0.7, outlier.alpha = 0.3) +
    labs(
      title = paste(var, "selon la souscription"),
      x = "Souscription",
      y = var
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold")
    )
  
  print(p)
}

# Densités (KDE)
vars_kde <- c("age", "balance", "duration")

for (var in vars_kde) {
  
  p <- ggplot(train_data, aes(x = .data[[var]], fill = souscription)) +
    geom_density(alpha = 0.5, adjust = 1) +
    labs(
      title = paste("Distribution de", var, "selon la souscription"),
      x = var,
      y = "Densité",
      fill = "Souscription"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold")
    )
  
  print(p)
}


# Durée de l’appel (duration)
# La distribution de la durée des appels montre une différence très marquée entre les souscripteurs et les non-souscripteurs.
# Les clients ayant souscrit présentent des durées d’appel nettement plus longues, avec une médiane et une dispersion significativement supérieures à celles des non-souscripteurs.
# Cette variable apparaît comme la plus discriminante parmi les variables numériques. Toutefois, la durée de l’appel n’est connue qu’après le contact et dépend fortement de l’intérêt manifesté par le client. Son utilisation dans un cadre prédictif ex ante peut donc introduire une fuite d’information et conduire à une surestimation des performances du modèle. Elle a néanmoins été conservée dans un premier temps afin d’évaluer la performance maximale atteignable.


# Solde bancaire (balance)
# Les souscripteurs présentent en moyenne un solde bancaire plus élevé que les non-souscripteurs, comme en témoigne une médiane légèrement supérieure.
# Les distributions sont fortement asymétriques à droite, avec la présence de valeurs extrêmes importantes dans les deux groupes.
# Malgré ce recouvrement élevé entre les distributions, le solde bancaire constitue un indicateur pertinent de la capacité d’épargne, susceptible d’apporter une information complémentaire dans la modélisation.


# Âge (age)
# La distribution de l’âge montre que les souscripteurs sont en moyenne légèrement plus âgés que les non-souscripteurs.
# La médiane de l’âge est plus élevée pour les clients ayant souscrit, bien que les distributions se recouvrent largement.
# Cela suggère que l’âge joue un rôle modéré dans la décision de souscription, en lien avec le cycle de vie, sans constituer à lui seul un facteur discriminant majeur.


# Nombre de contacts précédents (previous)
# Les boxplots indiquent que la majorité des clients n’a été contactée aucune ou très peu de fois lors de campagnes précédentes.
# Les souscripteurs présentent toutefois une médiane légèrement plus élevée que les non-souscripteurs, suggérant qu’un historique de contacts antérieurs peut augmenter marginalement la probabilité de souscription.
# Néanmoins, la forte concentration des valeurs autour de zéro limite le pouvoir discriminant de cette variable prise isolément.


# Délai depuis le dernier contact (pdays)
# La variable pdays met en évidence une différence notable entre les deux groupes.
# Les souscripteurs présentent plus fréquemment des valeurs positives de pdays, indiquant qu’ils avaient déjà été contactés lors de campagnes précédentes, tandis que la valeur zéro ou proche de zéro est dominante chez les non-souscripteurs.
# Cette variable capte donc un effet mémoire des campagnes passées, cohérent avec les résultats observés pour poutcome.


# Nombre de contacts durant la campagne (campaign)
# Les clients ayant souscrit ont généralement été contactés un nombre limité de fois, avec une médiane plus faible que celle des non-souscripteurs.
# À l’inverse, un nombre élevé de contacts est principalement observé chez les non-souscripteurs, suggérant un effet de saturation ou de lassitude.
# Ce résultat indique qu’une intensification excessive des contacts peut être contre-productive.


# Jour du mois (day)
# Aucune différence significative n’apparaît entre les souscripteurs et les non-souscripteurs concernant le jour du mois du contact.
# Les distributions sont très similaires, ce qui suggère que cette variable possède un pouvoir explicatif limité.
# Elle pourra donc être écartée ou considérée comme secondaire lors de la phase de modélisation.


# L’analyse des variables numériques met en évidence plusieurs facteurs associés à la souscription, notamment la durée de l’appel, le solde bancaire et l’historique des contacts. À l’inverse, certaines variables telles que le jour du mois présentent un pouvoir explicatif limité. Ces résultats guideront les choix effectués lors de la phase de modélisation, tant en termes de sélection de variables que de prévention des fuites d’information.



#### Matrice de corrélation – variables numériques
num_vars_corr <- train_data %>%
  select(where(is.numeric)) %>%
  select(-campaign) %>%   # optionnel selon ton choix
  names()
corr_mat <- cor(
  train_data %>% select(all_of(num_vars_corr)),
  use = "complete.obs"
)
corr_df <- as.data.frame(as.table(corr_mat))
colnames(corr_df) <- c("Var1", "Var2", "Correlation")
ggplot(corr_df, aes(x = Var1, y = Var2, fill = Correlation)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", Correlation)), size = 3) +
  scale_fill_gradient2(
    low = "blue",
    mid = "white",
    high = "red",
    midpoint = 0,
    name = "Corrélation"
  ) +
  labs(
    title = "Matrice de corrélation des variables numériques",
    x = "",
    y = ""
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

 
# La matrice de corrélation des variables numériques met en évidence une faible corrélation linéaire entre la majorité des variables, limitant ainsi le risque de multicolinéarité. Seul le couple pdays–previous présente une corrélation modérée, cohérente avec leur rôle commun dans la description de l’historique des contacts. Ces résultats suggèrent que les variables numériques apportent des informations complémentaires et peuvent être conservées dans la phase de modélisation.
# L’absence de corrélations élevées entre les variables numériques limite le risque 
# de redondance et contribue à une meilleure capacité de généralisation des modèles.


#### Analyse ciblée de duration
# Bien que la variable duration, correspondant à la durée de l’appel téléphonique, présente un fort pouvoir discriminant, elle n’est observable qu’à l’issue du contact et dépend directement de l’intérêt manifesté par le client au cours de l’échange. Son utilisation dans un modèle prédictif introduirait ainsi une fuite d’information et conduirait à une surestimation des performances.
# Dans une optique de prédiction ex ante, visant à estimer la probabilité de souscription avant le contact avec le client, la variable duration a donc été exclue de la phase de modélisation. Ce choix méthodologique garantit une évaluation plus réaliste et une meilleure capacité de généralisation des modèles.


###############################################################################
##  Préparation des données
###############################################################################

##### 1. Variable cible pour la modélisation

# Pour la phase de modélisation, la variable cible utilisée est souscription, 
# déjà encodée sous forme binaire :
#   
# 0 / no : non-souscription
# 1 / yes : souscription

# Variable cible
table(train_data$souscription)

##### Gestion du déséquilibre des classes (choix méthodologique)

## Ce déséquilibre est pris en compte lors de la phase de modélisation,
# notamment par l’introduction d’une pondération des observations et
# l’utilisation de métriques d’évaluation adaptées.

class_prop <- train_data %>%
  count(souscription) %>%
  mutate(prop = n / sum(n))

class_prop

# Création d'un poids inversement proportionnel
train_data <- train_data %>%
  mutate(
    poids_classe = if_else(
      souscription == "yes",
      1 / mean(souscription == "yes"),
      1 / mean(souscription == "no")
    )
  )


#### creation de nouvelle variable

rec <- recipe(souscription ~ ., data = train_data) %>%
  
  # Pondération des observations
  update_role(poids_classe, new_role = "case_weight") %>%
  
  # Feature engineering
  step_mutate(
    is_first_contact = if_else(pdays == -1, 1, 0),
    pdays_clean      = if_else(pdays == -1, NA_real_, as.numeric(pdays)),
    
    is_spring = if_else(month %in% c("mar", "apr", "may"), 1, 0),
    is_summer = if_else(month %in% c("jun", "jul", "aug"), 1, 0),
    is_autumn = if_else(month %in% c("sep", "oct", "nov"), 1, 0),
    is_winter = if_else(month %in% c("dec", "jan", "feb"), 1, 0),
    
    tranche_age = case_when(
      age < 30              ~ "Jeunes",
      age >= 30 & age < 45  ~ "Adultes",
      age >= 45 & age < 60  ~ "Seniors",
      age >= 60             ~ "Retraités"
    )
  ) %>%
  
  # Suppression des variables non observables / redondantes
  step_rm(duration, month, pdays, age) %>%
  
  # 🔑 CONVERSION character → factor
  step_string2factor(all_nominal_predictors()) %>%
  
  # Imputation
  step_impute_median(all_numeric_predictors()) %>%
  
  # Encodage one-hot
  step_dummy(all_nominal_predictors()) %>%
  
  # Normalisation
  step_normalize(all_numeric_predictors())


# Les étapes de feature engineering, d’imputation, d’encodage et de normalisation ont été intégrées dans une recette tidymodels afin de garantir une préparation cohérente des données. Les variables non observables ex ante ou redondantes ont été supprimées, tandis que des variables dérivées ont été construites pour capter l’historique de contact, la saisonnalité et le cycle de vie des individus. Cette approche permet d’éviter toute fuite d’information et d’assurer la reproductibilité des résultats.


###############################################################################
## MODÉLISATION : LOGISTIQUE, RANDOM FOREST, XGBOOST
###############################################################################

# ====================================================
# Modèle 1 : Régression logistique pondérée (baseline)
# ====================================================
# La régression logistique constitue le modèle de référence de ce projet.
# Elle permet d’estimer la probabilité qu’un client souscrive à un dépôt à terme
# bancaire à partir de ses caractéristiques socio-démographiques, financières
# et de son historique de contact.
#
# Ce modèle est particulièrement adapté au problème étudié, la variable cible
# étant binaire (souscription / non-souscription), et offre l’avantage d’une
# interprétabilité élevée des coefficients estimés.
#
# L’objectif de ce premier modèle est double :
# - fournir une baseline simple et robuste pour l’évaluation des performances,
# - servir de point de comparaison pour des modèles plus complexes développés
#   par la suite.
#
# La modélisation est réalisée dans une optique de prédiction ex ante, en excluant
# les variables non observables au moment de la décision (notamment la durée de
# l’appel), afin de garantir une évaluation réaliste et opérationnelle.

# Spécification du modèle
# Régression logistique comme modèle de référence (baseline)
log_spec <- logistic_reg(
  mode = "classification",
  engine = "glm"
)

# Construction du workflow
# Le workflow combine la recipe (préparation + pondération)
# et le modèle de régression logistique
wf_log <- workflow() %>%
  add_recipe(rec) %>%
  add_model(log_spec)

# Validation croisée (TRAIN uniquement)
set.seed(42)

folds <- vfold_cv(
  train_data,
  v = 5,
  strata = souscription
)

# Métriques adaptées au déséquilibre
metrics <- metric_set(
  roc_auc,
  recall,
  precision
)

cv_results_log <- fit_resamples(
  wf_log,
  resamples = folds,
  metrics   = metrics,
  control   = control_resamples(save_pred = TRUE)
)

collect_metrics(cv_results_log)
collect_predictions(cv_results_log)
show_notes(cv_results_log)


# Entraîner le modèle final sur tout le TRAIN
final_log_model <- fit(
  wf_log,
  data = train_data
)

#poids_classe dans le TEST
test_data <- test_data %>%
  mutate(poids_classe = 1)

# Prédire sur le TEST
test_pred <- predict(
  final_log_model,
  new_data = test_data,
  type = "prob"
) %>%
  bind_cols(
    predict(final_log_model, new_data = test_data, type = "class")
  ) %>%
  bind_cols(
    test_data %>% select(souscription)
  )

#Calculer les métriques sur le TEST
metric_set(
  roc_auc,
  recall,
  precision
)(
  test_pred,
  truth = souscription,
  estimate = .pred_class,
  .pred_yes
)


#Matrice de confusion (TEST)
conf_mat(
  test_pred,
  truth    = souscription,
  estimate = .pred_class
)


#ajuster le seuil
test_pred <- test_pred %>%
  mutate(
    pred_02 = factor(
      if_else(.pred_yes >= 0.2, "yes", "no"),
      levels = c("no", "yes")
    )
  )

conf_mat(
  test_pred,
  truth = souscription,
  estimate = pred_02
)

 
## Interprétation du modèle – Coefficients de la régression logistique

final_log_model <- fit(
  wf_log,
  data = train_data
)
# Extraire le modèle glm depuis le workflow
glm_model <- extract_fit_engine(final_log_model)

# Extraire les coefficients
coef_df <- broom::tidy(glm_model)

# Aperçu des coefficients
head(coef_df)

# Calcul des odds ratios
coef_df <- coef_df %>%
  mutate(
    odds_ratio = exp(estimate)
  ) %>%
  arrange(desc(abs(estimate)))

coef_df

coef_signif <- coef_df %>%
  filter(p.value < 0.05)

coef_signif

# L’analyse des coefficients de la régression logistique met en évidence plusieurs déterminants majeurs de la souscription à un dépôt à terme bancaire. Le facteur le plus discriminant est l’historique des campagnes précédentes. En effet, la variable poutcome_success présente un coefficient positif élevé (β = 0,39) et un odds ratio de 1,48, indiquant qu’un client ayant déjà souscrit lors d’une campagne antérieure voit ses chances de souscription augmenter d’environ 48 % toutes choses égales par ailleurs. Ce résultat souligne l’existence d’un fort effet de fidélisation et de confiance, faisant de l’historique client un levier central de ciblage.
# 
# Le niveau d’endettement constitue un autre déterminant clé de la décision de souscription. Les clients disposant d’un prêt immobilier présentent une probabilité de souscription significativement plus faible (β = −0,32 ; OR = 0,72), ce qui correspond à une diminution d’environ 28 % des chances de souscrire. De manière similaire, la présence d’un prêt personnel est associée à une baisse de la probabilité de souscription (β = −0,16 ; OR = 0,85). Ces résultats sont cohérents avec l’intuition économique : un endettement plus élevé réduit la capacité d’épargne disponible. À l’inverse, le solde bancaire joue un rôle positif, bien que plus modéré : une augmentation du solde est associée à une hausse des chances de souscription (β = 0,06 ; OR = 1,06), traduisant une meilleure situation financière.
# 
# Le cycle de vie des clients apparaît également comme un facteur explicatif important. Les retraités se distinguent par une probabilité de souscription significativement plus élevée (β = 0,19 ; OR = 1,21), soit une augmentation d’environ 21 % des chances de souscrire par rapport à la catégorie de référence. Les clients plus jeunes présentent également un effet positif, bien que plus modéré (β = 0,14 ; OR = 1,15). Ces résultats confirment que certaines tranches d’âge, notamment les retraités, sont particulièrement réceptives aux produits d’épargne sécurisés.
# 
# La stratégie de contact influence fortement l’efficacité de la campagne. Le nombre de contacts effectués durant la campagne a un effet négatif marqué sur la probabilité de souscription (β = −0,31 ; OR = 0,73), indiquant qu’un contact supplémentaire réduit les chances de souscrire d’environ 27 %. Ce résultat met en évidence un effet de lassitude, suggérant qu’une intensification excessive des appels est contre-productive. Par ailleurs, les clients dont le canal de contact est inconnu présentent une probabilité de souscription nettement plus faible (β = −0,46 ; OR = 0,63), ce qui souligne l’importance de la qualité des informations de contact pour maximiser l’efficacité commerciale.
# 
# Une saisonnalité significative est également observée. Les périodes du printemps (β = 0,17 ; OR = 1,18) et de l’automne (β = 0,11 ; OR = 1,11) sont associées à des probabilités de souscription plus élevées, confirmant les résultats obtenus lors de l’analyse exploratoire. Ces effets suggèrent que le calendrier des campagnes joue un rôle important dans la décision d’épargne des clients.
# 
# Enfin, certaines caractéristiques socio-démographiques et professionnelles sont associées à des probabilités de souscription plus faibles. Les clients appartenant aux catégories blue-collar (β = −0,08 ; OR = 0,92), services (β = −0,06 ; OR = 0,94) ou entrepreneur (β = −0,06 ; OR = 0,94) présentent des chances de souscription inférieures à la moyenne, reflétant des contraintes de revenus ou une instabilité financière plus marquée.
# 
# Dans l’ensemble, les résultats du modèle sont économiquement cohérents et confirment la pertinence de la régression logistique dans une optique de prédiction ex ante. Le modèle met clairement en évidence des leviers opérationnels pour l’institution bancaire, notamment le ciblage prioritaire des clients peu endettés, la valorisation de l’historique des campagnes réussies, la limitation du nombre de contacts et le choix de périodes de campagne plus favorables.



###### 4. Évaluation des performances

# S'assurer que le poids existe dans le test (poids neutre)
test_data <- test_data %>%
  mutate(poids_classe = 1)

# Prédictions sur le test
test_pred <- predict(
  final_log_model,
  new_data = test_data,
  type = "prob"
) %>%
  bind_cols(
    predict(final_log_model, test_data, type = "class"),
    test_data %>% select(souscription)
  )

head(test_pred)


#MÉTRIQUES CLASSIQUES
# Accuracy (indépendante de la classe positive)
accuracy(test_pred, truth = souscription, estimate = .pred_class)

# Precision (pour "yes")
precision(
  test_pred,
  truth = souscription,
  estimate = .pred_class,
  event_level = "second"
)

# Recall (pour "yes" – la plus importante ici)
recall(
  test_pred,
  truth = souscription,
  estimate = .pred_class,
  event_level = "second"
)

# F1-score
f_meas(
  test_pred,
  truth = souscription,
  estimate = .pred_class,
  event_level = "second"
)

#MATRICE DE CONFUSION
conf_mat(
  test_pred,
  truth = souscription,
  estimate = .pred_class,
  event_level = "second"
) %>%
  autoplot(type = "heatmap") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(
    title = "Matrice de confusion – Régression logistique",
    x = "Classe prédite",
    y = "Classe réelle"
  )


#COURBE ROC & AUC
# Courbe ROC
roc_auc(
  test_pred,
  truth = souscription,
  .pred_yes,
  event_level = "second"   # "yes" est l’événement
)


# AUC
roc_curve(
  test_pred,
  truth = souscription,
  .pred_yes,
  event_level = "second"
) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_abline(linetype = "dashed", color = "gray") +
  labs(
    title = "Courbe ROC – Régression logistique",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal()


# ==========================
## MODÈLE 2 : RANDOM FOREST 
# ==========================
set.seed(42)

# Spécification du modèle Random Forest
rf_spec <- rand_forest(
  mode  = "classification",
  trees = 500,
  mtry  = tune(),
  min_n = tune()
) %>%
  set_engine(
    "ranger",
    importance = "impurity",
    probability = TRUE
  )

# Workflow
wf_rf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_spec)

# Validation croisée
folds <- vfold_cv(
  train_data,
  v = 5,
  strata = souscription
)

rf_grid <- grid_regular(
  mtry(range = c(3, 10)),
  min_n(range = c(10, 100)),
  levels = 2
)

rf_metrics <- metric_set(
  roc_auc,
  recall,
  precision
)

rf_res <- tune_grid(
  wf_rf,
  resamples = folds,
  grid      = rf_grid,
  metrics   = rf_metrics,
  control   = control_grid(save_pred = TRUE)
)

# Sélection et entraînement final
best_rf <- select_best(rf_res, metric = "recall")

final_rf <- finalize_workflow(
  wf_rf,
  best_rf
)

final_rf_fit <- fit(final_rf, data = train_data)

# Évaluation sur le TEST
test_pred_rf <- predict(final_rf_fit, test_data, type = "prob") %>%
  bind_cols(
    predict(final_rf_fit, test_data, type = "class"),
    test_data %>% select(souscription)
  )

# Métriques
roc_auc(test_pred_rf, truth = souscription, .pred_yes, event_level = "second")
recall(test_pred_rf, truth = souscription, estimate = .pred_class, event_level = "second")
precision(test_pred_rf, truth = souscription, estimate = .pred_class, event_level = "second")

# Matrice de confusion
conf_mat(
  test_pred_rf,
  truth = souscription,
  estimate = .pred_class
) %>%
  autoplot(type = "heatmap") +
  labs(
    title = "Matrice de confusion – Random Forest",
    x = "Classe prédite",
    y = "Classe réelle"
  )

# Courbe ROC
roc_rf <- roc_curve(
  test_pred_rf,
  truth = souscription,
  .pred_yes,
  event_level = "second"
)

ggplot(roc_rf, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "steelblue", linewidth = 1.2) +
  geom_abline(linetype = "dashed", color = "grey50") +
  labs(
    title = "Courbe ROC – Random Forest",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal()


# Vérification de l'écart Train/Validation pour Random Forest
rf_res %>% 
  collect_metrics() %>% 
  filter(.metric == "roc_auc") %>%
  select(mean, std_err)


# Utilisons le seuil de 0.25 pour être plus "offensif"
seuil_offensif <- 0.2

test_pred_rf_custom <- test_pred_rf %>%
  mutate(
    pred_class_custom = factor(
      if_else(.pred_yes >= seuil_offensif, "yes", "no"),
      levels = c("no", "yes")
    )
  )

# Comparaison des matrices de confusion
# Avant (Seuil 0.5)
conf_mat(test_pred_rf, truth = souscription, estimate = .pred_class)

# Après (Seuil 0.2)
conf_mat(test_pred_rf_custom, truth = souscription, estimate = pred_class_custom)


# ====================
## MODÈLE 3 : XGBOOST 
# ====================

# Recipe spécifique XGBoost (sans poids)
rec_xgb <- rec %>%
  update_role(poids_classe, new_role = "predictor") %>%
  step_rm(poids_classe)

# Ratio de déséquilibre (TRAIN uniquement)
ratio_xgb <- sum(train_data$souscription == "no") /
  sum(train_data$souscription == "yes")

# Spécification du modèle XGBoost
xgb_spec <- boost_tree(
  trees = 500,
  tree_depth = tune(),
  learn_rate = tune(),
  mtry = tune(),
  loss_reduction = tune(),
  sample_size = tune()
) %>%
  set_engine(
    "xgboost",
    scale_pos_weight = ratio_xgb,
    eval_metric = "auc"
  ) %>%
  set_mode("classification")

# Workflow
wf_xgb <- workflow() %>%
  add_recipe(rec_xgb) %>%
  add_model(xgb_spec)

# Grille d’hyperparamètres
set.seed(42)

xgb_grid <- grid_latin_hypercube(
  tree_depth(range = c(3, 8)),
  learn_rate(range = c(0.01, 0.2)),
  finalize(mtry(), train_data),
  loss_reduction(),
  sample_size = sample_prop(),
  size = 20
)

# Validation croisée parallélisée
cl <- makePSOCKcluster(parallel::detectCores() - 1)
registerDoParallel(cl)

xgb_res <- tune_grid(
  wf_xgb,
  resamples = folds,
  grid = xgb_grid,
  metrics = metric_set(roc_auc, recall),
  control = control_grid(save_pred = TRUE)
)

stopCluster(cl)

# Modèle final
best_xgb <- select_best(xgb_res, metric = "roc_auc")

final_xgb <- finalize_workflow(
  wf_xgb,
  best_xgb
)

final_xgb_fit <- fit(final_xgb, data = train_data)

# Évaluation sur le TEST
xgb_test_pred <- predict(final_xgb_fit, test_data, type = "prob") %>%
  bind_cols(test_data %>% select(souscription))

# AUC
roc_auc(
  xgb_test_pred,
  truth = souscription,
  .pred_yes,
  event_level = "second"
)

# Ajustement du seuil (objectif métier)
seuil <- 0.2

xgb_eval <- xgb_test_pred %>%
  mutate(
    pred_class = factor(
      if_else(.pred_yes >= seuil, "yes", "no"),
      levels = c("no", "yes")
    )
  )

# Matrice de confusion
conf_mat(
  xgb_eval,
  truth = souscription,
  estimate = pred_class
)

# Recall & Precision
recall(
  xgb_eval,
  truth = souscription,
  estimate = pred_class,
  event_level = "second"
)

precision(
  xgb_eval,
  truth = souscription,
  estimate = pred_class,
  event_level = "second"
)


# Extraction correcte du moteur
xgb_engine <- extract_fit_engine(final_xgb_fit)

# Préparation des données (X)
# Nous avons utilisé 'bake' pour que les données soient au format numérique 
# attendu par XGBoost (transformation des variables catégorielles en dummies)
data_for_shap <- bake(
  prep(rec_xgb), 
  new_data = train_data %>% slice_sample(n = 500), 
  all_predictors(),
  composition = "matrix" 
)

# Fonction de prédiction ajustée
p_fun <- function(object, newdata) {
  predict(object, newdata = newdata)
}

# Calcul SHAP 
set.seed(42)
shap_values <- fastshap::explain(
  xgb_engine,          
  X = data_for_shap, 
  pred_wrapper = p_fun, 
  nsim = 50, 
  adjust = TRUE
)

# Conversion en dataframe 
shap_df <- as.data.frame(shap_values)

###############################################################################
## VISUALISATION SHAP – IMPORTANCE GLOBALE (XGBOOST)
###############################################################################

# Importance globale = moyenne des valeurs absolues SHAP
shap_importance <- shap_df %>%
  summarise(across(everything(), ~ mean(abs(.), na.rm = TRUE))) %>%
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "importance"
  ) %>%
  arrange(desc(importance))

# Top 15 variables
shap_importance_top <- shap_importance %>%
  slice_head(n = 15)

# Plot
ggplot(shap_importance_top,
       aes(x = reorder(variable, importance), y = importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Importance globale des variables (SHAP – XGBoost)",
    subtitle = "Moyenne des contributions absolues aux probabilités",
    x = "Variables",
    y = "Importance SHAP moyenne"
  ) +
  theme_minimal(base_size = 13)


## ANALYSE DE PROFITABILITÉ (VISION BUSINESS – XGBOOST)

# Hypothèses économiques
cost_call   <- 2     # € par appel
gain_client <- 60    # € par souscription

profit_data <- xgb_test_pred %>%
  arrange(desc(.pred_yes)) %>%
  mutate(
    rang = row_number(),
    part_base = rang / n(),
    cost = rang * cost_call,
    revenue = cumsum(if_else(souscription == "yes", gain_client, 0)),
    profit = revenue - cost
  )

# Point de profit maximal
idx_max_profit <- which.max(profit_data$profit)
best_pct       <- profit_data$part_base[idx_max_profit]
best_threshold <- profit_data$.pred_yes[idx_max_profit]

# Courbe de profit
ggplot(profit_data, aes(x = part_base, y = profit)) +
  geom_line(color = "#2c3e50", linewidth = 1.4) +
  geom_vline(xintercept = best_pct, linetype = "dashed", color = "red") +
  annotate(
    "text",
    x = best_pct,
    y = max(profit_data$profit) * 0.85,
    label = paste0(
      "Profit max à ", round(best_pct * 100, 1), "% de la base\n",
      "Seuil optimal = ", round(best_threshold, 3)
    ),
    hjust = -0.05
  ) +
  scale_x_continuous(labels = scales::percent) +
  labs(
    title = "Courbe de profitabilité commerciale – XGBoost",
    x = "% de clients contactés (triés par score)",
    y = "Profit net estimé (€)"
  ) +
  theme_minimal(base_size = 13)



###############################################################################
## TABLEAU DE SYNTHÈSE FINAL – COMPARAISON DES MODÈLES
###############################################################################

# AUC
auc_log <- roc_auc(test_pred, truth = souscription, .pred_yes, event_level = "second")$.estimate
auc_rf  <- roc_auc(test_pred_rf, truth = souscription, .pred_yes, event_level = "second")$.estimate
auc_xgb <- roc_auc(xgb_test_pred, truth = souscription, .pred_yes, event_level = "second")$.estimate

# Recall (classe positive = yes)
recall_log <- recall(test_pred, truth = souscription, estimate = .pred_class, event_level = "second")$.estimate
recall_rf  <- recall(test_pred_rf, truth = souscription, estimate = .pred_class, event_level = "second")$.estimate
recall_xgb <- recall(xgb_eval, truth = souscription, estimate = pred_class, event_level = "second")$.estimate

# Tableau final
model_comparison <- tibble(
  Modèle = c("Régression logistique", "Random Forest", "XGBoost"),
  AUC    = round(c(auc_log, auc_rf, auc_xgb), 3),
  Recall = round(c(recall_log, recall_rf, recall_xgb), 3)
)

print(model_comparison)


## Nombre de personne à appeller
# Seuils business à tester
seuils <- c(0.50,0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15,  0.10, 0.05)

business_table <- map_dfr(seuils, function(s) {
  
  tmp <- xgb_test_pred %>%
    mutate(
      pred_class = if_else(.pred_yes >= s, 1, 0)
    )
  
  nb_appels <- sum(tmp$pred_class)
  nb_souscriptions <- sum(tmp$souscription == "yes" & tmp$pred_class == 1)
  
  tibble(
    Seuil_probabilité = s,
    Clients_contactés = nb_appels,
    Taux_contactés = round(nb_appels / nrow(tmp) * 100, 1),
    Souscriptions_attendues = nb_souscriptions,
    Coût_total = nb_appels * cost_call,
    Gain_total = nb_souscriptions * gain_client,
    Profit_net = (nb_souscriptions * gain_client) - (nb_appels * cost_call)
  )
})

View(business_table)


## ramdom
###############################################################################
## NOMBRE DE PERSONNES À APPELER – RANDOM FOREST
###############################################################################

# Seuils business à tester
seuils <- c(0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05)

business_table_rf <- purrr::map_dfr(seuils, function(s) {
  
  tmp <- test_pred_rf %>%
    mutate(
      pred_class = if_else(.pred_yes >= s, 1, 0)
    )
  
  nb_appels <- sum(tmp$pred_class)
  nb_souscriptions <- sum(tmp$souscription == "yes" & tmp$pred_class == 1)
  
  tibble(
    Seuil_probabilité      = s,
    Clients_contactés     = nb_appels,
    Taux_contactés        = round(nb_appels / nrow(tmp) * 100, 1),
    Souscriptions_attendues = nb_souscriptions,
    Coût_total        = nb_appels * cost_call,
    Gain_total         = nb_souscriptions * gain_client,
    Profit_net         = (nb_souscriptions * gain_client) -
      (nb_appels * cost_call)
  )
})

View(business_table_rf)


############export
library(dplyr)

coef_word <- coef_signif %>%
  select(
    Variable = term,
    Coefficient = estimate,
    p_value = p.value,
    `Odds ratio` = odds_ratio
  ) %>%
  mutate(
    stars = case_when(
      p_value < 0.001 ~ "***",
      p_value < 0.01  ~ "**",
      p_value < 0.05  ~ "*",
      TRUE ~ ""
    ),
    Coefficient = round(Coefficient, 3),
    `Odds ratio` = round(`Odds ratio`, 2),
    p_value = formatC(p_value, format = "e", digits = 2)
  )
library(flextable)

ft_coef <- flextable(
  coef_word %>%
    rename(`Signif.` = stars)
) %>%
  autofit() %>%
  theme_vanilla() %>%
  set_caption("Coefficients significatifs de la régression logistique")
library(officer)

docs <- read_docx() %>%
  body_add_flextable(ft_coef) %>%
  body_add_par(
    value = "Note : *** p < 0,001 ; ** p < 0,01 ; * p < 0,05",
    style = "Normal"
  )

print(docs, target = "coefficients_significatifs_logistique.docx")


##Courbes ROC comparées – Logistique, Random Forest et XGBoost

# ROC – Régression logistique
roc_log <- roc_curve(
  test_pred,
  truth = souscription,
  .pred_yes,
  event_level = "second"
) %>%
  mutate(Modèle = "Régression logistique")

# ROC – Random Forest
roc_rf <- roc_curve(
  test_pred_rf,
  truth = souscription,
  .pred_yes,
  event_level = "second"
) %>%
  mutate(Modèle = "Random Forest")

# ROC – XGBoost
roc_xgb <- roc_curve(
  xgb_test_pred,
  truth = souscription,
  .pred_yes,
  event_level = "second"
) %>%
  mutate(Modèle = "XGBoost")

roc_all <- bind_rows(roc_log, roc_rf, roc_xgb)
ggplot(
  roc_all,
  aes(x = 1 - specificity, y = sensitivity, color = Modèle)
) +
  geom_line(linewidth = 1.2) +
  geom_abline(
    linetype = "dashed",
    color = "grey50"
  ) +
  labs(
    title = "Courbes ROC comparées – Modèles prédictifs",
    subtitle = "Régression logistique, Random Forest et XGBoost",
    x = "Taux de faux positifs (1 – Spécificité)",
    y = "Taux de vrais positifs (Sensibilité)",
    color = "Modèle"
  ) +
  theme_minimal(base_size = 13)

auc_values <- tibble(
  Modèle = c("Régression logistique", "Random Forest", "XGBoost"),
  AUC = c(
    auc_log,
    auc_rf,
    auc_xgb
  )
)

auc_values


###############################################################################
## CLUSTERING – VERSION COHÉRENTE AVEC TRAIN/TEST
###############################################################################

# 1️⃣ On utilise la recette déjà entraînée sur TRAIN
rec_prep <- prep(rec)

# 2️⃣ On transforme TOUT df avec cette recette
df_transformed <- bake(
  rec_prep,
  new_data = df
)

# 3️⃣ On retire la variable cible
df_cluster <- df_transformed %>%
  select(-souscription)

###############################################################################
# 4️⃣ Clustering sur les données transformées
###############################################################################

set.seed(42)

wss <- purrr::map_dbl(1:10, function(k) {
  kmeans(df_cluster, centers = k, nstart = 20)$tot.withinss
})

tibble(k = 1:10, wss = wss) %>%
  ggplot(aes(k, wss)) +
  geom_line(color = "steelblue") +
  geom_point() +
  labs(
    title = "Méthode du coude – Clustering cohérent",
    x = "Nombre de clusters",
    y = "Within Sum of Squares"
  ) +
  theme_minimal()

k_opt <- 4

set.seed(42)

kmeans_model <- kmeans(
  df_cluster,
  centers = k_opt,
  nstart = 25
)

df$cluster <- factor(kmeans_model$cluster)

###############################################################################
# 5️⃣ Analyse des clusters
###############################################################################

df %>%
  count(cluster) %>%
  mutate(proportion = round(n / sum(n) * 100, 1))

df %>%
  group_by(cluster) %>%
  summarise(
    effectif = n(),
    taux_souscription = round(mean(souscription == "yes") * 100, 1)
  )
