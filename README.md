# IFT870 — TP4 : Réglage d'Hyperparamètres (KNN)


## Description  des tâches à réaliser :
Onvousfournit un ensemble de données stockées dans un fichier au format .csv (TP4_data.csv). L’ensemble des données contient 10,000 observations représentées suivant 4 variables (Attribut1,Attribut2, Attribut3, Attribut4). Les données sont segmentées en 20 classes (0,1,...,19). La classe d’une observation est représentée par la valeur de la variable Classe dans le fichier de données. L’objectif du TP est d’implémenter, utiliser et comparer les résultats de fonctions de réglage d’hyperparamètres de type GridSearch pour le modèle de classification KNeighborsClassifier. Les hyperparamètres à régler sont : n_neighbors dans l’intervalle de valeurs entières [1,20], et p dans l’intervalle de valeurs entières [1,10].

**Donc**
Ce TP implémente et compare des méthodes de réglage d'hyperparamètres pour KNN sur un dataset de 10 000 échantillons (4 attributs, 20 classes équilibrées). Hyperparamètres : `n_neighbors` [1-20], `p` [1-10].

## Prérequis
- Python 3.8+
- Librairies : `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `skopt`

Installation : `pip install numpy pandas scikit-learn matplotlib seaborn scipy scopt`

## Structure
- `tp4.ipynb` : Analyse complète
- `TP4-data.csv` : Dataset
- `README.md` : Documentation

## Analyse Détaillée

### Exploration Dataset
- Chargement avec `pandas.read_csv()`.
- Dimensions : 10 000 lignes, 5 colonnes (4 features + classe).
- Pas de valeurs manquantes.
- 20 classes équilibrées.
- Visualisation : histogramme des classes avec `seaborn`.

### 1. Implémentation Fonctions  : Vous devez implémenter :

#### 1.(a)  une fonction  `model_score(X, y, class_model, params)`  
qui prend en paramètre une matrice X de données, un vecteur y de classes correspondant, un modèle de classification `(exemple : class_model = KNeighborsClassifier())`, et un paramétrage exemple : params = `({’n_neighbors’: 10, ’p’: 5})`. La fonction retourne la moyenne de score de précision (accuracy) par validation croisée en divisant les données en 5 parties.


#### 1.(b)  une fonction `bruteforce_optimisation(class_model, grille_param, X, y)`
qui prend en paramètre un modèle de classification, une grille de paramètres `(exemple : grille_param= {’n_neighbors’: range(1,11), ’p’: range(1,6)})`, une matrice X de données, et un vecteur y de classes correspondant. La fonction explore tout l’espace de recherche des paramétrages et retourne un seul paramétrage de score maximal `(exemple : max_params = {’n_neighbors’: 10, ’p’: 5})`.

#### 1.(c)  une fonction `randomize_optimisation(class_model, grille_param, X, y, sample_percent)`
Échantillonnage aléatoire (ex. 30%).
- Sélectionne `sample_percent` de la grille.
- Évalue et retourne meilleurs params.

#### 1.(d)  une fonction `halving_optimisation(class_model, grille_param, X, y, n_splitting)`
Successive Halving (ex. 5 étapes).
- Réduit candidats et augmente données progressivement.
- Retourne paramètre final.

#### 1.(e)  une fonction `bayesian_optimisation(class_model, grille_param, X, y, s_size, n_iter)`
Optimisation bayésienne manuelle.
- Utilise `GaussianProcessRegressor` + fonction d'acquisition.
- Itère pour améliorer l'estimation.

### 2. Comparaison Méthodes

#### 2.(a) Partitionnement
- Split stratifié : 70% train, 30% test (`train_test_split`).

#### 2.(b) Heatmaps
- Calcul accuracy train/test pour chaque paramètre.
- Visualisation avec `seaborn.heatmap`.
- Projection PCA : 20 classes en amas distincts (variance ~50%).

#### 2.(c) Comparaison 8 Méthodes
Méthodes testées :
1. Bruteforce
2. Randomize (30%)
3. Halving (5 steps)
4. Bayesian (manuelle)
5. GridSearchCV
6. RandomizedSearchCV
7. HalvingGridSearchCV
8. gp_minimize (skopt)

- Mesure temps, meilleurs params, score CV, accuracy test.
- Résultats : Toutes atteignent >99% accuracy ; bruteforce lent, autres plus efficaces.

## Conclusion
TP démontre l'importance du tuning. Méthodes bayésiennes et halving équilibrent efficacité et performance sur ce dataset simple avec forte séparabilité.
— elle utilise la fonction d’approximation pour prédire les scores (moyennes et écart
types) de tous les paramétrages;
— elle trouve la moyenne maximum prédite max_pred_moy;
— elle échantillonne s_size paramétrages;
— elle utilise la fonction d’approximation pour prédire les scores (moyennes et écart
types) des paramétrages échantillonnés;
— les résultats de cette prédiction sont transformés en une distribution de probabilité
en utilisant la fonction de distribution cumulative (cdf), comme suit : probabilite
= cdf((moyennes- max_pred_moy) / (ecart_types + 10−6));
— le paramétrage de probabilité maximum est choisi : max_param;
— max_param est ajouté à l’échantillon E, puis E est utilisé pour ré-estimer la fonction
d’approximation.
Après la dernière itération, le paramétrage de score maximum dans l’échantillon E est
choisi.

2. Comparaison de fonctions :
(a) Proposez un partitionnement des données en données d’entraînement et données de test.
(b) Présentez un graphique des scores d’entraînement, et un graphique des scores de test,
pour tous les paramétrages de l’espace de recherche, sous forme de heatmap.
(c) Appliquez les fonctions suivantes pour le réglage des hyperparamètres, et commentez les
résultats :
— bruteforce_optimisation
— randomize_optimisation avec sample_percent = 30
— halving_optimisation avec s_splitting = 5
— bayesian_optimisation avec s_size = 5 et n_iter = 100
— model_selection.GridSearchCV avec scoring=’accuracy’
— model_selection.RandomizedSearchCV avec n_iter = 60etscoring=’accuracy’
— model_selection.HalvingGridSearchCV avec factor = 5etscoring=’accuracy’
— skopt.gp_minimize avec comme modèle de score 1- accuracy
(d) En faisant varier le partitionnement des données et les paramètres des fonctions de ré
glage des hyperparamètres, comparez les performances des fonctions en termes de temps
de calcul, et de capacité à trouver un paramétrage optimal. Commentez les résultats.