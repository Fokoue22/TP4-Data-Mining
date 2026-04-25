# IFT870 — TP4 : Réglage d'Hyperparamètres (KNN)

## Intégrants
- **tedt6643** - Thomas Bylan Tedjoutsem Fokoue
- **qaia7168** - Ahmed Qais

## Enseignant
- **Aida Ouangraoua**

## Description du Projet
Ce travail pratique explore les méthodes de réglage d'hyperparamètres pour un classificateur KNN (K-Nearest Neighbors) appliqué à un dataset de classification multi-classes. Nous implémentons et comparons plusieurs stratégies d'optimisation : recherche exhaustive (brute force), recherche aléatoire, halving, optimisation bayésienne manuelle, et leurs équivalents via scikit-learn.

Le dataset contient 4 attributs numériques et une variable cible avec 20 classes équilibrées, sans valeurs manquantes.

## Prérequis
- Python 3.8+
- Bibliothèques : `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `skopt`

Installez les dépendances via :
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy scopt
```

## Structure du Projet
- `tp4.ipynb` : Notebook Jupyter contenant l'analyse complète
- `TP4-data.csv` : Dataset utilisé pour l'analyse
- `README.md` : Ce fichier de documentation

## Analyse Détaillée

### Exploration du Dataset
Nous commençons par charger et explorer le dataset pour comprendre sa structure.

**Étapes réalisées :**
1. Chargement du fichier `TP4-data.csv` avec `pandas.read_csv()`.
2. Affichage des premières lignes avec `df.head(5)`.
3. Vérification des dimensions : `(nombre_d'échantillons, 5)` (4 attributs + 1 cible).
4. Contrôle des valeurs manquantes : aucune valeur manquante détectée.
5. Analyse de la répartition des classes : 20 classes équilibrées.
6. Visualisation de la distribution des classes avec un histogramme via `seaborn.histplot()`.

**Interprétation :** Le dataset est propre et prêt pour l'analyse, avec une structure simple et des classes équilibrées.

### 1. Implémentation des Fonctions

#### 1.(a) Fonction `model_score(X, y, class_model, params)`
Cette fonction évalue un modèle KNN avec des paramètres donnés en utilisant la validation croisée stratifiée.

**Étapes d'implémentation :**
1. Création d'un pipeline incluant `StandardScaler` (nécessaire pour KNN) et le modèle KNN cloné.
2. Ajustement des paramètres du pipeline avec le préfixe `'knn__'`.
3. Utilisation de `StratifiedKFold` avec 5 plis pour la validation croisée.
4. Calcul des scores de précision avec `cross_val_score` et retour de la moyenne.

**Code clé :**
```python
pipe = Pipeline([('scaler', StandardScaler()), ('knn', clone(class_model))])
pipeline_params = {f"knn__{k}": v for k, v in params.items()}
pipe.set_params(**pipeline_params)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
return scores.mean()
```

#### 1.(b) Fonction `bruteforce_optimisation(class_model, grille_param, X, y)`
Exploration exhaustive de toutes les combinaisons de la grille de paramètres.

**Étapes d'implémentation :**
1. Itération sur toutes les combinaisons générées par `ParameterGrid`.
2. Évaluation de chaque combinaison avec `model_score`.
3. Mise à jour du meilleur score et des meilleurs paramètres à chaque itération.
4. Retour des meilleurs paramètres trouvés.

**Code clé :**
```python
for params in ParameterGrid(grille_param):
    current_score = model_score(X, y, class_model, params)
    if current_score > best_score:
        best_score = current_score
        best_params = params
return best_params
```

**Grille utilisée :** `{"n_neighbors": range(1, 21), "p": range(1, 11)}` (200 combinaisons).

#### 1.(c) Fonction `randomize_optimisation(class_model, grille_param, X, y, sample_percent)`
Échantillonnage aléatoire d'un pourcentage de la grille complète.

**Étapes d'implémentation :**
1. Génération de la grille complète avec `ParameterGrid`.
2. Calcul du nombre d'échantillons basé sur `sample_percent` (ex. 30%).
3. Sélection aléatoire des indices sans remise.
4. Évaluation uniquement des paramètres échantillonnés.
5. Retour des meilleurs paramètres dans l'échantillon.

**Code clé :**
```python
n_samples = int(len(full_grid) * (sample_percent / 100))
sampled_indices = np.random.choice(len(full_grid), n_samples, replace=False)
sampled_grid = [full_grid[i] for i in sampled_indices]
```

#### 1.(d) Fonction `halving_optimisation(class_model, grille_param, X, y, n_splitting)`
Stratégie de réduction progressive (Successive Halving) avec augmentation de la taille des données.

**Étapes d'implémentation :**
1. Initialisation avec tous les paramètres candidats.
2. Pour chaque étape `s` de 1 à `n_splitting` :
   - Calcul de la taille des données : `N * s / n_splitting`.
   - Échantillonnage aléatoire des données.
   - Évaluation de tous les candidats actuels.
   - Tri et conservation des meilleurs candidats : `P * (n_splitting - s) / n_splitting`.
3. À la dernière étape, retour du meilleur paramètre.

**Code clé :**
```python
data_size = int(N * s / n_splitting)
indices = np.random.choice(N, data_size, replace=False)
X_sub, y_sub = X.iloc[indices], y.iloc[indices]
n_to_keep = max(1, int(P * (n_splitting - s) / n_splitting))
```

#### 1.(e) Fonction `bayesian_optimisation(class_model, grille_param, X, y, s_size, n_iter)`
Optimisation bayésienne manuelle utilisant un processus gaussien.

**Étapes d'implémentation :**
1. Échantillonnage initial `s_size` de paramètres et évaluation.
2. Entraînement d'un `GaussianProcessRegressor` avec noyau Matern.
3. Pour chaque itération :
   - Prédiction sur toute la grille pour estimer les moyennes et écarts-types.
   - Calcul de la fonction d'acquisition (probabilité d'amélioration) sur un échantillon candidat.
   - Sélection et évaluation du meilleur candidat.
4. Retour du meilleur paramètre observé.

**Code clé :**
```python
means, stds = gp.predict(grid_array, return_std=True)
probas = norm.cdf((m_cand - max_pred_moy) / (s_cand + 1e-6))
```

### 2. Comparaison des Méthodes

#### 2.(a) Partitionnement des Données
Division stratifiée du dataset en ensembles d'entraînement et de test.

**Étapes réalisées :**
1. Utilisation de `train_test_split` avec `stratify=y` pour maintenir la répartition des classes.
2. Paramètres : 70% entraînement, 30% test, `random_state=42`.
3. Vérification des dimensions : `(train_size, 4)` et `(test_size, 4)`.

**Interprétation :** Le split stratifié évite les biais dus à l'absence d'une classe dans le test. L'accuracy reste une métrique pertinente.

#### 2.(b) Heatmaps des Scores
Visualisation des performances sur toute la grille de paramètres.

**Étapes réalisées :**
1. Entraînement d'un modèle pour chaque combinaison de paramètres.
2. Calcul des accuracies sur train et test.
3. Création de heatmaps avec `seaborn.heatmap` pour `n_neighbors` vs `p`.
4. Analyse visuelle des zones de haute performance.

**Interprétation :** Les scores élevés (>0.99) indiquent une forte séparabilité des classes. Le paysage est plat, expliquant la stabilité des méthodes.

**Projection PCA :**
- Réduction à 2 composantes expliquant ~50% de la variance.
- Visualisation des 20 classes en amas distincts, confirmant la séparabilité.

#### 2.(c) Application et Comparaison des 8 Méthodes
Exécution et benchmarking des stratégies d'optimisation.

**Méthodes implémentées :**
1. **Bruteforce** : Exploration exhaustive.
2. **Randomize (30%)** : Échantillonnage à 30%.
3. **Halving (5 steps)** : Réduction progressive.
4. **Bayesian (manuelle)** : Optimisation bayésienne avec GP.
5. **GridSearchCV** : Implémentation scikit-learn.
6. **RandomizedSearchCV** : Version randomisée scikit-learn.
7. **HalvingGridSearchCV** : Halving scikit-learn.
8. **gp_minimize (skopt)** : Optimisation bayésienne avec skopt.

**Étapes pour chaque méthode :**
1. Mesure du temps d'exécution avec `time.perf_counter()`.
2. Récupération des meilleurs paramètres.
3. Calcul du score CV moyen sur train.
4. Évaluation de l'accuracy finale sur test.
5. Stockage des résultats dans un DataFrame pour comparaison.

**Résultats clés :**
- Toutes les méthodes trouvent des paramètres optimaux avec accuracy >0.99.
- Temps : Bruteforce le plus lent, méthodes randomisées/halving plus rapides.
- Performance : Méthodes bayésiennes efficaces pour explorer l'espace.

## Conclusion
Ce TP démontre l'importance du réglage d'hyperparamètres pour optimiser les performances des modèles ML. Les méthodes implémentées offrent un équilibre entre exhaustivité et efficacité, avec les approches bayésiennes particulièrement adaptées aux espaces de recherche complexes.

Le dataset utilisé, bien que simple, illustre parfaitement les concepts enseignés, avec une forte séparabilité permettant d'atteindre des accuracies quasi-parfaites.
avec ce modèle retourne un vecteur des moyennes et un vector des écart-types des distri
butions prédictives à chaque donnée. La méthode commence par générer un échantillon
E de s_size paramétrages qu’elle utilise pour estimer (fit) la fonction d’approximation.
Puis, elle répète n_iter fois le processus suivant :
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