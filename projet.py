# Matteo PIETRI
# Sara CASTRO LÓPEZ

import pandas as pd  # package for high-performance, easy-to-use data structures and data analysis
import numpy as np  # fundamental package for scientific computing with Python
import math
import utils
import scipy.stats as stats
import matplotlib.pyplot as plt


#####
# QUESTION 1.1 : Calcul de la probabilité a priori
#####
def getPrior(dataframe):
    """
    Calcule la probabilité a priori d'une classe et l'intervalle de confiance à 95%.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Le dataframe contenant les données, avec une colonne 'target'.

    Returns
    -------
        Un dictionnaire contenant 'estimation', 'min5pourcent', et 'max5pourcent'.
    """


    n = len(dataframe)
    p = dataframe['target'].mean()

    marge_erreur = 1.96 * math.sqrt(p * (1 - p) / n)
    borne_inf = max(0, p - marge_erreur)
    borne_sup = min(1, p + marge_erreur)

    return {
        'estimation': float(p),
        'min5pourcent': float(borne_inf),
        'max5pourcent': float(borne_sup)
    }

#####
# QUESTION 1.2 : Programmation orientée objet dans la hiérarchie des Classifier
#####
class APrioriClassifier(utils.AbstractClassifier):
    """
    Classifieur a priori basé sur la classe majoritaire.
    Hérite de AbstractClassifier.
    """

    def __init__(self, dataframe):
        """
        Initialise le classifier en calculant la classe majoritaire à partir de la fonction getPrior(df).

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Le DataFrame contenant les données avec une colonne 'target'.
        """
        self.prior = getPrior(dataframe)
        self.majority_class = 1 if self.prior['estimation'] > 0.5 else 0  # Classe majoritaire

    #####
    # QUESTION 1.2.a : Estimer la classe majoritaire
    #####
    def estimClass(self, attrs):
        """
        À partir d'un dictionnaire d'attributs, estime la classe 0 ou 1

        Parameters
        ----------
        attrs: Dict[str, value]
        Le dictionnaire nom-valeur des attributs

        Returns
        -------
        La classe 0 ou 1 estimée
        """
        return self.majority_class

    #####
    ## QUESTION 1.2.b : Évaluation de classifieurs
    #####
    def statsOnDF(self, dataset):
        """
        Analyse les performances du classifieur sur un DataFrame donné.

        Parameters
        ----------
        dataset : pandas.DataFrame
            Le DataFrame contenant les données à analyser, avec une colonne 'target'.

        Returns
        -------
        dict
            Un dictionnaire contenant les statistiques TP, TN, FP, FN, ainsi que la précision et le rappel.
        """
        vp = vn = fp = fn = 0

        # Utiliser une boucle sur le DataFrame avec .itertuples()
        for t in dataset.itertuples():
            dic = t._asdict()  # Convertir la ligne en dictionnaire
            classe_reelle = dic['target']
            classe_predit = self.estimClass(utils.getNthDict(dataset, t.Index))  # Utiliser getNthDict pour récupérer les attributs

            if classe_reelle == 1 and classe_predit == 1:
                vp += 1
            elif classe_reelle == 0 and classe_predit == 0:
                vn += 1
            elif classe_reelle == 0 and classe_predit == 1:
                fp += 1
            elif classe_reelle == 1 and classe_predit == 0:
                fn += 1

        # Calcul des métriques
        marge_erreur = vp / (vp + fp) if (vp + fp) > 0 else 0
        rappel = vp / (vp + fn) if (vp + fn) > 0 else 0

        return {
            'VP': vp,
            'VN': vn,
            'FP': fp,
            'FN': fn,
            'Précision': round(marge_erreur, 16),
            'Rappel': round(rappel, 16)
        }

#####
# QUESTION 2.1.a : Probabilités conditionelles
#####
def P2D_l(df, attr):
    """
    Calcule la probabilité conditionnelle P(attr | target) pour toutes les valeurs
    possibles de l'attribut et des classes cibles.

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données.
    attr : str
        Le nom de la colonne (attribut) pour laquelle on calcule P(attr | target).

    Returns
    -------
    dict
        Un dictionnaire imbriqué où les probabilités sont calculées sans lissage de Laplace.
    """
    valeurs_target = df['target'].unique()
    valeurs_attribut = df[attr].unique()
    p = {}

    # Calculer les probabilités conditionnelles P(attr | target)
    for valeur_target in valeurs_target:
        valeur_target_int = int(valeur_target)  # Convertir pour éviter numpy.int64
        subset = df[df['target'] == valeur_target]
        total_cible = len(subset)

        # Initialiser le dictionnaire pour cette classe
        p[valeur_target_int] = {}

        for valeur_attribut in valeurs_attribut:
            valeur_attribut_int = int(valeur_attribut)  # Convertir pour éviter numpy.int64
            count_attribut = len(subset[subset[attr] == valeur_attribut])  # Compter les occurrences

            prob = count_attribut / total_cible if total_cible > 0 else 0
            p[valeur_target_int][valeur_attribut_int] = prob  # ajout au dictionnaire

    return p

#####
# QUESTION 2.1.b : Probabilités conditionelles
#####
def P2D_p(df, attr):
    """
    Calcule la probabilité conditionnelle P(target | attr) pour toutes les valeurs
    possibles de l'attribut donné et des classes cibles.

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données.
    attr : str
        Le nom de la colonne (attribut) pour laquelle on calcule P(target | attr).

    Returns
    -------
    dict
        Un dictionnaire imbriqué où les clés sont les valeurs de l'attribut
        et les sous-clés sont les classes cibles avec leurs probabilités.
    """
    valeurs_attribut = df[attr].unique()
    valeurs_target = df['target'].unique()
    p = {}

    for valeur_attribut in valeurs_attribut:
        valeur_attribut_int = int(valeur_attribut)  # Convertir en entier pour éviter numpy.int64
        subset = df[df[attr] == valeur_attribut]
        total_attribut = len(subset)
        p[valeur_attribut_int] = {}

        for valeur_target in valeurs_target:
            valeur_target_int = int(valeur_target) # Convertir en entier pour éviter numpy.int64
            count_target = len(subset[subset['target'] == valeur_target])
            prob = count_target / total_attribut if total_attribut > 0 else 0
            p[valeur_attribut_int][valeur_target_int] = prob

    return p

#####
# QUESTION 2.2 : Classifieurs 2D par maximum de vraisemblance
#####
class ML2DClassifier(APrioriClassifier):
    """
    Classifieur 2D basé sur le principe du maximum de vraisemblance.
    Hérite de APrioriClassifier.
    """

    def __init__(self, dataframe, attr_column):
        """
        Initialise le classifieur en construisant une table de probabilités P2D
        pour l'attribut donné.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Le DataFrame contenant les données avec une colonne 'target'.
        attr_column : str
            Le nom de l'attribut utilisé pour construire la table P2D.
        """
        super().__init__(dataframe)  # Initialise la classe parent
        self.attr_column = attr_column
        self.table_P2D = self._construct_P2D(dataframe, attr_column)

    def _construct_P2D(self, data, attr):
        """
        Construit une table de probabilités conditionnelles P2D pour l'attribut donné.

        Parameters
        ----------
        data : pandas.DataFrame
            Le DataFrame contenant les données avec une colonne 'target'.
        attr : str
            Le nom de l'attribut pour lequel construire la table P2D.

        Returns
        -------
        dict
            Une table P2D contenant les probabilités conditionnelles pour chaque classe
            (target = 0 ou target = 1) et les valeurs possibles de l'attribut.
        """
        return P2D_l(data, attr)

    def estimClass(self, individual):
        """
        Estime la classe d'un individu donné en utilisant la table P2D.

        Parameters
        ----------
        individual : dict
            Un dictionnaire représentant les attributs d'un individu, y compris
            la colonne correspondant à attr_column.

        Returns
        -------
        int
            La classe estimée (0 ou 1) en fonction de la probabilité maximale.
        """
        # Récupérer la valeur de l'attribut correspondant à attr_column
        value = individual[self.attr_column]

        # Initialiser un dictionnaire pour stocker les probabilités par classe
        target_probs = {}

        # Parcourir les classes dans table_P2D (0 et 1)
        for target in self.table_P2D:
            # Récupérer la probabilité associée à la valeur de l'attribut pour la classe actuelle
            prob = self.table_P2D[target].get(value, 0)  # Retourne 0 si la valeur n'existe pas
            target_probs[target] = prob

        # Retourner la classe avec la probabilité maximale
        return max(target_probs, key=target_probs.get)

#####
# QUESTION 2.3 : Classifieurs 2D par maximum a posteriori
#####
class MAP2DClassifier(APrioriClassifier):
    """
    Classifieur 2D basé sur le principe du maximum à priori.
    Hérite de APrioriClassifier.
    """

    def __init__(self, dataframe, attr_column):
        """
        Initialise le classifieur en construisant une table de probabilités P2Dp
        pour l'attribut donné.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Le DataFrame contenant les données avec une colonne 'target'.
        attr_column : str
            Le nom de l'attribut utilisé pour construire la table P2Dp.
        """
        super().__init__(dataframe)  # Initialise la classe parent
        self.attr_column = attr_column
        self.table_P2Dp = self._construct_P2Dp(dataframe, attr_column)

    def _construct_P2Dp(self, data, attr):
        """
        Construit une table de probabilités conditionnelles P2D pour l'attribut donné.

        Parameters
        ----------
        data : pandas.DataFrame
            Le DataFrame contenant les données avec une colonne 'target'.
        attr : str
            Le nom de l'attribut pour lequel construire la table P2Dp.

        Returns
        -------
        dict
            Une table P2Dp contenant les probabilités conditionnelles pour chaque valeur de l'attribut
            (attr = a) et les valeurs possibles de la clase (target = 1, target = 0).
        """
        return P2D_p(data, attr)

    def estimClass(self, individual):
        """
        Estime la classe d'un individu donné en utilisant la table P2Dp.

        Parameters
        ----------
        individual : dict
            Un dictionnaire représentant les attributs d'un individu.

        Returns
        -------
        int
            La classe estimée (0 ou 1) en fonction de la probabilité conditionnelle calculée.
        """
        # Récupérer la valeur de l'attribut correspondant à attr_column
        value = individual[self.attr_column]

        # Initialiser un dictionnaire pour stocker les probabilités par classe
        target_probs = {}

        # Parcourir les classes dans table_P2Dp (0 et 1)
        for target in self.table_P2Dp[value]:
            # Récupérer la probabilité conditionnelle associée à la valeur de l'attribut
            target_probs[target] = self.table_P2Dp[value][target]

        # Retourner la classe avec la probabilité maximale
        # En cas d'égalité, retourne 0 comme spécifié dans l'énoncé
        return max(target_probs, key=lambda t: (target_probs[t], -t))  # Priorité à 0 en cas d'égalité
    
#####
# QUESTION 2.4 : Comparaison
##### 
'''
Avant d’exprimer notre opinion, nous tenons à souligner que les meilleures mesures de prédiction 
peuvent varier en fonction du contexte dans lequel elles sont utilisées. Il est donc essentiel 
d’évaluer chaque cas avec soin afin de parvenir à une conclusion adaptée. 

Dans ce problème, l’objectif est de prédire si une personne est malade (target = 1) ou en bonne santé 
(target = 0) en utilisant des attributs individuels comme l’âge, le sexe, et d’autres caractéristiques. 

Parmi les classifieurs développés, le classifieur a priori (APrioriClassifier) est le plus simple, 
car il prédit toujours la classe majoritaire du jeu de données. Sa précision dépend uniquement de la proportion 
de la classe majoritaire dans l’échantillon, ce qui serait acceptable s’il existait une nette majorité dans les classes. 
Cependant, ce n’est pas le cas ici, car l’équilibre entre les classes n’est pas suffisamment marqué, 
ce qui limite considérablement son utilité. 

En revanche, le classifieur par maximum de vraisemblance (ML2DClassifier) améliore significativement 
les performances en prenant en compte les probabilités conditionnelles P(attr∣target), atteignant une bonne 
précision dans notre contexte. Toutefois, il ne tient pas compte des proportions globales des classes, ce qui peut 
poser problème dans des jeux de données déséquilibrés. 

Enfin, le classifieur par maximum à posteriori (MAP2DClassifier) combine les probabilités conditionnelles 
avec les proportions globales des classes, ce qui lui permet de mieux gérer des scénarios réels où les classes sont souvent 
déséquilibrées. Bien qu’il puisse avoir une précision légèrement inférieure à celle du classifieur précédent, 
nous estimons qu’il est globalement le plus robuste et fiable pour ce type de problème. 

Pour ces raisons, nous recommandons le MAP2DClassifier comme la meilleure option dans ce cas.
'''
#####

#####
# QUESTION 3.1 : Complexité en mémoire
##### 
def _format_memory(taille):
    """
    Formate la taille mémoire en unités lisibles (Go, Mo, Ko, octets).

    Cette fonction décompose une taille donnée (en octets) en unités plus grandes,
    comme les gigaoctets (Go), mégaoctets (Mo) et kilooctets (Ko), tout en conservant
    les octets restants. Par exemple, une taille de 12345678 octets sera formatée
    comme "11Mo 772Ko 470o".

    Parameters
    ----------
    taille : int
        La taille mémoire en octets à formater.

    Returns
    -------
    str
        Une chaîne de caractères représentant la taille décomposée en Go, Mo, Ko, et octets,
        par ordre décroissant d'unité.
    
    Exemple
    -------
    >>> _format_memory(12345678)
    '11Mo 772Ko 470o'

    >>> _format_memory(1025)
    '1Ko 1o'
    """
    unités = [("Go", 1024**3), ("Mo", 1024**2), ("Ko", 1024), ("o", 1)]
    decomposition = []
    for unité, facteur in unités:
        if taille >= facteur:
            valeur = taille // facteur
            decomposition.append(f"{valeur}{unité.lower()}")
            taille %= facteur
    return " ".join(decomposition)

def nbParams(df, attrs=None):
    """
    Calcule et affiche la mémoire nécessaire pour une table P(target | attrs) en Go, Mo, Ko, etc.

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données.
    attrs : list of str, optional
        Liste des attributs à utiliser avec 'target' pour calculer la mémoire.
        Si aucun attribut n'est spécifié, tous les attributs seront utilisés.

    Returns
    -------
    int
        Taille en octets nécessaire pour la table.
    """
    # Utiliser tous les attributs si aucun n'est spécifié
    if not attrs:
        attrs = df.columns.tolist()
    
    # Calculer le nombre de combinaisons uniques
    valeurs_uniques = [len(P2D_p(df, attr)) for attr in attrs]
    total_combinations = np.prod(valeurs_uniques)
    
    # Calculer la mémoire requise (8 octets par combinaison)
    memoire = int(total_combinations * 8)  # En octets

    # Afficher le résultat formaté
    if memoire > 1024:
        print(f"{len(attrs)} variable(s) : {memoire}o = {_format_memory(memoire)}")
    else:
        print(f"{len(attrs)} variable(s) : {memoire}o")

    return memoire

#####
# QUESTION 3.2 : Complexité en mémoire sous hypothèse d'indépendance complète
##### 
def nbParamsIndep(df, attrs=None):
    """
    Calcule la mémoire nécessaire pour une table P(attrs) sous l'hypothèse d'indépendance
    des variables, c'est-à-dire P(A, B, C, ...) = P(A) * P(B) * P(C) ...

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données.
    attrs : list of str, optional
        Liste des attributs pour lesquels on calcule la mémoire.
        Si aucun attribut n'est spécifié, tous les attributs seront utilisés.

    Returns
    -------
    int
        Taille en octets nécessaire pour les tables indépendantes.
    """
    # Utiliser tous les attributs si aucun n'est spécifié
    if not attrs:
        attrs = df.columns.tolist()
    
    # Calculer la mémoire pour chaque attribut indépendamment
    memoire = sum(len(P2D_p(df, attr)) * 8 for attr in attrs)

    # Afficher le résultat formaté
    if memoire > 1024:
        print(f"{len(attrs)} variable(s) : {memoire}o = {_format_memory(memoire)}")
    else:
        print(f"{len(attrs)} variable(s) : {memoire}o")

    return memoire

#####
# QUESTION 3.3.a. Montrer que P(A,B,C)=P(A)*P(B|A)*P(C|B)
#####
'''
P(A,B,C) = P(A) * P(B,C|A) = P(A) * P(B|A) * P(C|B,A) => Si C et A sont indépendantes => P(A) * P(B|A) * P(C|B)
'''
#####

#####
# QUESTION 3.3.b. Si les 3 variables A, B et C ont 5 valeurs, quelle est la taille mémoire en octet nécessaire pour représenter cette distribution avec et sans l'utilisation de l'indépendance conditionnelle ?
#####
'''
Cas 1 : Sans indépendance conditionnelle
--------------------------------------------------

On doit stocker toutes les combinaisons possibles de A, B et C. Cela se calcule comme suit :
  nbCombinaisons = |A| * |B| * |C| = 5 * 5 * 5 = 125

Chaque combinaison occupe 8 octets, donc la mémoire totale nécessaire est :
  MemTotale = 125 * 8 = 1000 octets

Cas 2 : Avec indépendance conditionnelle partielle
--------------------------------------------------
Sous l'hypothèse d'indépendance conditionnelle :
  P(A, B, C) = P(A) * P(B|A) * P(C|B)

On calcule la mémoire pour représenter chaque composante séparément :

  1. P(A) :
     - On doit stocker les probabilités de A (5 valeurs).
     - MemA = 5 * 8 = 40 octets
  2. P(B|A) :
     - Pour chaque valeur de A (5 valeurs), on stocke les probabilités de B (5 valeurs).
     - MemBA = 5 * 5 * 8 = 200 octets
  3. P(C|B) :
     - Pour chaque valeur de B (5 valeurs), on stocke les probabilités de C (5 valeurs).
     - MemCB = 5 * 5 * 8 = 200 octets

Ainsi, la mémoire totale nécessaire est :
MemTotale = 40 + 200 + 200 = 440 octets

Conclusion :
L'indépendance conditionnelle partielle réduit significativement la mémoire nécessaire, démontrant une représentation plus efficace pour la distribution P(A, B, C).
'''
#####

#####
# QUESTION 4.1. Exemples #### RÉVISER ####
#####
'''
Cas 1
-------------------------------------------------------------
Si les 5 variables sont totalement indépendantes, cela signifie que :
P(A, B, C, D, E) = P(A) ⋅ P(B) ⋅ P(C) ⋅ P(D) ⋅ P(E)

Cela pourrait être représenté par un graphe où aucun sommet n'a de parent.
Autrement dit, le graphe résultant serait un graphe sans arêtes (graphe nul ou vide).

Cas 2
-------------------------------------------------------------
Si les variables ne sont pas indépendantes entre elles, cela signifie que :
P(A, B, C, D, E) = P(A) ⋅ P(B|A) ⋅ P(C|A, B) ⋅ P(D|A, B, C) ⋅ P(E|A, B, C, D)
Le graphe résultant sera un graphe orienté complet.
Cela signifie que chaque sommet a une flèche dirigée vers tous les autres sommets.
'''
#####

#####
# QUESTION 4.2. Naïve Bayes 
#####
'''

Cas 1 : Décomposition de la vraisemblance
-----------------------------------------------------------------------------
P(attr1, attr2, attr3, ..., attrN | target) = P(attr1 | target) * P(attr2 | target) * P(attr3 | target) * ... * P(attrN | target)

Grâce à l'hypothèse d'indépendance conditionnelle, la probabilité conjointe des attributs conditionnée à la variable cible 
est simplifiée en un produit des probabilités conditionnelles de chaque attribut.

Cas 2 : Décomposition de la distribution à posteriori
-----------------------------------------------------------------------------
La probabilité a posteriori est calculée grâce au théorème de Bayes, qui s'écrit comme suit :
P(target | attr1, attr2, ..., attrN) = P(attr1, attr2, ..., attrN | target) * P(target) / P(attr1, attr2, ..., attrN)

En utilisant l'hypothèse de Naïve Bayes (indépendance conditionnelle des attributs), la vraisemblance peut être réécrite comme : 
P(attr1, attr2, ..., attrN | target) = P(attr1 | target) * P(attr2 | target) * ... * P(attrN | target)

Ainsi, la probabilité a posteriori devient :
P(target | attr1, attr2, ..., attrN) ∝ P(target) * P(attr1 | target) * P(attr2 | target) * ... * P(attrN | target)
'''

#####
# QUESTION 4.3.a. Modèle graphique et naïve bayes
#####
def drawNaiveBayes(df, obj="target") :
    """
    Dessine le modèle graphique de Naïve Bayes basé sur un DataFrame.

    Cette fonction génère un graphe orienté où le nœud cible (`obj`) est connecté par des flèches 
    à tous les autres attributs du DataFrame, suivant le modèle graphique de Naïve Bayes. 
    Dans ce modèle, tous les attributs sont supposés indépendants conditionnellement à la cible (`obj`).

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les colonnes représentant les attributs et la cible.
    obj : str, optionnel
        Le nom de la colonne représentant la cible (par défaut "target").

    Returns
    -------
    Image
        L'image du graphe générée par la bibliothèque `utils.drawGraph`.
    """
    attrs = [attr for attr in df.columns if attr != obj]  # On ne prend pas en compte le target
    graph = ";".join([f"{obj}->{attr}" for attr in attrs])  # On construit les conexiones

    return utils.drawGraph(graph)

#####
# QUESTION 4.3.b. Modèle graphique et naïve bayes
#####
def nbParamsNaiveBayes(df, obj="target", attrs=None):
    """
    Calcule et affiche la mémoire nécessaire pour les tables P(target | attr)
    en utilisant l'hypothèse de Naïve Bayes (indépendance conditionnelle des attributs).

    Cette fonction calcule la taille mémoire totale en tenant compte des combinaisons possibles
    entre les valeurs de la cible (target) et les attributs fournis. Un ajustement de -16 octets
    est appliqué pour éviter de doubler la mémoire pour P(target), déjà incluse dans les tables
    P(target | attr).

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données.
    obj : str, optional
        La colonne représentant la cible (par défaut "target").
    attrs : list of str, optional
        Liste des attributs à utiliser pour calculer la mémoire.
        Si aucun attribut n'est spécifié, toutes les colonnes sont utilisées.

    Returns
    -------
    int
        Taille totale en octets nécessaire pour les tables.
    """
    # Exclure la colonne cible si attrs n'est pas spécifié
    if attrs is None:
        attrs = [col for col in df.columns]

    # Calculer la mémoire totale
    if len(attrs) == 0:
        memoire_totale = 16  # target=1 ; target=0
    else:
        memoire_totale = 0  
        for attr in attrs:
            nb_valeurs_target = df[obj].nunique()  # Nombre de valeurs uniques dans target
            nb_valeurs_attr = df[attr].nunique()   # Nombre de valeurs uniques dans l'attribut
            memoire_table = nb_valeurs_target * nb_valeurs_attr
            memoire_totale += memoire_table
        
        # Conversion en octets et ajustement pour éviter de doubler P(target)
        memoire_totale = memoire_totale * 8 - 16

    # Afficher le résultat formaté (octets, Ko, Mo, etc.)
    if memoire_totale > 1024:
        print(f"{len(attrs)} variable(s) : {memoire_totale}o = {_format_memory(memoire_totale)}")
    else:
        print(f"{len(attrs)} variable(s) : {memoire_totale}o")
    return memoire_totale

#####
# QUESTION 4.4. Classifieurs Naïve Bayes
#####
class MLNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur Naïve Bayes basé sur le Maximum de Vraisemblance (ML).
    Hérite de APrioriClassifier.
    """
    def __init__(self, dataframe):
        """
        Initialise le classifieur Naïve Bayes avec les probabilités conditionnelles et a priori.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Le DataFrame contenant les données d'apprentissage.
        """
        super().__init__(dataframe)  # Appel au constructeur de APrioriClassifier
        self.attrs = [col for col in dataframe.columns if col != 'target'] # On prend tous les attributs
        self.cond_probs = {attr: P2D_l(dataframe, attr) for attr in self.attrs}  # P(attrN | target) , N=1...len(attrs)

    def estimProbas(self, x):
        """
        Calcule les vraisemblances P(attr1, attr2, ..., attrN | target).

        Parameters
        ----------
        x : dict
            Un dictionnaire représentant un individu (attr1: val1, attr2: val2, ...).

        Returns
        -------
        dict
            Un dictionnaire contenant les vraisemblances pour les classes cible 0 et 1.
        """
        # Initialiser les résultats pour les deux classes cibles
        result1 = 1  # Pour target=0
        result2 = 1  # Pour target=1

        # Parcourir les attributs
        for attr in self.attrs:
            attr_value = x.get(attr, None)  # Récupérer la valeur de l'attribut dans x

            # Calculer pour target=0
            if attr_value in self.cond_probs[attr][0]:
                result1 *= self.cond_probs[attr][0][attr_value]
            else:
                result1 *= 0  # Probabilité nulle si la valeur n'est pas présente

            # Calculer pour target=1
            if attr_value in self.cond_probs[attr][1]:
                result2 *= self.cond_probs[attr][1][attr_value]
            else:
                result2 *= 0  # Probabilité nulle si la valeur n'est pas présente

        # Retourner les résultats sous forme de dictionnaire
        return {0: result1, 1: result2}


    def estimClass(self, x):
        """
        Estime la classe d'un individu en utilisant les vraisemblances (ML).

        Parameters
        ----------
        x : dict
            Un dictionnaire représentant un individu.

        Returns
        -------
        object
            La classe estimée (valeur de target).
        """
        probas = self.estimProbas(x)
        return max(probas, key=probas.get)  # Retourne la classe avec la vraisemblance maximale

class MAPNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur Naïve Bayes basé sur le Maximum A Posteriori (MAP).
    Hérite de APrioriClassifier.
    """
    def __init__(self, dataframe):
        """
        Initialise le classifieur Naïve Bayes avec les probabilités conditionnelles
        et les probabilités a priori P(target).

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Le DataFrame contenant les données d'apprentissage.
        """
        super().__init__(dataframe)  # Appel au constructeur de APrioriClassifier
        self.df = dataframe
        self.attrs = [col for col in dataframe.columns if col != 'target']  # On prend tous les attributs

        # Calcul des probabilités conditionnelles P(attr | target)
        self.cond_probs = {attr: P2D_l(dataframe, attr) for attr in self.attrs}

        # Calcul des probabilités a priori P(target)
        self.prior_probs = {
            0: 1 - self.prior['estimation'],  # P(target=0)
            1: self.prior['estimation']       # P(target=1)
        }

    def estimProbas(self, x):
        """
        Calcule les probabilités a posteriori P(target | attr1, attr2, ..., attrN).

        Parameters
        ----------
        x : dict
            Un dictionnaire représentant un individu (attr1: val1, attr2: val2, ...).

        Returns
        -------
        dict
            Un dictionnaire contenant les probabilités a posteriori pour chaque classe cible.
        """
        # Estimer P(attrs | target) pour chaque classe
        result1 = 1  # Pour target=0
        result2 = 1  # Pour target=1

        for attr in self.attrs:
            attr_value = x.get(attr, None)
            if attr_value in self.cond_probs[attr][0]:
                result1 *= self.cond_probs[attr][0][attr_value]
            else:
                result1 *= 0

            if attr_value in self.cond_probs[attr][1]:
                result2 *= self.cond_probs[attr][1][attr_value]
            else:
                result2 *= 0

        # Ajouter les probabilités a priori
        p = self.prior_probs.get(1, 0)  # P(target=1)
        s = result1 * (1 - p) + result2 * p  # Dénominateur de normalisation

        if s != 0:
            return {0: result1 * (1 - p) / s, 1: result2 * p / s}
        return {0: 0, 1: 0}


    def estimClass(self, x):
        """
        Estime la classe d'un individu en utilisant les probabilités a posteriori (MAP).

        Parameters
        ----------
        x : dict
            Un dictionnaire représentant un individu.

        Returns
        -------
        object
            La classe estimée (valeur de target).
        """
        probas = self.estimProbas(x)
        return max(probas, key=probas.get)  # Retourne la classe avec la probabilité a posteriori maximale

#####
# QUESTION 5.1. Voir si un certain attribut est indépendant de target
#####
def isIndepFromTarget(df, attr, x):
    """
    Détermine si un attribut est indépendant de 'target' selon le test de chi².
    
    Parameters
    ----------
    df : pandas.DataFrame
        Le dataframe contenant les données.
    attr : str
        Le nom de l'attribut à étudier.
    x : float
        Le seuil de signification (p-value).

    Returns
    -------
    bool
        True si l'attribut est indépendant de 'target', False sinon.
    """
    # Créer une table de contingence (P(attr, target)) directement avec pandas
    contingency_table = pd.crosstab(df[attr], df['target'])
    
    # Appliquer le test de chi²
    _, p_value, _, _ = stats.chi2_contingency(contingency_table)
    
    # Comparer le p-value avec le seuil
    return p_value >= x

#####
# QUESTION 5.2. Classifieurs Naïve Bayes Reduced
#####
class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
    """
    Classifieur Naïve Bayes Reduced basé sur le Maximum de Vraisemblance (ML).
    Hérite de MLNaiveBayesClassifier.
    """
    def __init__(self, dataframe, x):
        """
        Initialise le classifieur Naïve Bayes avec les probabilités conditionnelles et a priori.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Le DataFrame contenant les données d'apprentissage.
        """
        super().__init__(dataframe)  # Appel au constructeur de APrioriClassifier
        # Filtrer les attributs significatifs
        self.x = x
        # On ne prend que les attributs qui ne sont pas indépendants avec target
        self.attrs = [col for col in dataframe.columns if col != 'target' and not isIndepFromTarget(dataframe, col, self.x)]
        self.cond_probs = {attr: P2D_l(dataframe, attr) for attr in self.attrs}  # P(attrN | target) , N=1...len(attrs)

    def draw(self, obj='target'):
        """
        Crée une chaîne de caractères représentant un graphe dirigé où
        l'objet `obj` est connecté à tous les autres attributs.

        Parameters
        ----------
        obj : str
            Le nom de l'objet central du graphe (par défaut 'target').

        Returns
        -------
        str
            Une représentation graphique sous forme de chaîne de caractères.
        """
        # Créer les arêtes (obj -> chaque attribut sauf obj)
        edges = [f"{obj}->{attr};" for attr in self.attrs if attr != obj]
    
        # Générer la chaîne et la dessiner avec utils.drawGraph
        graph_string = "".join(edges)
        return utils.drawGraph(graph_string)

class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):
    """
    Classifieur Naïve Bayes Reduced basé sur le Maximum A Posteriori (MAP).
    Hérite de MAPNaiveBayesClassifier.
    """
    def __init__(self, dataframe, x):
        """
        Initialise le classifieur Naïve Bayes avec les probabilités conditionnelles
        et les probabilités a priori P(target).

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Le DataFrame contenant les données d'apprentissage.
        """
        super().__init__(dataframe)  # Appel au constructeur de APrioriClassifier
        self.x = x 
        # On ne prend que les attributs qui ne sont pas indépendants avec target
        self.attrs = [col for col in dataframe.columns if col != 'target' and not isIndepFromTarget(dataframe, col, self.x)]

        # Calcul des probabilités conditionnelles P(attr | target)
        self.cond_probs = {attr: P2D_l(dataframe, attr) for attr in self.attrs}

    def draw(self, obj='target'):
        """
        Crée une chaîne de caractères représentant un graphe dirigé où
        l'objet `obj` est connecté à tous les autres attributs.

        Parameters
        ----------
        obj : str
            Le nom de l'objet central du graphe (par défaut 'target').

        Returns
        -------
        str
            Une représentation graphique sous forme de chaîne de caractères.
        """
        # Créer les arêtes (obj -> chaque attribut sauf obj)
        edges = [f"{obj}->{attr};" for attr in self.attrs if attr != obj]
    
        # Générer la chaîne et la dessiner avec utils.drawGraph
        graph_string = "".join(edges)
        return utils.drawGraph(graph_string)

#####
# QUESTION 6.1. ON DOIT LA FAIRE
#####
# ...

#####
# QUESTION 6.2. Représentation graphique
#####
def mapClassifiers(dic, df): 
    """
    Trace une représentation graphique des classificateurs en termes de précision et rappel.

    Cette fonction calcule les métriques de précision et de rappel pour chaque classificateur
    contenu dans le dictionnaire fourni, puis affiche ces métriques sur un graphique avec des
    limites calculées dynamiquement.

    Parameters
    ----------
    dic : dict
        Un dictionnaire où les clés sont des noms de classificateurs (str) et les valeurs sont des
        instances de classificateurs capables de fournir des statistiques.
    df : pandas.DataFrame
        Le dataframe contenant les données sur lesquelles évaluer les classificateurs.

    Returns
    -------
    None
        Cette fonction n'a pas de retour mais affiche un graphique représentant les performances
        des classificateurs.
    """
    # Initialisation des listes pour stocker les résultats
    results = []
    precisions = []
    recalls = []
    
    # Parcourir chaque classificateur dans le dictionnaire
    for name, classifier in dic.items():
        # Obtenir les statistiques de précision et de rappel
        stats = classifier.statsOnDF(df)  # On suppose que statsOnDF retourne un dictionnaire avec 'Précision' et 'Rappel'
        precision = stats['Précision']
        recall = stats['Rappel']
        # Ajouter les résultats au tableau
        results.append((name, precision, recall))
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculer les limites des axes avec une marge
    x_min = min(precisions) - 0.02  # Limite minimale pour l'axe X (avec une marge de 0.02)
    x_max = max(precisions) + 0.02  # Limite maximale pour l'axe X (avec une marge de 0.02)
    y_min = min(recalls) - 0.02  # Limite minimale pour l'axe Y (avec une marge de 0.02)
    y_max = max(recalls) + 0.02  # Limite maximale pour l'axe Y (avec une marge de 0.02)

    # Configurer la figure avec constrained_layout pour un ajustement automatique
    plt.figure(figsize=(6, 6), facecolor='lightgray', constrained_layout=True)  # Fond gris clair avec ajustement

    # Tracer les points pour chaque classificateur
    for name, precision, recall in results:
        plt.scatter(precision, recall, color='red', marker='x', s=70)  # Points en forme de croix
        plt.text(precision + 0.003, recall - 0.003, name, fontsize=9, color='black')  # Étiquettes près des points

    # Ajouter des lignes de référence pour les axes
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.axvline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # Configurer les titres et étiquettes
    plt.title("Evaluation des Classifieurs (Précision vs. Rappel)", fontsize=14, pad=10)
    plt.xlabel("Précision", fontsize=12, labelpad=5)
    plt.ylabel("Rappel", fontsize=12, labelpad=5)

    # Définir les limites dynamiques pour les axes
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Personnaliser les bordures ("spines")
    ax = plt.gca()  # Obtenir l'objet des axes courants
    ax.spines['top'].set_linewidth(1)  # Épaisseur de la bordure supérieure
    ax.spines['right'].set_linewidth(1)  # Épaisseur de la bordure droite
    ax.spines['bottom'].set_linewidth(1)  # Épaisseur de la bordure inférieure
    ax.spines['left'].set_linewidth(1)  # Épaisseur de la bordure gauche
    ax.spines['top'].set_color('black')  # Couleur de la bordure supérieure
    ax.spines['right'].set_color('black')  # Couleur de la bordure droite
    ax.spines['bottom'].set_color('black')  # Couleur de la bordure inférieure
    ax.spines['left'].set_color('black')  # Couleur de la bordure gauche

    # Désactiver la grille
    plt.grid(False)

    # Afficher le graphique
    plt.show()


#####
# QUESTION 6.3. ON DOIT LA FAIRE
#####
# ...