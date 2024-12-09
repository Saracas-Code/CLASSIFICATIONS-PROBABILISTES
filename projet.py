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
        super().__init__(dataframe)  
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
        super().__init__(dataframe)  
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
        value = individual[self.attr_column]
        target_probs = {}

        for target in self.table_P2Dp[value]:
            target_probs[target] = self.table_P2Dp[value][target]

        return max(target_probs, key=lambda t: (target_probs[t], -t))  
    
#####
# QUESTION 2.4 : Comparaison
##### 
# Avant d’exprimer notre opinion, nous tenons à souligner que les meilleures mesures de prédiction peuvent varier en fonction du contexte dans lequel elles sont utilisées. Il est donc essentiel d’évaluer chaque cas avec soin afin de parvenir à une conclusion adaptée. 
#
# Dans ce problème, l’objectif est de prédire si une personne est malade (target = 1) ou en bonne santé (target = 0) en utilisant des attributs individuels comme l’âge, le sexe, et d’autres caractéristiques. 
#
# Parmi les classifieurs développés, le classifieur a priori (APrioriClassifier) est le plus simple, car il prédit toujours la classe majoritaire du jeu de données. Sa précision dépend uniquement de la proportion de la classe majoritaire dans l’échantillon, ce qui serait acceptable s’il existait une nette majorité dans les classes. 
#
# Cependant, ce n’est pas le cas ici, car l’équilibre entre les classes n’est pas suffisamment marqué, ce qui limite considérablement son utilité. En revanche, le classifieur par maximum de vraisemblance (ML2DClassifier) améliore significativement 
#les performances en prenant en compte les probabilités conditionnelles P(attr∣target), atteignant une bonne précision dans notre contexte. Toutefois, il ne tient pas compte des proportions globales des classes, ce qui peut poser problème dans des jeux de données déséquilibrés. 
#
# Enfin, le classifieur par maximum à posteriori (MAP2DClassifier) combine les probabilités conditionnelles avec les proportions globales des classes, ce qui lui permet de mieux gérer des scénarios réels où les classes sont souvent déséquilibrées. Bien qu’il puisse avoir une précision légèrement inférieure à celle du classifieur précédent, nous estimons qu’il est globalement le plus robuste et fiable pour ce type de problème. 
#
# Pour ces raisons, nous recommandons le MAP2DClassifier comme la meilleure option dans ce cas.
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
    
    valeurs_uniques = [len(P2D_p(df, attr)) for attr in attrs]
    total_combinations = np.prod(valeurs_uniques)
    
    memoire = int(total_combinations * 8)  # En octets

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

    if memoire > 1024:
        print(f"{len(attrs)} variable(s) : {memoire}o = {_format_memory(memoire)}")
    else:
        print(f"{len(attrs)} variable(s) : {memoire}o")

    return memoire

#####
# QUESTION 3.3.a. Montrer que P(A,B,C)=P(A)*P(B|A)*P(C|B)
#####
# P(A,B,C) = P(A) * P(B,C|A) = P(A) * P(B|A) * P(C|B,A) => Si C et A sont indépendantes => P(A) * P(B|A) * P(C|B)
#####

#####
# QUESTION 3.3.b. Si les 3 variables A, B et C ont 5 valeurs, quelle est la taille mémoire en octet nécessaire pour représenter cette distribution avec et sans l'utilisation de l'indépendance conditionnelle ?
#####
# Cas 1 : Sans indépendance conditionnelle
# ----------------------------------------
# Dans ce cas, toutes les combinaisons possibles de A, B et C doivent être stockées.
# Le nombre total de combinaisons se calcule ainsi :
#   nbCombinaisons = |A| * |B| * |C| = 5 * 5 * 5 = 125
#
# Chaque combinaison occupe 8 octets en mémoire, donc la mémoire totale nécessaire est :
#   MemTotale = 125 * 8 = 1000 octets
#

# Cas 2 : Avec indépendance conditionnelle partielle
# ---------------------------------------------------
# Sous l'hypothèse d'indépendance conditionnelle, nous avons :
#   P(A, B, C) = P(A) * P(B|A) * P(C|B)
#
# La mémoire nécessaire est calculée en représentant chaque composante séparément :
#
#  1. P(A) :
#     - Nous devons stocker les probabilités associées à A (5 valeurs).
#     - Mémoire requise : MemA = 5 * 8 = 40 octets
#
#  2. P(B|A) :
#     - Pour chaque valeur de A (5 valeurs), nous stockons les probabilités de B (5 valeurs).
#     - Mémoire requise : MemBA = 5 * 5 * 8 = 200 octets
#
#  3. P(C|B) :
#     - Pour chaque valeur de B (5 valeurs), nous stockons les probabilités de C (5 valeurs).
#     - Mémoire requise : MemCB = 5 * 5 * 8 = 200 octets
#
# Ainsi, la mémoire totale nécessaire dans ce cas est :
#   MemTotale = MemA + MemBA + MemCB = 40 + 200 + 200 = 440 octets
#
# Comparaison :
#   - Sans indépendance conditionnelle : 1000 octets
#   - Avec indépendance conditionnelle partielle : 440 octets
#
# Conclusion : L'indépendance conditionnelle permet donc une réduction significative de la mémoire nécessaire.
#####

#####
# QUESTION 4.1. Exemples 
#####
#Cas 1
#-------------------------------------------------------------
#Si les 5 variables sont totalement indépendantes, cela signifie que :
#P(A, B, C, D, E) = P(A) ⋅ P(B) ⋅ P(C) ⋅ P(D) ⋅ P(E)
#
#Cela pourrait être représenté par un graphe où aucun sommet n'a de parent.
#Autrement dit, le graphe résultant serait un graphe sans arêtes (graphe nul ou vide).
#util.drawGraph("A;B;C;D;E")
#
#Cas 2
#-------------------------------------------------------------
#Si toutes les variables doivent être dépendantes les unes des autres, il suffirait de réaliser un graphe contenant un cycle. 
#Ce cycle montrerait la dépendance entre toutes les variables.
#utils.drawGraph("A->B;B->C;C->D;D->E;E->A")
#####

#####
# QUESTION 4.2. Naïve Bayes 
#####
#Cas 1 : Décomposition de la vraisemblance
#-----------------------------------------------------------------------------
#P(attr1, attr2, attr3, ..., attrN | target) = P(attr1 | target) * P(attr2 | target) * P(attr3 | target) * ... * P(attrN | target)
#
#Grâce à l'hypothèse d'indépendance conditionnelle, la probabilité conjointe des attributs conditionnée à la variable cible 
#est simplifiée en un produit des probabilités conditionnelles de chaque attribut.
#
#Cas 2 : Décomposition de la distribution à posteriori
#-----------------------------------------------------------------------------
#La probabilité a posteriori est calculée grâce au théorème de Bayes, qui s'écrit comme suit :
#P(target | attr1, attr2, ..., attrN) = P(attr1, attr2, ..., attrN | target) * P(target) / P(attr1, attr2, ..., attrN)
#
#En utilisant l'hypothèse de Naïve Bayes (indépendance conditionnelle des attributs), la vraisemblance peut être réécrite comme : 
#P(attr1, attr2, ..., attrN | target) = P(attr1 | target) * P(attr2 | target) * ... * P(attrN | target)
#
#Ainsi, la probabilité a posteriori devient :
#P(target | attr1, attr2, ..., attrN) ∝ P(target) * P(attr1 | target) * P(attr2 | target) * ... * P(attrN | target)
#####

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
    attrs = [attr for attr in df.columns if attr != obj]  
    graph = ";".join([f"{obj}->{attr}" for attr in attrs])  

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
        memoire_totale = 16  
    else:
        memoire_totale = 0  
        for attr in attrs:
            nb_valeurs_target = df[obj].nunique()  
            nb_valeurs_attr = df[attr].nunique()   
            memoire_table = nb_valeurs_target * nb_valeurs_attr
            memoire_totale += memoire_table
        
        # Conversion en octets et ajustement pour éviter de doubler P(target)
        memoire_totale = memoire_totale * 8 - 16

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
        super().__init__(dataframe)  
        self.attrs = [col for col in dataframe.columns if col != 'target'] 
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
        result1 = 1  # Pour target=0
        result2 = 1  # Pour target=1

        for attr in self.attrs:
            attr_value = x.get(attr, None) 

            # Calculer pour target=0
            if attr_value in self.cond_probs[attr][0]:
                result1 *= self.cond_probs[attr][0][attr_value]
            else:
                result1 *= 0  # Probabilité nulle si la valeur n'est pas présente

            if attr_value in self.cond_probs[attr][1]:
                result2 *= self.cond_probs[attr][1][attr_value]
            else:
                result2 *= 0  

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
        return max(probas, key=probas.get)  

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
        super().__init__(dataframe)  
        self.df = dataframe
        self.attrs = [col for col in dataframe.columns if col != 'target']  

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

        p = self.prior_probs.get(1, 0)  # P(target=1)
        s = result1 * (1 - p) + result2 * p  

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
        return max(probas, key=probas.get)  

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
        super().__init__(dataframe)  
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
    
        graph_string = "".join(edges)
        return utils.drawGraph(graph_string)

#####
# QUESTION 6.1. Où se trouve le point idéal ?
#####
# Le point idéal se situe à (1,1).
# Une Précision P = 1 signifie qu'il n'y a pas de faux positifs, et un rappel R = 1 signifie qu'il n'y a pas de faux négatifs, donc le classifieur détecte tous les cas positifs sans aucune erreur.
#
# Pour comparer les classifieurs entre eux, on peut calculer les distances horizontales et verticales du point (Précision, Rappel) de chaque classifieur par rapport au point idéal (1,1). Plus un classifieur est proche de ce point idéal, meilleure est sa performance globale en termes de compromis entre Précision et Rappel.
#
# Cependant, il est souvent très difficile d'obtenir un classifieur parfaitement équilibré, car améliorer la Précision peut parfois entraîner une diminution du Rappel, et inversement. Ainsi, en fonction du problème spécifique, il peut être judicieux de sacrifier légèrement la Précision pour améliorer le Rappel, par exemple dans des contextes où il est crucial de minimiser les faux négatifs (comme en médecine pour détecter des maladies).
#
# De même, on peut privilégier la Précision si les faux positifs ont un coût élevé (comme dans la détection de fraudes). Ce compromis dépend donc des priorités et des implications des erreurs dans le contexte d'application.

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
    results = []
    precisions = []
    recalls = []
    
    for name, classifier in dic.items():
        # Obtenir les statistiques de précision et de rappel
        stats = classifier.statsOnDF(df)
        precision = stats['Précision']
        recall = stats['Rappel']
        results.append((name, precision, recall))
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculer les limites des axes avec une marge
    x_min = min(precisions) - 0.02  
    x_max = max(precisions) + 0.02  
    y_min = min(recalls) - 0.02  
    y_max = max(recalls) + 0.02  

    plt.figure(figsize=(6, 6), facecolor='lightgray', constrained_layout=True)  
    # Tracer les points pour chaque classificateur
    for name, precision, recall in results:
        plt.scatter(precision, recall, color='red', marker='x', s=70)  
        plt.text(precision + 0.003, recall - 0.003, name, fontsize=9, color='black')  

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
    ax.spines['top'].set_linewidth(1) 
    ax.spines['right'].set_linewidth(1)  
    ax.spines['bottom'].set_linewidth(1)  
    ax.spines['left'].set_linewidth(1) 
    ax.spines['top'].set_color('black')  
    ax.spines['right'].set_color('black')  
    ax.spines['bottom'].set_color('black') 
    ax.spines['left'].set_color('black')  

    plt.grid(False)
    plt.show()

#####
# QUESTION 6.3. Conclusion
#####
# À partir des graphiques obtenus, nous pouvons observer que les classifieurs ont été entraînés exclusivement avec le fichier train.csv, mais leur performance a été évaluée à la fois sur les données d'entraînement (train.csv) et sur les données de test (test.csv). Cela nous permet d'évaluer leur capacité de généralisation.
#
# Dans le premier graphique, où les mêmes données d'entraînement sont utilisées, les classifieurs affichent une performance élevée, ce qui est attendu puisqu'ils ont été optimisés spécifiquement pour ces données. Cependant, dans le second graphique, qui évalue les classifieurs sur un nouveau jeu de données (test.csv), leurs métriques, en particulier le rappel, diminuent de manière significative, ce qui est logique car ils sont confrontés à des données inédites.
#
# Il est important de noter que les classifieurs Naive Bayes sont ceux qui subissent la plus forte chute de performance. Cela peut s'expliquer par leur forte dépendance à l'hypothèse d'indépendance conditionnelle, qui est rarement respectée dans les ensembles de données réels. Ce comportement suggère que les Naive Bayes ont des  difficultés à généraliser sur ce problème, probablement à cause des dépendances entre les variables.
#
# En général, cette différence de performance entre train et test indique que certains modèles peuvent être surajustés aux données d'entraînement, tandis que d'autres ne sont peut-être pas adaptés pour capturer la complexité du problème. Il serait recommandé d’explorer des classifieurs plus robustes et d’améliorer la validation croisée afin d’évaluer plus précisément leur capacité de généralisation.

####################
# QUESTION 7 BONUS #
####################

#####
# QUESTION 7.1 : Calcul des informations mutuelles
#####
def MutualInformation(df, x, y):
    """
    Calcule l'information mutuelle entre deux variables x et y dans le DataFrame df
    en utilisant la formule donnée avec log base 2.

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données.
    x : str
        Le nom de la colonne représentant la variable X.
    y : str
        Le nom de la colonne représentant la variable Y.

    Returns
    -------
    float
        L'information mutuelle entre X et Y.
    """
    # Ici on fait le calcul pour les probabilités conjointes P(x, y)
    joint_probs = df.groupby([x, y]).size() / len(df)
    
    # Maintenant, les probabilités marginales P(x) et P(y)
    px = df[x].value_counts(normalize=True)
    py = df[y].value_counts(normalize=True)

    # Calcul de l'information mutuelle
    mutual_info = 0.0
    for (x_val, y_val), p_xy in joint_probs.items():
        p_x = px[x_val]
        p_y = py[y_val]
        if p_xy > 0:  # Ça c'est pour éviter log(0)
            mutual_info += p_xy * np.log2(p_xy / (p_x * p_y))
    
    return mutual_info

def ConditionalMutualInformation(df, x, y, z):
    """
    Calcule l'information mutuelle conditionnelle entre x et y, conditionnée à z, dans le DataFrame df, selon la formule donnée.

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données.
    x : str
        Le nom de la colonne représentant la variable X.
    y : str
        Le nom de la colonne représentant la variable Y.
    z : str
        Le nom de la colonne représentant la variable Z (condition).

    Returns
    -------
    float
        L'information mutuelle conditionnelle entre X et Y, conditionnée à Z.
    """
    # Probabilités conjointes P(x, y, z)
    joint_xyz = df.groupby([x, y, z]).size() / len(df)
    
    # Probabilités conjointes P(x, z) et P(y, z)
    joint_xz = df.groupby([x, z]).size() / len(df)
    joint_yz = df.groupby([y, z]).size() / len(df)
    
    # Probabilité marginale P(z)
    marginal_z = df[z].value_counts(normalize=True)

    # Calcul de l'information mutuelle conditionnelle
    cmi = 0.0
    for (x_val, y_val, z_val), p_xyz in joint_xyz.items():
        p_xz = joint_xz[(x_val, z_val)]
        p_yz = joint_yz[(y_val, z_val)]
        p_z = marginal_z[z_val]

        if p_xyz > 0 and p_xz > 0 and p_yz > 0 and p_z > 0:  # Eviter les divisions par zéro
            cmi += p_xyz * np.log2((p_z * p_xyz) / (p_xz * p_yz))
    
    return cmi

#####
# QUESTION 7.2 : Calcul de la matrice des poids
#####
def MeanForSymetricWeights(a):
    """
    Calcule la moyenne des poids pour une matrice symétrique avec une diagonale nulle.

    Parameters
    ----------
    a : numpy.ndarray
        Matrice symétrique avec une diagonale nulle.

    Returns
    -------
    float
        La moyenne des poids non nuls de la matrice.
    """
    # Vérifier si la matrice est symétrique
    if not np.allclose(a, a.T):
        raise ValueError("La matrice n'est pas symétrique.")
    
    # Extraire les éléments hors de la diagonale
    mask = ~np.eye(a.shape[0], dtype=bool)  # Mask pour exclure la diagonale
    non_diagonal_elements = a[mask]
    
    # Calculer la moyenne
    mean_weight = non_diagonal_elements.mean()
    return float(mean_weight)

def SimplifyConditionalMutualInformationMatrix(a):
    """
    Simplifie une matrice symétrique de diagonale nulle en annulant
    toutes les valeurs plus petites que la moyenne.

    Parameters
    ----------
    a : numpy.ndarray
        Matrice symétrique avec une diagonale nulle.

    Returns
    -------
    numpy.ndarray
        Matrice simplifiée où les poids inférieurs à la moyenne sont annulés.
    """
    # Calculer la moyenne des poids
    mean_weight = MeanForSymetricWeights(a)
    
    # Utiliser une masque pour annuler les valeurs inférieures à la moyenne
    a[a < mean_weight] = 0.0

    return a

#####
# QUESTION 7.3 : Arbre (forêt) optimal entre les attributs
#####
def Kruskal(df, a):
    """
    Implémente l'algorithme de Kruskal pour trouver l'arbre de poids maximal, avec un seuil de poids pour éliminer les arêtes faibles (< 0.25).

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les colonnes correspondant aux attributs.
    a : numpy.ndarray
        Matrice symétrique de poids, avec une diagonale nulle.

    Returns
    -------
    list
        Liste des arcs (attr1, attr2, poids) dans l'arbre de poids maximal,
        avec les poids >= 0.25.
    """

    # Vérifier que la matrice est symétrique
    if not np.allclose(a, a.T):
        raise ValueError("La matrice n'est pas symétrique.")
    
    # Obtenir les noms des attributs
    attributes = list(df.keys())

    ######## ALGORITHME DE KRUSKAL ########
    # Étape 1 : Extraire tous les bords (i, j, poids) de la matrice
    edges = []
    n = a.shape[0]
    for i in range(n):
        for j in range(i + 1, n):  # i < j pour éviter les doublons (symétrie)
            if a[i, j] > 0:  # On ignore les poids nuls
                edges.append((i, j, a[i, j]))

    # Étape 2 : Trier les bords par poids décroissant
    edges.sort(key=lambda x: x[2], reverse=True)

    # Union-Find pour détecter les cycles
    parent = list(range(n))
    rank = [0] * n

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])  # Compression de chemin
        return parent[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)
        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            elif rank[root1] < rank[root2]:
                parent[root1] = root2
            else:
                parent[root2] = root1
                rank[root1] += 1

    # Étape 3 : Construire l'arbre de poids maximal
    mst = []
    for i, j, weight in edges:
        if find(i) != find(j):  # PAS DE CYCLE
            mst.append((attributes[i], attributes[j], weight))
            union(i, j)

    # Étape 4 : Filtrer les arêtes avec un poids < 0.25. C'est à dire, on ne stocke que les poids plus significants (attributs dépendants) 
    threshold = 0.25
    mst_filtered = [(attr1, attr2, weight) for attr1, attr2, weight in mst if weight >= threshold]

    return mst_filtered

#####
# QUESTION 7.4 : Orientation des arcs entre attributs
#####
def ConnexSets(list_arcs):
    """
    Crée une liste des ensembles d'attributs connectés à partir d'une liste d'arcs.
    
    Parameters
    ----------
    list_arcs : list of tuple
        Liste d'arcs où chaque arc est représenté par un tuple (attribut1, attribut2, poids).
        
    Returns
    -------
    list of set
        Liste d'ensembles d'attributs connectés.
    """
    connex_sets = []

    for arc in list_arcs:
        a, b, _ = arc  # Extraire les attributs a et b
        set_a, set_b = None, None

        # Vérifier si a ou b est déjà dans un ensemble existant
        for s in connex_sets:
            if a in s:
                set_a = s
            if b in s:
                set_b = s

        if set_a and set_b:
            if set_a != set_b:
                # Fusionner les ensembles s'ils sont distincts
                set_a.update(set_b)
                connex_sets.remove(set_b)
        elif set_a:
            # Ajouter b à l'ensemble contenant a
            set_a.add(b)
        elif set_b:
            # Ajouter a à l'ensemble contenant b
            set_b.add(a)
        else:
            # Créer un nouveau ensemble avec a et b
            connex_sets.append(set([a, b]))

    return connex_sets

def OrientConnexSets(df, arcs, target_class):
    """
    Oriente les ensembles connexes en déterminant une racine pour chaque ensemble
    à l'aide de l'information mutuelle avec la classe cible.

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données.
    arcs : list of tuple
        Liste des arcs non orientés sous la forme (attribut1, attribut2, poids).
    target_class : str
        Le nom de la colonne représentant la classe cible.

    Returns
    -------
    list of tuple
        Liste des arcs orientés sous la forme (parent, enfant).
    """
    # Obtenir les ensembles connexes
    connex_sets = ConnexSets(arcs)
    oriented_arcs = []

    for connex_set in connex_sets:
        # Calculer l'information mutuelle entre chaque attribut de l'ensemble et la classe cible
        mutual_info = {attr: MutualInformation(df, target_class, attr) for attr in connex_set}

        # Trouver la racine (l'attribut avec la plus haute information mutuelle)
        root = max(mutual_info, key=mutual_info.get)

        # Construire un graphe à partir des arcs pour respecter les connexions
        graph = {attr: [] for attr in connex_set}
        for a, b, _ in arcs:
            if a in connex_set and b in connex_set:
                graph[a].append(b)
                graph[b].append(a)

        # Orienter les arcs en priorisant l'ordre des arcs donnés
        visited = set()
        stack = [root]

        while stack:
            parent = stack.pop()
            visited.add(parent)

            # Prioriser les enfants selon l'ordre dans la liste originale des arcs
            children = sorted(
                [child for child in graph[parent] if child not in visited],
                key=lambda x: next(
                    (i for i, arc in enumerate(arcs) if (arc[0] == parent and arc[1] == x) or (arc[0] == x and arc[1] == parent)),
                    float('inf')
                )
            )

            for child in children:
                oriented_arcs.append((parent, child))
                stack.append(child)

    return oriented_arcs

#####
# QUESTION 7.5 : Classifieur TAN
#####
class MAPTANClassifier(APrioriClassifier):
    """
    Classificateur basé sur le modèle Tree Augmented Naive Bayes (TAN).
    Hérite de la classe APrioriClassifier.

    Methods
    -------
    __init__(dataframe):
        Initialise le classificateur MAPTANClassifier à partir d'un DataFrame donné.
    """

    def __init__(self, dataframe):
        """
        Initialise la structure TAN en calculant l'information mutuelle conditionnelle entre les attributs, en construisant un arbre maximum et en orientant les connexions.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Le DataFrame contenant les données d'apprentissage, avec une colonne 'target'
            représentant la variable cible.

        Attributes
        ----------
        cmis : numpy.ndarray
            Une matrice 2D contenant l'information mutuelle conditionnelle pour chaque paire
            d'attributs donnée la variable cible.
        liste_arcs : list of tuple
            Une liste d'arcs non orientés représentant les connexions TAN maximales.
        oriented_connex_sets : list of tuple
            Une liste des arcs orientés résultant de l'arbre TAN construit.

        """
        super().__init__(dataframe)  
        self.dataframe = dataframe

        # Calculer la matrice d'information mutuelle conditionnelle (CMI)
        cmis = np.array([[0 if x == y else ConditionalMutualInformation(dataframe, x, y, "target") for x in dataframe.keys() if x != "target"] for y in dataframe.keys() if y != "target"])

        self.cmis = SimplifyConditionalMutualInformationMatrix(cmis)

        # Construire un arbre maximum avec Kruskal
        self.liste_arcs = Kruskal(dataframe, cmis)

        # Orienter les connexions dans l'arbre
        self.oriented_connex_sets = OrientConnexSets(dataframe, self.liste_arcs, 'target')

    
    def draw(self) : 
        """
        Dessine un graphe représentant la structure TAN.
        1. Connecte 'target' à tous les autres attributs.
        2. Ajoute les arcs orientés de oriented_connex_sets.
        """
        # Initialiser les connexions avec 'target'
        arcs = []
        attributes = [col for col in self.dataframe.columns if col != "target"]

        # Ajouter les connexions de 'target' vers chaque attribut
        for attr in attributes:
            arcs.append(f"target->{attr}")

        # Ajouter les connexions de oriented_connex_sets
        for parent, child in self.oriented_connex_sets:
            arcs.append(f"{parent}->{child}")

        # Formater les arcs en chaîne
        arcs_str = "; ".join(arcs)

        # Dessiner le graphe
        return utils.drawGraph(arcs_str)

    def estimProbas(self, x):
        """
        Estime les probabilités de chaque classe (0 ou 1) pour un individu donné.

        Parameters
        ----------
        x : dict
            Un dictionnaire représentant un individu (attr1: val1, attr2: val2, ...).

        Returns
        -------
        dict
            Un dictionnaire contenant les probabilités pour chaque classe cible (0 et 1).
        """
        # Initialiser avec les probabilités a priori
        result = {0: 1 - self.prior['estimation'], 1: self.prior['estimation']}

        # Obtenir tous les attributs du DataFrame sauf la colonne 'target'
        all_attributes = set(self.dataframe.columns) - {'target'}

        # Suivre les attributs déjà traités dans oriented_connex_sets
        processed_children = set()

        # Ajuster les probabilités avec les connexions TAN
        for parent, child in self.oriented_connex_sets:
            processed_children.add(child)
            for target_value in [0, 1]:
                parent_value = x[parent]
                child_value = x[child]

                # Filtrer les données pour calculer P(child | parent, target)
                subset = self.dataframe[
                    (self.dataframe[parent] == parent_value) & (self.dataframe['target'] == target_value)
                ]
                parent_target_subset = self.dataframe[
                    (self.dataframe['target'] == target_value) & (self.dataframe[parent] == parent_value)
                ]

                # Appliquer le lissage de Laplace
                unique_child_values = len(self.dataframe[child].unique())
                conditional_proba = (
                    subset[subset[child] == child_value].shape[0] + 1
                ) / (
                    parent_target_subset.shape[0] + unique_child_values
                )

                # Multiplier les probabilités accumulées
                result[target_value] *= conditional_proba

        # Calculer P(attr | target) pour les attributs non traités
        for attr in all_attributes - processed_children:
            for target_value in [0, 1]:
                attr_value = x[attr]

                # Récupérer le nombre de valeurs uniques pour l'attribut
                unique_attr_values = len(self.dataframe[attr].unique())

                # Calculer la probabilité avec le lissage de Laplace
                subset_target = self.dataframe[self.dataframe['target'] == target_value]
                count = len(subset_target[subset_target[attr] == attr_value])
                total = len(subset_target)

                conditional_proba = (count + 1) / (total + unique_attr_values)

                # Multiplier les probabilités accumulées
                result[target_value] *= conditional_proba

        # Normaliser les probabilités pour qu'elles totalisent 1
        total = result[0] + result[1]
        if total > 0:
            result[0] /= total
            result[1] /= total
        else:
            result = {0: 0.5, 1: 0.5}

        return result


    def estimClass(self, x):
        """
        Estime la classe (0 ou 1) avec la probabilité la plus élevée pour un individu donné.

        Parameters
        ----------
        x : dict
            Une ligne du DataFrame représentée sous forme de dictionnaire.

        Returns
        -------
        int
            La classe estimée (0 ou 1).
        """
        probas = self.estimProbas(x)
        return max(probas, key=probas.get)  # Retourne la clé avec la proba maximale

#####
# QUESTION 8 : CONCLUSION FINALE
#####
# L’analyse des performances des classifieurs bayésiens met en évidence des forces et des faiblesses propres à chaque méthode. Le MAPTANClassifier, élaboré dans la question bonus, se distingue par sa capacité à capturer les dépendances conditionnelles entre attributs grâce à sa structure TAN. Ses performances sont excellentes, avec une précision et un rappel proches de 1 sur les ensembles d’entraînement et de test. Toutefois, cette performance a un coût : le MAPTANClassifier est plus lourd à calculer et nécessite des ressources supplémentaires, ce qui peut poser des défis pour des bases de données de très grande taille.
#
# À l’inverse, les classifieurs ML2DClassifier et MAP2DClassifier, bien que plus simples, offrent des résultats intéressants. Si leurs performances sont inférieures à celles du MAPTANClassifier, ils se défendent mieux dans d’autres jeux de données où les dépendances entre attributs sont moins significatives. Ces modèles, rapides à entraîner et à tester, se révèlent adaptés pour des applications nécessitant des solutions légères et efficaces.
#
# Les modèles bayésiens naïfs (MLNaiveBayesClassifier et MAPNaiveBayesClassifier) affichent une bonne précision, mais leur rappel plus faible sur l’ensemble de test souligne les limites de leur hypothèse simpliste d’indépendance entre attributs, les empêchant de gérer des relations complexes. Les versions réduites (ReducedMAPNaiveBayesClassifier) constituent un compromis équilibré entre généralisation et simplicité.
# 
# En conclusion, le MAPTANClassifier est le plus performant de cette sélection de classifieurs, mais également le plus coûteux en ressources. Les classifieurs plus simples, tels que le ML2DClassifier et le MAP2DClassifier, restent des alternatives viables selon les contextes, en particulier pour des situations nécessitant un compromis entre complexité et efficacité. Le choix final doit toujours être guidé par les spécificités des données et les contraintes opérationnelles.