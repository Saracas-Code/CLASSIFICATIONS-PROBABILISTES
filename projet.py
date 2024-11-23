# Matteo PIETRI
# Sara CASTRO LÓPEZ

import pandas as pd  # package for high-performance, easy-to-use data structures and data analysis
import numpy as np  # fundamental package for scientific computing with Python
import math
import os
import utils


#####
# QUESTION 1.1 : Calcul de la probabilité a priori
#####
# ...
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
        prior = getPrior(dataframe)
        self.majority_class = 1 if prior['estimation'] > 0.5 else 0  # Classe majoritaire

    #####
    # QUESTION 1.2.a : Programmation orientée objet dans la hiérarchie des Classifier
    #####
    # ...
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
    # ...
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
# ...
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
# ...
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
# ...
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
# ...
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
