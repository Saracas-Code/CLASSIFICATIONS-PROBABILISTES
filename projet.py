# Matteo PIETRI
# Sara CASTRO LÓPEZ

import pandas as pd  # package for high-performance, easy-to-use data structures and data analysis
import numpy as np  # fundamental package for scientific computing with Python
import math
import os
import utils


#####
# Question 1.1 : calcul de la probabilité a priori
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

    print("{")
    print(f"  'estimation': {p},")
    print(f"  'min5pourcent': {borne_inf},")
    print(f"  'max5pourcent': {borne_sup}")
    print("}")

    return {
        'estimation': p,
        'min5pourcent': borne_inf,
        'max5pourcent': borne_sup
    }


#####
# Question 1.2.a : programmation orientée objet dans la hiérarchie des Classifier
#####
# ...
#####

class APrioriClassifier(utils.AbstractClassifier):
    """
    Classifieur a priori basé sur la classe majoritaire.
    Hérite de AbstractClassifier.
    """

    def __init__(self, dataframe):
        """
        Initialise le classifier en calculant la classe majoritaire à partir du DataFrame donné.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Le DataFrame contenant les données avec une colonne 'target'.
        """
        self.majority_class = dataframe['target'].mode()[0]  # Calcul de la classe majoritaire

    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1
        Parameters
        ----------
        attrs: Dict[str,value]
        le dictionnaire nom-valeur des attributs

        Returns
        -------
        la classe 0 ou 1 estimée
        """

        return self.majority_class

    #####
    # Question 1.2.b : évaluation de classifieurs
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

        # Parcourir les lignes du DataFrame pour confronter les classes réelles et estimées
        for _, ligne in dataset.iterrows():
            classe_reelle = ligne['target']
            classe_predit = self.estimClass({})

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
# 2.1.a - probabilités conditionelles
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
# 2.1.b - probabilités conditionelles
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
