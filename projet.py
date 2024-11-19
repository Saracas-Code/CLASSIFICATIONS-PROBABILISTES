# Mattéo PIETRI
# Sara CASTRO LÓPEZ

from statsmodels.stats.proportion import proportion_confint
import utils

##### 1. CLASSIFICATION À PRIORI #####
##### 
# Question 1.1. Calcul de la probabilité à priori
#####
def getPrior(df) :
    '''
    Calculer la probabilité de qu'une personne soit de la classe 1 (target=1) avec un intervalle de confiance du 95%

    Parameters
    ---------
        attrs: train (.csv)
            la base de données d'où obtiendra l'information
    
    Returns
    -------
        dictionnaire contenant 3 clés : 'estimation', 'min5pourcent', 'max5pourcent'
    '''

    # Compter le nombre total d'observations et celles qui sont target=1
    total = len(df)
    class_1_count = df["target"].sum()
    
    # Calculer la proportion
    proportion = class_1_count / total
    
    # Calculer l'intervale de confiance
    lower, upper = proportion_confint(class_1_count, total, alpha=0.05, method='wilson')

    # Retourner le dictionnaire demandé
    return { "estimation" : float(proportion), "min5pourcent" : float(lower), "max5pourcent" : float(upper) }

##### 
# Question 1.2. Programmation orienté objet dans la hiérarchie des Classifiers
#####
class APrioriClassifier(utils.AbstractClassifier) :
    
    def __init__(self, df):
        """
        Inicializa el clasificador con el DataFrame de entrenamiento.
        Calcula la clase mayoritaria.
        """
        self.majority_class = df["target"].mode()[0]  # Clase mayoritaria

    def estimClass(self, dic):
        """
        Predice siempre la clase mayoritaria.
        """
        return self.majority_class

    def statsOnDF(self, df):
        """
        Calcula estadísticas de calidad del clasificador.
        """
        total = len(df)
        correct = (df["target"] == self.majority_class).sum()
        accuracy = correct / total

        return {"accuracy": accuracy, "correct": correct, "total": total}


##### 2. CLASSIFICATION À PRIORI #####
##### 
# QUESTION 2.1
#####

def P2D_l(df, attr) :
    dictionnaire = {}
    
    






