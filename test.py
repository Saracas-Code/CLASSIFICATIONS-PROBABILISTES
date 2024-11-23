# Matteo PIETRI
# Sara CASTRO LÓPEZ

import pandas as pd
import utils
import projet

data = pd.read_csv("data/heart.csv")
data.head()
utils.viewData(data)
discretise = utils.discretizeData(data)
utils.viewData(discretise, kde=False)

print("Q1.1-----------------------------------------------------------------------------------------------------------")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
print(projet.getPrior(train))
print(projet.getPrior(test))

print("Q1.2.a---------------------------------------------------------------------------------------------------------")
cl = projet.APrioriClassifier(train)
print("Classe prédite pour n'importe quel individu :", cl.estimClass({}))

print("Q1.2.b---------------------------------------------------------------------------------------------------------")
cl = projet.APrioriClassifier(train)
print("test en apprentissage : {}".format(cl.statsOnDF(train)))
print("test en validation: {}".format(cl.statsOnDF(test)))

print("Q2.1.a---------------------------------------------------------------------------------------------------------")

p_thal_given_target = projet.P2D_l(train, 'thal')  # calcul P(thal=a|target=t)
print(p_thal_given_target)
print()
print(f"Dans la base train, la probabilité que thal=3 sachant que target=1 est {p_thal_given_target[1][3]}")

print("Q2.1.b---------------------------------------------------------------------------------------------------------")
p_target_given_thal = projet.P2D_p(train, 'thal')
print(p_target_given_thal)
print()
print(f"Dans la base train, la probabilité que target=1 sachant que thal=3 est {p_target_given_thal[3][1]}")

print("Q2.2---------------------------------------------------------------------------------------------------------")
cl=projet.ML2DClassifier(train,"thal") # cette ligne appelle projet.P2Dl(train,"thal")
for i in [0,1,2]:
    print("Estimation de la classe de l'individu {} par ML2DClassifier : {}".format(i,cl.estimClass(utils.getNthDict(train,i))))
print("test en apprentissage : {}".format(cl.statsOnDF(train)))
print("test en validation: {}".format(cl.statsOnDF(test)))

print("Q2.3---------------------------------------------------------------------------------------------------------")
cl=projet.MAP2DClassifier(train,"thal") # cette ligne appelle projet.P2Dp(train,"thal")
for i in [0,1,2]:
    print("Estimation de la classe de l'individu {} par MAP2DClasssifer) : {}".format(i,cl.estimClass(utils.getNthDict(train,i)))) 
print("test en apprentissage : {}".format(cl.statsOnDF(train)))
print("test en validation: {}".format(cl.statsOnDF(test)))
