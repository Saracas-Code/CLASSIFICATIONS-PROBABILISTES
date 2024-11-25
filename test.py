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

print("Q3.1---------------------------------------------------------------------------------------------------------")
projet.nbParams(train,['target'])
projet.nbParams(train,['target','thal'])
projet.nbParams(train,['target','age'])
projet.nbParams(train,['target','age','thal','sex','exang'])
projet.nbParams(train,['target','age','thal','sex','exang','slope','ca','chol'])
projet.nbParams(train) # seul résultat visible en sortie de cellule

print("Q3.2---------------------------------------------------------------------------------------------------------")
projet.nbParamsIndep(train[['target']])
projet.nbParamsIndep(train[['target','thal']])
projet.nbParamsIndep(train[['target','age']])
projet.nbParamsIndep(train[['target','age','thal','sex','exang']])
projet.nbParamsIndep(train[['target','age','thal','sex','exang','slope','ca','chol']])
projet.nbParamsIndep(train) # seul résultat visible en sortie de cellule

print("Q4.3.a---------------------------------------------------------------------------------------------------------")
projet.drawNaiveBayes(train,"target")

print("Q4.3.b---------------------------------------------------------------------------------------------------------")
projet.nbParamsNaiveBayes(train,'target',[])
projet.nbParamsNaiveBayes(train,'target',['target','thal'])
projet.nbParamsNaiveBayes(train,'target',['target','age'])
projet.nbParamsNaiveBayes(train,'target',['target','age','thal','sex','exang'])
projet.nbParamsNaiveBayes(train,'target',['target','age','thal','sex','exang','slope','ca','chol'])
projet.nbParamsNaiveBayes(train,'target') # seul résultat visible en sortie de cellule

print("Q4.4---------------------------------------------------------------------------------------------------------")
cl=projet.MLNaiveBayesClassifier(train)

for i in [0,1,2]:
    print(f"Estimation de la proba de l'individu {i} par MLNaiveBayesClassifier : {cl.estimProbas(utils.getNthDict(train,i))}")
    print(f"Estimation de la classe de l'individu {i} par MLNaiveBayesClassifier : {cl.estimClass(utils.getNthDict(train,i))}") 
    print("------")
print(f"test en apprentissage : {cl.statsOnDF(train)}")
print(f"test en validation    : {cl.statsOnDF(test)}")

cl=projet.MAPNaiveBayesClassifier(train)

for i in [0,1,2]:
    print(f"Estimation de la proba de l'individu {i} par MLNaiveBayesClassifier : {cl.estimProbas(utils.getNthDict(train,i))}")
    print(f"Estimation de la classe de l'individu {i} par MLNaiveBayesClassifier : {cl.estimClass(utils.getNthDict(train,i))}") 
    print("------")
print(f"test en apprentissage : {cl.statsOnDF(train)}")
print(f"test en validation    : {cl.statsOnDF(test)}")

print("Q5.1---------------------------------------------------------------------------------------------------------")
for attr in train.keys():
    if attr!='target':
        print(f"target independant de {attr} ? {'YES' if projet.isIndepFromTarget(train,attr,0.01) else 'no'}")
    
print("Q5.2---------------------------------------------------------------------------------------------------------")
cl=projet.ReducedMLNaiveBayesClassifier(train,0.05)
cl.draw()

for i in [0,1,2]:
    print(f"Estimation de la proba de l'individu {i} par ReducedMLNaiveBayesClassifier : {cl.estimProbas(utils.getNthDict(train,i))}")
    print(f"Estimation de la classe de l'individu {i} par ReducedMLNaiveBayesClassifier : {cl.estimClass(utils.getNthDict(train,i))}") 
    print("------")
print(f"test en apprentissage : {cl.statsOnDF(train)}")
print(f"test en validation    : {cl.statsOnDF(test)}")

cl=projet.ReducedMLNaiveBayesClassifier(train,0.01)
cl.draw()

for i in [0,1,2]:
    print(f"Estimation de la proba de l'individu {i} par ReducedMLNaiveBayesClassifier : {cl.estimProbas(utils.getNthDict(train,i))}")
    print(f"Estimation de la classe de l'individu {i} par ReducedMLNaiveBayesClassifier : {cl.estimClass(utils.getNthDict(train,i))}") 
    print("------")
print(f"test en apprentissage : {cl.statsOnDF(train)}")
print(f"test en validation    : {cl.statsOnDF(test)}")

cl=projet.ReducedMAPNaiveBayesClassifier(train,0.01)
cl.draw()

for i in [0,1,2]:
    print(f"Estimation de la proba de l'individu {i} par ReducedMAPNaiveBayesClassifier : {cl.estimProbas(utils.getNthDict(train,i))}")
    print(f"Estimation de la classe de l'individu {i} par ReducedMAPNaiveBayesClassifier : {cl.estimClass(utils.getNthDict(train,i))}") 
    print("------")
print(f"test en apprentissage : {cl.statsOnDF(train)}")
print(f"test en validation    : {cl.statsOnDF(test)}")

print("Q6.2---------------------------------------------------------------------------------------------------------")
projet.mapClassifiers({"1":projet.APrioriClassifier(train),
                "2":projet.ML2DClassifier(train,"exang"),
                "3":projet.MAP2DClassifier(train,"exang"),
                "4":projet.MAPNaiveBayesClassifier(train),
                "5":projet.MLNaiveBayesClassifier(train),
                "6":projet.ReducedMAPNaiveBayesClassifier(train,0.01),
                "7":projet.ReducedMLNaiveBayesClassifier(train,0.01),
                },train)

projet.mapClassifiers({"1":projet.APrioriClassifier(train),
                "2":projet.ML2DClassifier(train,"exang"),
                "3":projet.MAP2DClassifier(train,"exang"),
                "4":projet.MAPNaiveBayesClassifier(train),
                "5":projet.MLNaiveBayesClassifier(train),
                "6":projet.ReducedMAPNaiveBayesClassifier(train,0.01),
                "7":projet.ReducedMLNaiveBayesClassifier(train,0.01),
                },test)