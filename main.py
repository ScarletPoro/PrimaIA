#import moduli
import pandas as pd
import sklearn.tree as dtc

print("----------------")
print("La mia prima AI")

# gestisco errore csv
'''
try:
    giocatori = pd.read_csv('giocatori.csv')
    #print(giocatori)
except FileNotFoundError:
    print("Il file 'giocatori.csv' non è stato trovato.")
except pd.errors.EmptyDataError:
    print("Il file 'giocatori.csv' è vuoto o non contiene dati CSV validi.")
except pd.errors.ParserError:
    print("Errore durante la lettura del file CSV.")
'''
#imposto dati imput
x=giocatori.drop(columns=['videogame'])
#imposto dati export
y=giocatori['videogame']

#imposto decisionTreeClassifier
modello = dtc.DecisionTreeClassifier()
modello.fit(x.values,y.values)

#imposto dati da calcolare e stampo
previsione = modello.predict([[0,31],[2,35]])
print(previsione)