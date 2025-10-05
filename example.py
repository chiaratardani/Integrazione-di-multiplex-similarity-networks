import numpy as np
from utils import *
from aggregation import *

'''Generiamo, per mostrare come usare il codice, tre matrici
di similarità (matrici completamente positive) random, di dimensione (3 x 3).'''
# Parametri
n = 3       # Dimensione della matrice (n x n)
k = 5       # Dimensione interna (numero di colonne di B)
num_matr = 3  # Numero di matrici da generare

matrices = []
for _ in range(num_matr):
    # Genera una matrice B (n x k) casuale con elementi non negativi.
    # np.random.rand, infatti, genera elementi random da una distribuzione
    # uniforme in [0,1].
    B = np.random.rand(n, k)  # Elementi in [0, 1]
    
    # Calcola la matrice completamente positiva A = B * B^T
    A = B @ B.T
    
    # Assicurati che gli elementi siano non negativi (per costruzione lo sono):
    # la condizione che vogliamo testare è np.all(A >= 0); viene, dunque, generata
    # una matrice booleana (n x n), in cui un elemento è True se quello 
    # corrispondente in A è >= 0, altrimenti è False. La stringa è il 
    # messaggio di errore che deve ritornare nel caso in cui la condizione non sia soddisfatta.
    assert np.all(A >= 0), "La matrice non è completamente positiva!" 
    matrices.append(A)

# Mostra le matrici generate
for i, A in enumerate(matrices):
    print(f"Matrice {i+1}:\n{A}\n")

'''Ora possiamo utilizzare la funzione 'aggregate'
per ottenere la media di queste matrici con il metodo di
aggregazione auspicato (inserito dall'utente).'''
snf_result = aggregate(matrices, 'snf')
frob_result = aggregate(matrices, 'weighted_mean')
riem_result = aggregate(matrices, 'geometric')
wass_result = aggregate(matrices, 'wasserstein')
