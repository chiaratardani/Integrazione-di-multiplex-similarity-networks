# Integrazione di network di similarità multiplex
## 💻 Autore 
**Chiara Tardani**
## 🚀 Obiettivo
L'obiettivo delle funzioni implementate è quello di integrare le reti di similarità, passate in input, in un network (di similarità) unico. I metodi di aggregazione proposti sono stati presi da <https://github.com/DedeBac/SimilarityMatrixAggregation> per quanto riguarda il calcolo dei tre baricentri (Similarity Matrix Average, SMA): 
```markdown
```python
class WeightedMeanAggregator(SimilarityMatrixAggregator) # Aggregatore per la media aritmetica pesata con Frobenius
class GeometricAggregator(SimilarityMatrixAggregator) # Aggregatore per la media geometrica Riemanniana
class WassersteinAggregator(SimilarityMatrixAggregator) # Aggregatore per la media di Wasserstein
```
e da <https://github.com/maxconway/SNFtool/tree/master>, da cui è stato preso il codice in R per il tool SNF (Similarity Network Fusion), è stato tradotto in linguaggio Python e trasformato a sua volta in una classe:
```markdown
```python
class SNFAggregator(SimilarityMatrixAggregator) #  Aggregatore che usa Similarity Network Fusion (SNF) 
```
L'obiettivo è infatti sfruttare la potenza di Python come paradigma di programmazione a oggetti, creando delle classi che rappresentino aggregatori di matrici di similarità, con l'idea di trasformare il codice in una libreria Python fruibile.
## 📚 Struttura
Il repository contiene i seguenti scripts .py:

- **aggregation.py** : che contiene le classi dei quattro aggregatori proposti;

- **utils.py** : che contiene delle funzioni che servono per quasi tutti i moduli.
## ✨ Caratteristiche
Il cuore del progetto è l'uso di una classe astratta, SimilarityMatrixAggregator:
```markdown
```python
 
```

