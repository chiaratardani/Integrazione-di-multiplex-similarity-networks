# Integrazione di network di similarità multiplex
## 💻 Autore 
**Chiara Tardani**
## 🚀 Obiettivo
L'obiettivo delle funzioni implementate è quello di integrare le reti di similarità passate in input in un network (di similarità) unico. I metodi di aggregazione proposti sono stati presi da <https://github.com/DedeBac/SimilarityMatrixAggregation> per quanto riguarda il calcolo dei tre baricentri: 
```markdown
class WeightedMeanAggregator(SimilarityMatrixAggregator) # Aggregatore per la media aritmetica pesata di Frobenius
class GeometricAggregator(SimilarityMatrixAggregator) # Aggregatore per la media geometrica Riemanniana
class 
    
