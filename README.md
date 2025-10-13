# Integrazione di network di similarità multiplex
## 💻 Autore 
**Chiara Tardani**
## 🚀 Obiettivo
L'obiettivo delle funzioni implementate è quello di integrare le reti di similarità multiplex, passate in input, in un network (di similarità) unico, ovvero monoplex. I metodi di aggregazione proposti sono stati presi da <https://github.com/DedeBac/SimilarityMatrixAggregation> per quanto riguarda il calcolo dei tre baricentri (Similarity Matrix Average, SMA): 
```markdown
```python
class WeightedMeanAggregator(SimilarityMatrixAggregator) # Aggregatore per la media aritmetica pesata con Frobenius
class GeometricAggregator(SimilarityMatrixAggregator) # Aggregatore per la media geometrica Riemanniana
class WassersteinAggregator(SimilarityMatrixAggregator) # Aggregatore per la media di Wasserstein
```
e da <https://github.com/maxconway/SNFtool/tree/master>, da cui è stato preso il codice in R per il tool SNF (Similarity Network Fusion), è stato tradotto in linguaggio Python e trasformato a sua volta in una classe:
```markdown
```python
class SNFAggregator(SimilarityMatrixAggregator) # Aggregatore che usa Similarity Network Fusion (SNF) 
```
L'obiettivo è infatti sfruttare la potenza di Python come paradigma di programmazione a oggetti, creando delle classi che rappresentino aggregatori di matrici di similarità, con l'idea di trasformare il codice in una libreria Python fruibile.
## 📚 Struttura
Il repository contiene i seguenti scripts .py:

- **aggregation.py** : che contiene le classi dei quattro aggregatori proposti;

- **utils.py** : che contiene delle funzioni che servono per quasi tutti i moduli;

- **example.py** : che contiene un esempio di applicazione del codice a matrici di similarità random.
## ✨ Caratteristiche
Il cuore del progetto è l'uso di una classe astratta, `SimilarityMatrixAggregator`:
```markdown
```python
 class SimilarityMatrixAggregator(ABC):
    """Classe base astratta per aggregatori di matrici di similarità."""
    
    def __init__(self, matrices: List[np.ndarray], weights: Optional[np.ndarray] = None):
        self.matrices = matrices
        self.weights, self.weights_source, self.RV_matrix = self._resolve_weights(weights)
        self._validate_input()
        self.weight_evaluation = self._compute_weight_evaluation()
    
    def _validate_input(self) -> None:
        """Valida gli input delle matrici."""
        if not self.matrices:
            raise ValueError("La lista di matrici non può essere vuota")
        
        # Controlla che tutte le matrici siano numpy array 2D
        for i, matrix in enumerate(self.matrices):
            if not isinstance(matrix, np.ndarray):
                raise TypeError(f"Matrice {i} non è un numpy array")
            if matrix.ndim != 2:
                raise ValueError(f"Matrice {i} non è 2D")
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"Matrice {i} non è quadrata")
        
        # Controlla dimensioni consistenti
        first_shape = self.matrices[0].shape
        for i, matrix in enumerate(self.matrices[1:]):
            if matrix.shape != first_shape:
                raise ValueError(f"Matrice {i+1} ha dimensioni diverse dalla prima")
                
     def _resolve_weights(self, weights: Optional[np.ndarray]) -> Tuple[np.ndarray, str, Optional[np.ndarray]]:
        """Gestisce i pesi: se forniti li valida, altrimenti li calcola.
        In particolare, resituisce (pesi, fonte, matrice RV)."""
        if weights is not None:
            # Validazione pesi utente
            if len(weights) != len(self.matrices):
                raise ValueError(f"Numero di pesi ({len(weights)}) != numero di matrici ({len(self.matrices)})")
            if not np.all(weights >= 0):
                raise ValueError("Tutti i pesi devono essere non negativi")
            if np.sum(weights) == 0:
                raise ValueError("La somma dei pesi non può essere zero")
            
            normalized_weights = weights / np.sum(weights)  # Normalizza
            return normalized_weights, "provided", None  # Nessuna matrice RV per pesi utente
        else:
            # Calcola pesi specifici per il metodo
            computed_weights, RV_matrix = self._compute_method_specific_weights()
            return computed_weights, "computed", RV_matrix
            
    # Utilizziamo il decoratore @abstractmethod per segnare i metodi (metodi astratti)
    # che devono essere implementati dalle classi figlie.
    @abstractmethod
    def _compute_method_specific_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        pass            
    
    @abstractmethod
    def aggregate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calcola l'aggregato delle matrici."""
        pass
        
    # Metodo che restituisce i pesi utilizzati per l'aggregazione (dopo aver creato 
    # l'aggregatore): otteniamo un array NumPy oppure None se i pesi non sono 
    # disponibili (il tipo Optional è usato per precauzione).
    def get_weights(self) -> Optional[np.ndarray]:
        """Restituisce i pesi utilizzati per l'aggregazione."""
        return self.weights
```
da cui derivano le classi figlie citate sopra, che ereditano i metodi di `SimilarityMatrixAggregator(ABC)` e sovrascrivono i metodi astratti (segnati dal decoratore `@abstractmethod`) in modo coerente. In questo modo, risulta semplice aggiungere dei nuovi aggregatori (e, quindi, delle nuove classi figlie):
```markdown
```python
class NewAggregator(SimilarityMatrixAggregator):
  def _compute_method_specific_weights(self) -> Tuple[np.ndarray, np.ndarray]:
     """Qui viene implementato il calcolo specifico
     dei pesi relativi al metodo NewAggregator""".
     return custom_weights_calculation(self.matrices)
  def aggregate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
     """Qui viene implementata l'aggregazione."""
     pass
```
