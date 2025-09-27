import numpy as np
from scipy import linalg
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from utils import *

class SimilarityMatrixAggregator(ABC):
    """Classe base astratta per aggregatori di matrici di similarità."""
    
    def __init__(self, matrices: List[np.ndarray]):
        self.matrices = matrices
        self.weights: Optional[np.ndarray] = None
        self._validate_input()
    
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
    
    @abstractmethod
    def aggregate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calcola l'aggregato delle matrici."""
        pass
    
    def get_weights(self) -> Optional[np.ndarray]:
        """Restituisce i pesi utilizzati per l'aggregazione."""
        return self.weights
        

class WeightedMeanAggregator(SimilarityMatrixAggregator):
    """Aggregatore per la media aritmetica pesata con Frobenius."""
    
    def __init__(self, matrices: List[np.ndarray], weights: Optional[np.ndarray] = None):
        super().__init__(matrices)
        self.weights_source = "provided" if weights is not None else "computed"
        if weights is not None:
            if len(weights) != len(matrices):
                raise ValueError("Il numero di pesi deve corrispondere al numero di matrici")
            if not np.all(weights >= 0):
                raise ValueError("Tutti i pesi devono essere non negativi")
            self.weights = weights / np.sum(weights)  # Normalizza
        else:
            # Calcola pesi automaticamente
            self._compute_weights()
    
    def _compute_weights(self) -> None:
        """Calcola i pesi usando il metodo di Frobenius."""
        # Crea matrice RV per calcolare similarità tra matrici
        n = len(self.matrices)
        RV = np.identity(n)
        
        for i in range(n-1):
            for j in range(i+1, n):
                RV[i, j] = np.trace(self.matrices[i].T @ self.matrices[j]) / (
                    np.linalg.norm(self.matrices[i], 'fro') * np.linalg.norm(self.matrices[j], 'fro'))
        
        RV = RV + RV.T - np.identity(n)
        self.weights = frobenius_weights(RV)
    
    def aggregate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        weighted_sum = np.zeros_like(self.matrices[0])
        for w, matrix in zip(self.weights, self.matrices):
            weighted_sum += w * matrix
        
        info = {
            "method": "weighted_arithmetic_mean", 
            "weights": self.weights.copy(),
            "weight_evaluation": eval_weights(np.diag(self.weights))
        }
        return weighted_sum, info

class GeometricAggregator(SimilarityMatrixAggregator):
    """Aggregatore per la media geometrica Riemanniana."""
    
    def __init__(self, matrices: List[np.ndarray], weights: Optional[np.ndarray] = None,
                 max_iter: int = 200, tolerance: float = 1e-12, corr_factor: float = 0):
        super().__init__(matrices)
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.corr_factor = corr_factor
        self.weights = weights if weights is not None else riem_weights(matrices)[0]
        self.weights_source = "provided" if weights is not None else "computed"
        self.convergence_history: List[float] = []
    
    def _geommean_two(self, A: np.ndarray, B: np.ndarray, t: float) -> np.ndarray:
        """Calcola la media geometrica pesata di due matrici."""
        # Verifica matrici definite positive
        if not (np.all(np.linalg.eigvalsh(A) > 0) and np.all(np.linalg.eigvalsh(B) > 0)):
            A = eigenval_perturb(A)
            B = eigenval_perturb(B)
        
        # Scegli matrice meglio condizionata
        if np.linalg.cond(A) >= np.linalg.cond(B):
            A, B = B, A
            t = 1 - t
        
        # Calcola media geometrica
        lowRA = np.linalg.cholesky(A)
        uppRA = lowRA.T
        invchol1 = np.linalg.inv(lowRA)
        invchol2 = np.linalg.inv(uppRA)
        
        V = invchol1 @ B @ invchol2
        U, diag_vec, _ = np.linalg.svd(V, hermitian=True)
        D = np.diag(diag_vec)
        
        if self.corr_factor != 0:
            D = D + np.min(np.diag(D)) + self.corr_factor
        
        Dpower = np.diag(np.power(np.diag(D), t))
        middle = U @ Dpower @ U.T
        return lowRA @ middle @ uppRA
    
    def aggregate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        if len(self.matrices) == 1:
            return self.matrices[0], {"method": "geometric_mean", "iterations": 0}
        
        if len(self.matrices) == 2:
            result = self._geommean_two(self.matrices[0], self.matrices[1], self.weights[0])
            info = {"method": "geometric_mean", "iterations": 1}
            return result, info
        
        # Algoritmo iterativo per più di 2 matrici
        k = 0
        X_current = self.matrices[k]
        self.convergence_history = []
        
        for iteration in range(self.max_iter):
            k_next = modif_mod(k + 1, len(self.matrices))
            w_exp = self.weights[k_next - 1]
            denom = np.sum([self.weights[modif_mod(i, len(self.matrices)) - 1] 
                          for i in range(k + 1)])
            
            t = w_exp / denom
            X_next = self._geommean_two(X_current, self.matrices[k_next - 1], t)
            
            # Calcola errore di convergenza
            diff = X_next - X_current
            error = np.trace(diff @ diff.T) / np.trace(X_current @ X_current.T)
            self.convergence_history.append(error)
            
            if error <= self.tolerance:
                info = {
                    "method": "geometric_mean", 
                    "iterations": iteration + 1,
                    "convergence_history": self.convergence_history.copy(),
                    "final_error": error
                }
                return X_next, info
            
            X_current = X_next
            k += 1
        
        info = {
            "method": "geometric_mean", 
            "iterations": self.max_iter,
            "convergence_history": self.convergence_history.copy(),
            "final_error": self.convergence_history[-1],
            "warning": "Raggiunto numero massimo di iterazioni"
        }
        return X_current, info

class WassersteinAggregator(SimilarityMatrixAggregator):
    """Aggregatore per la media di Wasserstein."""
    
    def __init__(self, matrices: List[np.ndarray], weights: Optional[np.ndarray] = None,
                 max_iter: int =  10, tolerance: float = 2e-8):
        super().__init__(matrices)
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.weights = weights if weights is not None else riem_weights(matrices)[0]
        self.weights_source = "provided" if weights is not None else "computed"
        self.convergence_history: List[float] = []
    
    def _kx_compute(self, X: np.ndarray) -> np.ndarray:
        """Calcola l'operatore K(X) per l'algoritmo di Wasserstein."""
        rad = square_root_matrix(X)
        negrad = np.linalg.inv(rad)
        to_sum = [np.zeros_like(X) for _ in range(len(self.matrices))]
        
        for i, matrix in enumerate(self.matrices):
            a = rad @ matrix @ rad
            sqa = square_root_matrix(a)
            to_sum[i] = self.weights[i] * sqa
        
        somma = np.sum(to_sum, axis=0)
        sq_somma = somma @ somma
        return negrad @ sq_somma @ negrad
    
    def aggregate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        if len(self.matrices) == 1:
            return self.matrices[0], {"method": "wasserstein_mean", "iterations": 0}
        
        if len(self.matrices) == 2:
            # Implementazione forma chiusa per 2 matrici
            s1 = (self.weights[0]**2) * self.matrices[0]
            s2 = (self.weights[1]**2) * self.matrices[1]
            s12 = self.weights[0] * self.weights[1] * (
                square_root_matrix(self.matrices[0] @ self.matrices[1]) + 
                square_root_matrix(self.matrices[1] @ self.matrices[0]))
            result = s1 + s2 + s12
            info = {"method": "wasserstein_mean", "iterations": 1}
            return result, info
        
        # Algoritmo iterativo per più matrici
        X_current = self.matrices[0]  # Inizia con la prima matrice
        self.convergence_history = []
        
        for iteration in range(self.max_iter):
            X_next = self._kx_compute(X_current)
            
            # Calcola errore di convergenza
            diff = X_next - X_current
            error = np.trace(diff @ diff.T) / np.trace(X_current @ X_current.T)
            self.convergence_history.append(error)
            
            if error <= self.tolerance:
                info = {
                    "method": "wasserstein_mean",
                    "iterations": iteration + 1,
                    "convergence_history": self.convergence_history.copy(),
                    "final_error": error
                }
                return X_next, info
            
            X_current = X_next
        
        info = {
            "method": "wasserstein_mean",
            "iterations": self.max_iter,
            "convergence_history": self.convergence_history.copy(),
            "final_error": self.convergence_history[-1],
            "warning": "Raggiunto numero massimo di iterazioni"
        }
        return X_current, info

# Funzione di convenienza per l'aggregazione
def aggregate(matrices: List[np.ndarray], method: str = 'mean', **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Funzione helper per aggregare matrici usando diversi metodi.
    
    Args:
        matrices: Lista di matrici di similarità
        method: Metodo di aggregazione ('mean', 'weighted_mean', 'geometric', 'wasserstein')
        **kwargs: Parametri aggiuntivi specifici del metodo
    
    Returns:
        Matrice aggregata e informazioni sul processo
    """
    # Definiamo un dizionario che associa ogni stringa 'method' alla classe corrispondente
    aggregators = { 
        'mean': MeanAggregator,
        'weighted_mean': WeightedMeanAggregator,
        'geometric': GeometricAggregator,
        'wasserstein': WassersteinAggregator
    }
    
    if method not in aggregators:
        raise ValueError(f"Metodo non supportato: {method}. Metodi disponibili: {list(aggregators.keys())}")
    aggregator = aggregators[method](matrices, **kwargs)
    barycenter, info = aggregator.aggregate() 
    norm_barycenter = normalize_simmat(barycenter)
    # Restituiamo un dizionario completo (con la versione originale del baricentro computato, la versione normalizzata,
    # così che gli elementi diagonali siano uguali ad 1, e info sul metodo di aggregazione)
    return {    
        'original': barycenter,
        'normalized': normalized_barycenter,
        'info': info
    }
