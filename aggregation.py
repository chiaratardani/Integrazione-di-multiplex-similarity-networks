import numpy as np
from scipy import linalg
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from utils import *

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

    def _compute_weight_evaluation(self) -> Optional[float]:
        """Calcola la valutazione dei pesi usando la matrice RV."""
        if self.RV_matrix is not None:
            return eval_weights(self.RV_matrix)
        return None  # Se non disponibile

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


class WeightedMeanAggregator(SimilarityMatrixAggregator):
    """Aggregatore per la media aritmetica pesata con Frobenius."""

    def __init__(self, matrices: List[np.ndarray], weights: Optional[np.ndarray] = None):
        super().__init__(matrices, weights)  # Matrici e pesi gestiti dalla classe base (astratta)

    def _compute_method_specific_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calcola i pesi (e matrice RV) usando il metodo di Frobenius."""
        # Crea matrice RV per calcolare similarità tra matrici
        n = len(self.matrices)
        RV = np.identity(n)

        for i in range(n-1):
            for j in range(i+1, n):
                RV[i, j] = np.trace(self.matrices[i].T @ self.matrices[j]) / (
                    np.linalg.norm(self.matrices[i], 'fro') * np.linalg.norm(self.matrices[j], 'fro'))

        RV = RV + RV.T - np.identity(n)
        weights = frobenius_weights(RV)
        return weights, RV

    def aggregate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        weighted_sum = np.zeros_like(self.matrices[0], dtype=np.float64)
        for w, matrix in zip(self.weights, self.matrices):
            weighted_sum += w * matrix

        info = {
            "method": "weighted_arithmetic_mean",
            "weights": self.weights.copy(),
            "RV_matrix": self.RV_matrix,  # Matrice RV
            "weight_evaluation": self.weight_evaluation  # Funzione che valuta quanto bene i pesi rappresentano le matrici
        }
        return weighted_sum, info

class GeometricAggregator(SimilarityMatrixAggregator):
    """Aggregatore per la media geometrica Riemanniana."""

    def __init__(self, matrices: List[np.ndarray], weights: Optional[np.ndarray] = None, k_init: int = 0,
                 max_iter: int = 200, tolerance: float = 1e-12, corr_factor: float = 0):
        super().__init__(matrices, weights)
        self.k_init = k_init
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.corr_factor = corr_factor
        self.convergence_history: List[float] = []

    def _compute_method_specific_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calcola pesi e matrice di correlazione automaticamente con riem_weights."""
        weights, corr_matrix = riem_weights(self.matrices)
        return weights, corr_matrix  # corr_matrix è la nostra "matrice RV"

    def _geommean_two(self, A: np.ndarray, B: np.ndarray, t: float) -> np.ndarray:
        """Calcola la media geometrica pesata di due matrici (A e B),
        con peso t (compreso tra 0 ed 1)."""
        # Verifica matrici definite positive, altrimenti le 'perturba'
        if not (np.all(np.linalg.eigvalsh(A) > 0) and np.all(np.linalg.eigvalsh(B) > 0)):
            A = eigenval_perturb(A)
            B = eigenval_perturb(B)

        # Scegli matrice meglio condizionata: scambia le matrici (se necessario),
        # per avere quella meglio condizionata come prima.
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
        self.convergence_history = [] # Reset dell'attributo ad ogni chiamata di 'aggregate'
        if len(self.matrices) == 1:
            return self.matrices[0], {"method": "geometric_mean", "iterations": 0, "weights_source": self.weights_source,"RV_matrix": self.RV_matrix,
            "weight_evaluation": self.weight_evaluation}

        if len(self.matrices) == 2: # Forma chiusa della media geometrica pesata
            result = self._geommean_two(self.matrices[1], self.matrices[0], self.weights[0])
            info = {"method": "geometric_mean", "iterations": 1, "weights_source": self.weights_source,"RV_matrix": self.RV_matrix,
            "weight_evaluation": self.weight_evaluation}
            return result, info

        # Algoritmo iterativo per più di 2 matrici
        k = self.k_init  # Indice iniziale parametrizzabile
        jk = modif_mod(k, len(self.matrices))  # Indice circolare
        X_current = self.matrices[jk - 1]  # Matrice di partenza

        for iter_count in range(1, self.max_iter + 1):
            jk_next = modif_mod(k + 1, len(self.matrices))
            w_exp = self.weights[jk_next - 1]
            S_next = self.matrices[jk_next - 1]

            # Calcolo denominatore
            denom = 0
            indices = np.array(list(range(k + 1))) + 1
            for i in indices:
                denom += self.weights[modif_mod(i, len(self.matrices)) - 1]

            t = w_exp / denom
            X_next = self._geommean_two(X_current, S_next, t)

            # Calcola errore di convergenza
            diff = X_next - X_current
            diff = X_next - X_current
            num_err = np.trace(diff @ diff.T)
            den_err = np.trace(X_current @ X_current.T)
            error = num_err / den_err
            self.convergence_history.append(error)

            # Controllo convergenza
            if error <= self.tolerance:
                print(f"Convergenza raggiunta all'iterazione: {iter_count}")
                info = {
                    "method": "geometric_mean",
                    "iterations": iter_count,
                    "weights_source": self.weights_source,
                    "RV_matrix": self.RV_matrix,
                    "weight_evaluation": self.weight_evaluation,
                    "convergence_history": self.convergence_history.copy(),
                    "final_error": error
               }
                return X_next, info # In caso di convergenza, riotorniamo X_next (l'ultimo computato)

            # Prepariamo la prossima iterazione
            X_current = X_next
            k += 1

        print('Raggiunto numero massimo di iterazioni')
        info = {
            "method": "geometric_mean",
            "iterations":  self.max_iter,
            "weights_source": self.weights_source,
            "RV_matrix": self.RV_matrix,
            "weight_evaluation": self.weight_evaluation,
            "convergence_history": self.convergence_history.copy(),
            "final_error": self.convergence_history[-1] if self.convergence_history else float('inf') # Controllo se la lista è vuota per robustezza
        }

        return X_current, info

class WassersteinAggregator(SimilarityMatrixAggregator):
    """Aggregatore per la media di Wasserstein."""

    def __init__(self, matrices: List[np.ndarray], weights: Optional[np.ndarray] = None,
                 max_iter: int =  10, tolerance: float = 2e-8):
        super().__init__(matrices, weights)
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.convergence_history: List[float] = []

    def _compute_method_specific_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calcola pesi e matrice di correlazione automaticamente con riem_weights."""
        weights, corr_matrix = riem_weights(self.matrices)
        return weights, corr_matrix

    # Funzione helper per la computazione del baricentro di Wasserstein
    # per più di due matrici (implementa iterazione fixed-point).
    def _kx_compute(self, X: np.ndarray) -> np.ndarray:
        """Calcola l'operatore K(X) per l'algoritmo di Wasserstein, con X=X_current."""
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
        self.convergence_history = [] # Reset della storia di convergenza
        if len(self.matrices) == 1:
            return self.matrices[0], {"method": "wasserstein_mean", "iterations": 0, "weights_source": self.weights_source,
            "RV_matrix": self.RV_matrix, "weight_evaluation": self.weight_evaluation}

        if len(self.matrices) == 2:
            # Implementazione forma chiusa per 2 matrici
            s1 = (self.weights[0]**2) * self.matrices[0]
            s2 = (self.weights[1]**2) * self.matrices[1]
            s12 = self.weights[0] * self.weights[1] * (
                square_root_matrix(self.matrices[0] @ self.matrices[1]) +
                square_root_matrix(self.matrices[1] @ self.matrices[0]))
            result = s1 + s2 + s12
            info = {"method": "wasserstein_mean", "iterations": 1, "weights_source": self.weights_source,
            "RV_matrix": self.RV_matrix, "weight_evaluation": self.weight_evaluation}
            return result, info

        # Algoritmo iterativo per più matrici
        import random
        k = random.randint(0, len(self.matrices) - 1) # Scegliamo random una matrice iniziale
        X_current = self.matrices[k]
        self.convergence_history = []

        for iter_count in range(1, self.max_iter + 1):
            X_next = self._kx_compute(X_current)

            # Calcola errore di convergenza
            diff = X_next - X_current
            error = np.trace(diff @ diff.T) / np.trace(X_current @ X_current.T)
            self.convergence_history.append(error)

            if error <= self.tolerance:
                print(f"Convergenza raggiunta all'iterazione: {iter_count}")
                info = {
                    "method": "wasserstein_mean",
                    "iterations": iter_count,
                    "weights_source": self.weights_source,
                    "RV_matrix": self.RV_matrix,
                    "weight_evaluation": self.weight_evaluation,
                    "convergence_history": self.convergence_history.copy(),
                    "final_error": error
                }
                return X_next, info

            X_current = X_next

        print('Raggiunto numero massimo di iterazioni')
        info = {
            "method": "wasserstein_mean",
            "iterations": self.max_iter,
            "weights_source": self.weights_source,
            "RV_matrix": self.RV_matrix,
            "weight_evaluation": self.weight_evaluation,
            "convergence_history": self.convergence_history.copy(),
            "final_error": self.convergence_history[-1] if self.convergence_history else float('inf')
        }
        return X_current, info


class SNFAggregator(SimilarityMatrixAggregator):
    """
    Aggregatore che usa Similarity Network Fusion (SNF) per integrare
    multiple matrici di similarità.
    """

    def __init__(self, matrices: List[np.ndarray], weights: Optional[np.ndarray] = None,
                 K: int = 20, t: int = 20, alpha: float = 0.5):
        """
        Args:
            matrices: Liste di matrici di similarità da fondere
            K: Numero di nearest neighbors per la costruzione della matrice di affinità (default: 20)
            t: Numero di iterazioni per la fusione (default: 20)
            alpha: Parametro di regolarizzazione (default: 1.0)
        """
        super().__init__(matrices, weights)
        self.K = K
        self.t = t
        self.alpha = alpha

    # Per conservare i moduli relativi ai pesi (richiesti dai tre aggregatori precedenti),
    # impostiamo, per SNF (che non richiede pesi), dei pesi uniformi.
    def _compute_method_specific_weights(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        SNF non usa pesi tradizionali, quindi restituiamo pesi uniformi.
        La fusione avviene attraverso l'algoritmo SNF stesso.
        """
        return np.ones(len(self.matrices)) / len(self.matrices), None # La matrice RV è None

    def _affinity_matrix(self, W: np.ndarray, K: int) -> np.ndarray:
        """Costruisce la matrice di affinità da una matrice di similarità."""
        n = W.shape[0]
        affinity = np.zeros((n, n))

        for i in range(n):
            # Trova i K nearest neighbors (escludendo se stesso)
            indices = np.argsort(W[i])[::-1][1:K+1] # Indici dei K più simili
            for j in indices:
                affinity[i, j] = W[i, j]

        # Rendi simmetrica
        affinity = (affinity + affinity.T) / 2
        return affinity # Ritorniamo una matrice sparsa che rappresenta le affinità locali

    def _normalized_cut(self, W: np.ndarray) -> np.ndarray:
        """Applica il normalized cut alla matrice di similarità W.
        Lo scopo è normalizzare la matrice (normalizzata riga per riga) 
        per bilanciare l'influenza dei nodi con molti collegamenti."""
        # Calcola la matrice dei gradi (diagonale con somma delle righe di W)
        D = np.diag(np.sum(W, axis=1))

        # Calcola D^(-1/2)
        D_sqrt_inv = np.linalg.pinv(np.sqrt(D)) # Matrice pseudo-inversa

        # Normalized cut: D^(-1/2) * W * D^(-1/2)
        W_normalized = D_sqrt_inv @ W @ D_sqrt_inv
        return W_normalized

    def aggregate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applica l'algoritmo SNF per fondere le matrici di similarità.
        """
        if len(self.matrices) == 1:
            info = {
                "method": "snf",
                "iterations": 0,
                "RV_matrix": self.RV_matrix,  # Per mantenere consistenza
                "weight_evaluation": self.weight_evaluation,
                "weights_source": self.weights_source,
                "parameters": {"K": self.K, "t": self.t, "alpha": self.alpha}
            }
            return self.matrices[0], info

        # Normalizza ogni matrice di similarità
        W_normalized = []
        for W in self.matrices: # W è una matrice di similarità
            W_norm = self._normalized_cut(W)
            W_normalized.append(W_norm)

        # Costruisce le matrici di similarità locale per ogni vista
        S_local = []
        for W in W_normalized:
            S = self._affinity_matrix(W, self.K)
            S_local.append(S)

        # Algoritmo di fusione SNF: W_current è una lista di matrici,
        # ognuna rappresentante una vista, e in ogni iterazione aggiorniamo ogni vista in base alle altre
        W_current = W_normalized.copy()

        for iteration in range(self.t): # Iterazione di fusione
            W_new = []
            for i in range(len(W_current)):
                # Combina le similarità dalle altre viste (ogni matrice
                # in self.matrices rappresenta una 'vista' diversa sui dati)
                other_views = [W_current[j] for j in range(len(W_current)) if j != i]

                if other_views:
                    # Media delle altre viste
                    W_other = np.mean(other_views, axis=0)
                else:
                    W_other = W_current[i]  # Se c'è solo una matrice, usa se stessa

                # Aggiorna la similarità: fusione della vista corrente con le altre
                W_update = S_local[i] @ W_other @ S_local[i].T # dove S_local[i] è la matrice di affinità locale della vista i

                # Applichiamo il normalized cut a W_update e regolarizziamo
                W_update = self._normalized_cut(W_update)

                # Regolarizzazione
                W_update = self.alpha * W_update + (1 - self.alpha) * W_current[i]
                W_new.append(W_update)

            W_current = W_new

        # Matrice fusa finale (media di tutte le viste): media di tutte le matrici di
        # affinità dopo t-iterazioni)
        W_fused = np.mean(W_current, axis=0)

        info = {
            "method": "snf",
            "iterations": self.t,
            "RV_matrix": self.RV_matrix,
            "weight_evaluation": self.weight_evaluation,
            "weights_source": self.weights_source,
            "parameters": {"K": self.K, "t": self.t, "alpha": self.alpha}
        }

        return W_fused, info

# Funzione di convenienza per l'aggregazione
def aggregate(matrices: List[np.ndarray], method: str = 'snf', weights: Optional[np.ndarray] = None, **kwargs) ->  Dict[str, Any]:
    """
    Funzione helper per aggregare matrici usando diversi metodi.

    Args:
        matrices: Lista di matrici di similarità
        method: Metodo di aggregazione ('weighted_mean', 'geometric', 'wasserstein', 'snf')
        weights: Pesi opzionali per l'aggregazione
        **kwargs: Parametri aggiuntivi specifici del metodo

    Returns:
        Matrice aggregata e informazioni sul processo
    """
    # Definiamo un dizionario che associa ogni stringa 'method' alla classe corrispondente
    aggregators = {
        'weighted_mean': WeightedMeanAggregator,
        'geometric': GeometricAggregator,
        'wasserstein': WassersteinAggregator,
        'snf': SNFAggregator
    }

    if method not in aggregators:
        raise ValueError(f"Metodo non supportato: {method}. Metodi disponibili: {list(aggregators.keys())}")
    aggregator = aggregators[method](matrices, weights=weights, **kwargs)
    result, info = aggregator.aggregate()
    normalized_result = normalize_simmat(result)
    # Restituiamo un dizionario completo (con la versione originale del baricentro computato, la versione normalizzata,
    # così che gli elementi diagonali siano uguali ad 1, e info sul metodo di aggregazione)
    return {
        'aggregated_matrix': result,
        'normalized_aggregated_matrix': normalized_result,
        'info': info
    }
