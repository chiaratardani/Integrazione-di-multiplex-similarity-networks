import numpy as np
import scipy
from scipy import linalg
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import linalg
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union

def cos_sim(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calcola la matrice di similarità coseno tra le righe di una matrice."""
    B_square = B.dot(B.T)
    dim = B_square.shape[0]
    sim_mat = np.zeros((dim, dim))
    
    for i in range(dim):
        for j in range(dim):
            if B_square[i,i] > 0 and B_square[j,j] > 0:  # Evita divisione per zero
                sim_mat[i,j] = B_square[i,j] / np.sqrt(B_square[i,i] * B_square[j,j])
    
    return sim_mat, B_square
    
# Questa funzione deve essere utilizzata per normalizzare i valori dei baricentri
def normalize_simmat(sim_mat: np.ndarray) -> np.ndarray: 
    """Normalizza una matrice di similarità."""
    sim_mat_norm = np.zeros(sim_mat.shape)
    for i in range(sim_mat.shape[0]):
        for j in range(sim_mat.shape[0]):
            if sim_mat[i,i] > 0 and sim_mat[j,j] > 0:
                sim_mat_norm[i,j] = sim_mat[i,j] / ((sim_mat[i,i]**(1/2)) * (sim_mat[j,j]**(1/2)))
    return sim_mat_norm

def modif_mod(k: int, m: int) -> int:
    """Funzione che restituisce m quando 
    il resto (modulo k%m) è 0."""
    res = k % m
    return res if res != 0 else m

def eval_weights(RVcoeff: np.ndarray) -> float:
    """Valuta quanto bene i pesi rappresentano le matrici."""
    eigenvals = np.linalg.eigvals(RVcoeff)
    eigenvals_sorted = np.sort(eigenvals)[::-1]
    e_max = eigenvals_sorted[0]
    return e_max / np.sum(eigenvals_sorted)

def eigenval_perturb(data_matrix: np.ndarray, tol_eig: float = 2e-12, 
                    pert_tol: float = 2e-12) -> np.ndarray:
    """Perturba leggermente gli autovalori per stabilizzare matrici quasi singolari.
    tol_eig è una piccola costante usata per perturbare gli autovalori, mentre
    pert_tol è una tolleranza sulla distanza della matrice dall'originale."""
    eva, evec = np.linalg.eigh(data_matrix)
    eva_neg = eva[eva < 0]
    
    if len(eva_neg) == 0:
        return data_matrix
    
    eva_t = eva + np.abs(np.min(eva_neg)) + tol_eig
    reconstructed = np.dot(evec * eva_t, evec.conj().T)
    return reconstructed

def frobenius_weights(RV: np.ndarray) -> np.ndarray:
    """Calcola pesi per la media aritmetica pesata usando Frobenius (prendendo
    l'autovettore corrispondente al più grande autovalore della matrice RV."""
    eigenvals, eigenvecs = np.linalg.eig(RV)
    idx = eigenvals.argsort()[::-1]
    max_evec = eigenvecs[:, idx[0]]
    weights_vector = max_evec / np.sum(max_evec)
    return weights_vector

def riem_weights(matrices: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Calcola pesi per la media Riemanniana, utilizzando la metrica di Riemann."""
    n = len(matrices)
    corr_mat = np.identity(n)
    
    for i in range(n-1):
        mi = matrices[i]
        for j in range(i+1, n):
            mj = matrices[j]
            corr_mat[i, j] = np.trace(mi.T @ mj) / (np.linalg.norm(mi, 'fro') * np.linalg.norm(mj, 'fro'))
    
    corr_mat1 = corr_mat + corr_mat.T - np.identity(n)
    w_denom = np.ones((1, n)) @ (corr_mat1 - np.identity(n)) @ np.ones((n, 1))
    w_num = (corr_mat1 - np.identity(n)) @ np.ones((n, 1))
    w = w_num / w_denom
    return w.flatten(), corr_mat1

def square_root_matrix(A: np.ndarray) -> np.ndarray:
    """Calcola la radice quadrata di una matrice."""
    eigvals, V = np.linalg.eig(A)
    diag_root = np.diag([v**(1/2) for v in eigvals])
    return V @ diag_root @ np.linalg.inv(V)

