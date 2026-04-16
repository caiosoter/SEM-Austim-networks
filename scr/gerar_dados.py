import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import eigsh
import warnings
from dataclasses import dataclass
from typing import List, Dict
warnings.filterwarnings("ignore")


def generate_logits(R: int, noise_type: str):
    """
    Gera ruídos e aplica Padronização Empírica (Z-Score).
    Garante que todo ruído retornado tenha média=0 e std=1 exatos para aquela amostra.
    """

    if noise_type == "uniform":
        raw = np.random.uniform(0, 1, size=R)
        
    elif noise_type == "chi_square":
        raw = np.random.chisquare(df=2, size=R)
        
    elif noise_type == "exponential":
        raw = np.random.exponential(scale=1.0, size=R)
        
    elif noise_type == "laplace":
        raw = np.random.laplace(loc=0, scale=1, size=R)
        
    elif noise_type == "logistic":
        raw = np.random.logistic(loc=0, scale=1, size=R)
        
    elif noise_type == "student_t":
        raw = np.random.standard_t(df=3, size=R)
        
    elif noise_type == "F":
        raw = np.random.f(dfnum=5, dfden=10, size=R)

    elif noise_type == "gaussian":
        raw = np.random.normal(loc=0, scale=1, size=R)
        
    else:
        raise ValueError(f"Tipo de ruído desconhecido: {noise_type}")

    return (raw - np.mean(raw)) / np.std(raw)




def get_largest_eigenvalue_er(n, p):
    try:
        p = np.clip(p, 0.001, 0.999)
        G = ig.Graph.Erdos_Renyi(n=n, p=p, directed=False)
        if G.ecount() == 0: return 0.0
        val = G.eigenvector_centrality(return_eigenvalue=True)[1]
        return val
    except Exception:
        return np.nan



def get_largest_eigenvalue_ba(n, power, m=1):
    try:
        G = ig.Graph.Barabasi(n=n, m=m, power=power, directed=False)
        if G.ecount() == 0: return 0.0
        val = G.eigenvector_centrality(return_eigenvalue=True)[1]
        return val   
    except Exception:
        return np.nan


def get_largest_eigenvalue_ge(n, raio):
    try:
        G = ig.Graph.GRG(n, raio)
        if G.ecount() == 0: return 0.0
        val = G.eigenvector_centrality(return_eigenvalue=True)[1]
        return val
    except Exception:
        return np.nan
    

def get_largest_eigenvalue_kr(n, k):
    try:
        G = ig.Graph.K_Regular(n, k, directed=False)
        if G.ecount() == 0: return 0.0
        val = G.eigenvector_centrality(return_eigenvalue=True)[1]
        return val
    except Exception:
        return np.nan
    

    

def get_largest_eigenvalue_ws(n, p_rewire, k=3):
    try:
        p_rewire = np.clip(p_rewire, 0.001, 0.999)
        G = ig.Graph.Watts_Strogatz(dim=1, size=n, nei=k, p=p_rewire)
        if G.ecount() == 0: return 0.0
        val = G.eigenvector_centrality(return_eigenvalue=True)[1]
        return val
    except Exception:
        return np.nan




def padronizacao(x):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled





def gerar_estrutura_causal_logits(R, tipo_ruido, estrutura_causal):

    if estrutura_causal == {"A->B"}:
        ruido_A = generate_logits(R, tipo_ruido)
        ruido_B = generate_logits(R, tipo_ruido)

        A = 0.5 * generate_logits(R, tipo_ruido) + ruido_A
        B = 0.5 * A + ruido_B
        return A, B

    elif estrutura_causal == {"B->A"}:
        ruido_A = generate_logits(R, tipo_ruido)
        ruido_B = generate_logits(R, tipo_ruido)

        B = 0.5 * generate_logits(R, tipo_ruido) + ruido_B
        A = 0.5 * B + ruido_A
        return A, B

    elif estrutura_causal == {"A<->B"}:
        ruido_A = generate_logits(R, tipo_ruido)
        ruido_B = generate_logits(R, tipo_ruido)

        A = 0.5 * generate_logits(R, tipo_ruido) + ruido_A
        B = 0.5 * generate_logits(R, tipo_ruido) + ruido_B
        return A, B
    
    elif estrutura_causal == {"A->B", "B->C"}:
        ruido_B = generate_logits(R, tipo_ruido)
        ruido_C = generate_logits(R, tipo_ruido)

        A = generate_logits(R, tipo_ruido)
        B = 0.5 * A + ruido_B
        C = -0.5 * B + ruido_C
        return A, B, C
    
    elif estrutura_causal == {"B->A", "B->C"}:
        ruido_A = generate_logits(R, tipo_ruido)
        ruido_C = generate_logits(R, tipo_ruido)

        B = generate_logits(R, tipo_ruido)
        A = 0.5 * B + ruido_A
        C = -0.5 * B + ruido_C
        return A, B, C
    
    elif estrutura_causal == {"B->A", "C->A"}:
        ruido_A = generate_logits(R, tipo_ruido)
        C = generate_logits(R, tipo_ruido)
        B = generate_logits(R, tipo_ruido)
        A = 0.5*B - 0.5*C + ruido_A
        return A, B, C
    
    elif estrutura_causal == {"A->B", "A->C"}:
        ruido_B = generate_logits(R, tipo_ruido)
        ruido_C = generate_logits(R, tipo_ruido)

        A = generate_logits(R, tipo_ruido)
        B = 0.5 * A + ruido_B
        C = -0.5 * A + ruido_C
        return A, B, C

    
    elif estrutura_causal == {"A->B", "B->C", "A->C"}:
        ruido_B = generate_logits(R, tipo_ruido)
        ruido_C = generate_logits(R, tipo_ruido)
        A = generate_logits(R, tipo_ruido)
        B = 0.5* A + ruido_B
        C = -0.5 * A + 0.5 * B + ruido_C
        return A, B, C
    
    elif estrutura_causal == {"A->B", "B->C", "A->C", "C->D"}:
        ruido_B = generate_logits(R, tipo_ruido)
        ruido_C = generate_logits(R, tipo_ruido)
        ruido_D = generate_logits(R, tipo_ruido)

        A = generate_logits(R, tipo_ruido)
        B = 0.5* A + ruido_B
        C = -0.5 * A + 0.5 * B + ruido_C
        D = 0.5 * C + ruido_D
        return A, B, C, D
    
    return None


    




# GERA DADOS PARA GRAFOS COM DOIS VÉRTICES
def gerar_dados_AB(num_vertices, R, tipo_ruido, modelo_gerador, estrutura_causal):
    """
    Gera dados simulados com base no modelo gerador especificado.
    
    Parâmetros:
      num_vertices: Número de variáveis (vértices).
      R: Número de amostras.
      tipo_ruido: Tipo de ruído a ser gerado.
      modelo_gerador: Modelo de grafo para gerar a estrutura causal.
    
    Retorna:
      Dados simulados como um array NumPy de forma (R, num_vertices).
    """

    A, B = gerar_estrutura_causal_logits(R, tipo_ruido, estrutura_causal)

        
    logit_A = 1 / (1 + np.exp(-A))
    logit_B = 1 / (1 + np.exp(-B))

    eigen_A = np.zeros(R)
    eigen_B = np.zeros(R)


    if modelo_gerador == "erdos_renyi":
        
        for j in range(R):
            eigen_A[j] = get_largest_eigenvalue_er(num_vertices, p=logit_A[j])
            eigen_B[j] = get_largest_eigenvalue_er(num_vertices, p=logit_B[j])
    
    elif modelo_gerador == "barabasi_albert":
        m = 1  # Número de arestas a serem anexadas a cada novo nó
        power_A = logit_A * 2
        power_B = logit_B * 2

        for j in range(R):
            eigen_A[j] = get_largest_eigenvalue_ba(num_vertices, power_A[j], m)
            eigen_B[j] = get_largest_eigenvalue_ba(num_vertices, power_B[j], m)

    elif modelo_gerador == "geometric":
        
        for j in range(R):
            eigen_A[j] = get_largest_eigenvalue_ge(num_vertices, raio=logit_A[j])
            eigen_B[j] = get_largest_eigenvalue_ge(num_vertices, raio=logit_B[j])

    elif modelo_gerador == "k_regular":
        k_A = (logit_A * 10).astype(int)
        k_B = (logit_B * 10).astype(int)


        for j in range(R):
            eigen_A[j] = get_largest_eigenvalue_kr(num_vertices, k_A[j])
            eigen_B[j] = get_largest_eigenvalue_kr(num_vertices, k_B[j])

    elif modelo_gerador == "watts_strogatz":
        k_nei = 3
        for j in range(R):
            eigen_A[j] = get_largest_eigenvalue_ws(num_vertices, p_rewire=logit_A[j], k=k_nei)
            eigen_B[j] = get_largest_eigenvalue_ws(num_vertices, p_rewire=logit_B[j], k=k_nei)
    else:
        raise ValueError(f"Modelo gerador desconhecido: {modelo_gerador}")


    df = pd.DataFrame({'A': eigen_A, 'B': eigen_B})
    return df




# GERA DADOS PARA GRAFOS COM TRÊS VÉRTICES
def gerar_dados_ABC(num_vertices, R, tipo_ruido, modelo_gerador, estrutura_causal):
    """
    Gera dados simulados com base no modelo gerador especificado.
    
    Parâmetros:
      num_vertices: Número de variáveis (vértices).
      R: Número de amostras.
      tipo_ruido: Tipo de ruído a ser gerado.
      modelo_gerador: Modelo de grafo para gerar a estrutura causal.
    
    Retorna:
      Dados simulados como um array NumPy de forma (R, num_vertices).
    """
    
    A, B, C = gerar_estrutura_causal_logits(R, tipo_ruido, estrutura_causal)

        
    logit_A = 1 / (1 + np.exp(-A))
    logit_B = 1 / (1 + np.exp(-B))
    logit_C = 1 / (1 + np.exp(-C))

    eigen_A = np.zeros(R)
    eigen_B = np.zeros(R)
    eigen_C = np.zeros(R)


    if modelo_gerador == "erdos_renyi":
        
        for j in range(R):
            eigen_A[j] = get_largest_eigenvalue_er(num_vertices, p=logit_A[j])
            eigen_B[j] = get_largest_eigenvalue_er(num_vertices, p=logit_B[j])
            eigen_C[j] = get_largest_eigenvalue_er(num_vertices, p=logit_C[j])
    
    elif modelo_gerador == "barabasi_albert":
        m = 1  # Número de arestas a serem anexadas a cada novo nó
        power_A = logit_A * 2
        power_B = logit_B * 2
        power_C = logit_C * 2

        for j in range(R):
            eigen_A[j] = get_largest_eigenvalue_ba(num_vertices, power_A[j], m)
            eigen_B[j] = get_largest_eigenvalue_ba(num_vertices, power_B[j], m)
            eigen_C[j] = get_largest_eigenvalue_ba(num_vertices, power_C[j], m)

            

    elif modelo_gerador == "geometric":
        
        for j in range(R):
            eigen_A[j] = get_largest_eigenvalue_ge(num_vertices, raio=logit_A[j])
            eigen_B[j] = get_largest_eigenvalue_ge(num_vertices, raio=logit_B[j])
            eigen_C[j] = get_largest_eigenvalue_ge(num_vertices, raio=logit_C[j])


    elif modelo_gerador == "k_regular":
        k_A = (logit_A * 10).astype(int)
        k_B = (logit_B * 10).astype(int)
        k_C = (logit_C * 10).astype(int)

        for j in range(R):
            eigen_A[j] = get_largest_eigenvalue_kr(num_vertices, k_A[j])
            eigen_B[j] = get_largest_eigenvalue_kr(num_vertices, k_B[j])
            eigen_C[j] = get_largest_eigenvalue_kr(num_vertices, k_C[j])


    elif modelo_gerador == "watts_strogatz":
        k_nei = 3
        for j in range(R):
            eigen_A[j] = get_largest_eigenvalue_ws(num_vertices, p_rewire=logit_A[j], k=k_nei)
            eigen_B[j] = get_largest_eigenvalue_ws(num_vertices, p_rewire=logit_B[j], k=k_nei)
            eigen_C[j] = get_largest_eigenvalue_ws(num_vertices, p_rewire=logit_C[j], k=k_nei)

    else:
        raise ValueError(f"Modelo gerador desconhecido: {modelo_gerador}")


    df = pd.DataFrame({'A': eigen_A, 'B': eigen_B, 'C':eigen_C})
    return df







def get_caso_real_1():
    return {"formula": {"A<->B"}, "Caso":"Caso_1", "lista_variaveis":["A", "B"], "dados": gerar_dados_AB}

def get_caso_real_2():
    return {"formula": {"A->B"}, "Caso":"Caso_2", "lista_variaveis":["A", "B"], "dados": gerar_dados_AB}

def get_caso_real_3():
    return {"formula": {"A->B", "A->C"}, "Caso":"Caso_3", "lista_variaveis":["A", "B", "C"], "dados": gerar_dados_ABC}

def get_caso_real_4():
    return {"formula": {"A->B", "B->C", "A->C"}, "Caso":"Caso_4", "lista_variaveis":["A", "B", "C"], 'dados':gerar_dados_ABC}





@dataclass
class ConfiguracaoSimulacao:
    n_sims: int
    noises: List[str]
    vetor_R: List[int]
    vetor_vertices: List[int]
    modelos_grafos: List[str]
    modelos_candidatos: Dict[str, Dict]
    modelo_real: Dict[str, any]
    alpha: float = 0.05
    tol_rmsea: float = 0.08
    tol_aic: float = 2