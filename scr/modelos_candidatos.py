import itertools


def get_modelos_candidatosAB():
    return {
        "M1_Independente": {"formula": "B ~ 0*A\nA~~A\nB~~B", "Arestas": ["Independent"]},
        "M2_A_causa_B": {"formula": "B ~ A\nA~~A", "Arestas": ["A->B"]},
        "M3_B_causa_A": {"formula": "A ~ B\nB~~B", "Arestas": ["B->A"]}
    }



def get_modelos_candidatosABC():
    return {
        # --- TOTALMENTE INDEPENDENTES (NULL MODEL) ---
        # Regra: Ninguém tem "pai" real. Todas são exógenas.
        # 1. Estimamos variância de todas (~~).
        # 2. Forçamos regressão zero (0*) para testar a independência estatística.
        
        "M00_Independent_All": {
           "formula": "A ~~ A\nB ~~ B\nC ~~ C\nB ~ 0*A\nC ~ 0*A + 0*B",
            "Arestas": []
        },

        # --- PARCIALMENTE INDEPENDENTES (1 ARESTA + 1 ISOLADA) ---
        # Regra:
        # 1. A Raiz da aresta (ex: A em A->B) é exógena -> precisa de ~~.
        # 2. A Variável Isolada (ex: C) é exógena -> precisa de ~~ e recebe 0* das outras.
        
        # Par A->B (C isolado)
        "M19_Single_A_to_B_Ind_C": {
            # A é raiz (~~A). C é isolado (~~C).
            "formula": "B ~ A\nC ~ 0*A + 0*B\nA ~~ A\nC ~~ C",
            "Arestas": ["A->B"]
        },
        "M20_Single_B_to_A_Ind_C": {
            # B é raiz (~~B). C é isolado (~~C).
            "formula": "A ~ B\nC ~ 0*B + 0*A\nB ~~ B\nC ~~ C",
            "Arestas": ["B->A"]
        },

        # Par B->C (A isolado)
        "M21_Single_B_to_C_Ind_A": {
            # B é raiz (~~B). A é isolado (~~A).
            "formula": "C ~ B\nA ~ 0*B + 0*C\nB ~~ B\nA ~~ A",
            "Arestas": ["B->C"]
        },
        "M22_Single_C_to_B_Ind_A": {
            # C é raiz (~~C). A é isolado (~~A).
            "formula": "B ~ C\nA ~ 0*C + 0*B\nC ~~ C\nA ~~ A",
            "Arestas": ["C->B"]
        },

        # Par A->C (B isolado)
        "M23_Single_A_to_C_Ind_B": {
            # A é raiz (~~A). B é isolado (~~B).
            "formula": "C ~ A\nB ~ 0*A + 0*C\nA ~~ A\nB ~~ B",
            "Arestas": ["A->C"]
        },
        "M24_Single_C_to_A_Ind_B": {
            # C é raiz (~~C). B é isolado (~~B).
            "formula": "A ~ C\nB ~ 0*C + 0*A\nC ~~ C\nB ~~ B",
            "Arestas": ["C->A"]
        },

        # --- CADEIAS (CHAINS): X -> Y -> Z ---
        # Apenas a Raiz precisa de variância explícita.
        "M01_Chain_A_B_C": { "formula": "B ~ A\nC ~ B\nA ~~ A", "Arestas": ["A->B", "B->C"] },
        "M02_Chain_A_C_B": { "formula": "C ~ A\nB ~ C\nA ~~ A", "Arestas": ["A->C", "C->B"] },
        "M03_Chain_B_A_C": { "formula": "A ~ B\nC ~ A\nB ~~ B", "Arestas": ["B->A", "A->C"] },
        "M04_Chain_B_C_A": { "formula": "C ~ B\nA ~ C\nB ~~ B", "Arestas": ["B->C", "C->A"] },
        "M05_Chain_C_A_B": { "formula": "A ~ C\nB ~ A\nC ~~ C", "Arestas": ["C->A", "A->B"] },
        "M06_Chain_C_B_A": { "formula": "B ~ C\nA ~ B\nC ~~ C", "Arestas": ["C->B", "B->A"] },

        # --- FORKS (CAUSA COMUM): Y <- X -> Z ---
        # A Causa Comum é exógena -> precisa de ~~.
        "M07_Fork_A_causes_BC": { "formula": "B ~ A\nC ~ A\nA ~~ A", "Arestas": ["A->B", "A->C"] },
        "M08_Fork_B_causes_AC": { "formula": "A ~ B\nC ~ B\nB ~~ B", "Arestas": ["B->A", "B->C"] },
        "M09_Fork_C_causes_AB": { "formula": "A ~ C\nB ~ C\nC ~~ C", "Arestas": ["C->A", "C->B"] },

        # --- COLISORES (COLLIDERS): Y -> X <- Z ---
        # Os dois pais são exógenos -> precisam de ~~.
        "M10_Collider_A_from_BC": { "formula": "A ~ C + B\nB ~~ 0*C\nB ~~ B\nC ~~ C", "Arestas": ["B->A", "C->A"] },
        "M11_Collider_B_from_AC": { "formula": "B ~ A + C\nA ~~ 0*C\nA ~~ A\nC ~~ C", "Arestas": ["A->B", "C->B"] },
        "M12_Collider_C_from_AB": { "formula": "C ~ B + A\nB ~~ 0*A\nB ~~ B\nA ~~ A", "Arestas": ["A->C", "B->C"] },

        # --- COMPLETOS (FULLY CONNECTED) ---
        # Apenas a Raiz absoluta é exógena -> precisa de ~~.
        "M13_Full_A_B_C": { "formula": "B ~ A\nC ~ A + B\nA ~~ A", "Arestas": ["A->B", "B->C", "A->C"] },
        "M14_Full_A_C_B": { "formula": "C ~ A\nB ~ A + C\nA ~~ A", "Arestas": ["A->C", "C->B", "A->B"] },
        "M15_Full_B_A_C": { "formula": "A ~ B\nC ~ B + A\nB ~~ B", "Arestas": ["B->A", "A->C", "B->C"] },
        "M16_Full_B_C_A": { "formula": "C ~ B\nA ~ B + C\nB ~~ B", "Arestas": ["B->C", "C->A", "B->A"] },
        "M17_Full_C_A_B": { "formula": "A ~ C\nB ~ C + A\nC ~~ C", "Arestas": ["C->A", "A->B", "C->B"] },
        "M18_Full_C_B_A": { "formula": "B ~ C\nA ~ C + B\nC ~~ C", "Arestas": ["C->B", "B->A", "C->A"] }
    }




def get_modelos_completos_5v():
    """
    Gera todos os 120 modelos SEM para DAGs totalmente conectadas 
    (10 arestas) com 5 vértices (A, B, C, D, E).
    """
    vertices = ['Cluster_1', 'Cluster_2', 'Cluster_3', 'Cluster_4', 'Cluster_5']
    modelos = {}
    
    # 5! = 120 permutações possíveis (ordens topológicas)
    permutacoes = list(itertools.permutations(vertices))
    
    for i, ordem in enumerate(permutacoes, 1):
        # Desempacotando a ordem topológica
        # v1 é a raiz absoluta (exógena), v5 é o sumidouro absoluto (endógena final)
        v1, v2, v3, v4, v5 = ordem
        
        # Construindo a fórmula SEM (estilo lavaan)
        # Cada variável é regredida em TODAS as variáveis que vêm antes dela
        linhas_formula = [
            f"{v2} ~ {v1}",
            f"{v3} ~ {v1} + {v2}",
            f"{v4} ~ {v1} + {v2} + {v3}",
            f"{v5} ~ {v1} + {v2} + {v3} + {v4}",
            f"{v1} ~~ {v1}"  # Apenas a raiz absoluta precisa de variância explícita
        ]
        formula = "\n".join(linhas_formula)
        
        # Construindo a lista de arestas para conferência/plotagem
        arestas = [
            f"{v1}->{v2}", f"{v1}->{v3}", f"{v1}->{v4}", f"{v1}->{v5}",
            f"{v2}->{v3}", f"{v2}->{v4}", f"{v2}->{v5}",
            f"{v3}->{v4}", f"{v3}->{v5}",
            f"{v4}->{v5}"
        ]
        
        # Criando a chave do dicionário (ex: M001_Full_ABCDE)
        nome_modelo = f"M{i:03d}_Full_{''.join(ordem)}"
        
        modelos[nome_modelo] = {
            "formula": formula,
            "Arestas": arestas
        }
        
    return modelos
