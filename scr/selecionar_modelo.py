from typing import Dict, List, Tuple
import numpy as np
from gerar_dados import ConfiguracaoSimulacao



def selecionar_modelo_rmsea(acc_metricas: Dict[str, List[List[float]]], config: ConfiguracaoSimulacao) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List]]:

    """Seleciona modelos com base apenas no RMSEA. Um modelo é selecionado se ele possuir um RMSEA menor ou igual a 0.08, 
    caso não haja modelo nesta condição, os modelos saturados são selecionados.
   
    OBS: Se o modelo for saturado, o RMSEA será infinito.

    Args:
        acc_metricas (Dict[str, List[float]]): Dicionário com metricas, onde chave é nome do modelo e os valores são listas contendo as métricas.
        acc_parametros: (Dict[str, List[pd.DataFrame]]): Dicionário contendo os parâmetros das arestas de cada modelo.
        config (ConfiguracaoSimulacao): variável com as configurações da simulação.

    Returns:
        Dict[str, Dict[str, float]]: Estrutura de dados contendo os nomes dos modelos selecionados como chave e um dicionário de métricas como valores.

    Exemple:
        >> selecionar_modelo_rmsea(modelos, config)
        {
        "modelo1":{"DoF":..., "avg_AIC":..., "avg_RMSEA":..., ""avg_CFI"":..., "avg_TLI":..., "avg_chi_p_value":...},
        "modelo2":{"DoF":..., "avg_AIC":..., "avg_RMSEA":..., ""avg_CFI"":..., "avg_TLI":..., "avg_chi_p_value":...}
        }
    """

    NOMES_METRICAS = ["DoF",
                "avg_AIC", 
                "avg_BIC",
                "avg_RMSEA",
                "avg_CFI",
                "avg_TLI", 
                "avg_chi_p_value"]
    
    INDEX = 3

    medias_por_modelos = {m:(np.mean(v, axis=0)).tolist() for m, v in acc_metricas.items()}
    
    modelos_selecionados = dict()
    modelos_nao_saturados = []

    rmses = [v[INDEX] for m, v in medias_por_modelos.items()]
    min_rmse = min(rmses)

    if min_rmse <= config.tol_rmsea:
        modelos_selecionados = {m:dict(zip(NOMES_METRICAS, v)) for m, v in medias_por_modelos.items() if abs(min_rmse - v[INDEX]) <= 1e-4}
        modelos_nao_saturados = modelos_selecionados.keys()
    else:
        modelos_selecionados = {m:dict(zip(NOMES_METRICAS, v)) for m, v in medias_por_modelos.items() if np.isinf(v[INDEX]) or np.isnan(v[INDEX])}
    
    ic_metricas_por_modelos = {m:(np.percentile(v, [2.5, 97.5], axis=0)) for m, v in acc_metricas.items() if m in modelos_nao_saturados}
    return (modelos_selecionados, ic_metricas_por_modelos)
    

def escolhe_modelo_vencedor_geral(metricas_acumuladas, contador, config):
    NOMES_METRICAS = ["DoF",
                "avg_AIC", 
                "avg_BIC",
                "avg_RMSEA",
                "avg_CFI",
                "avg_TLI", 
                "avg_chi_p_value"
                ]
    
    
    dicionario_final = {}
    maximo = max(contador.values())/config.n_sims
    modelos_selecionados = {m:v/config.n_sims for m,v in contador.items() if maximo - v/config.n_sims <= 1e-4}
    for m, v in modelos_selecionados.items():
        metricas_media = np.mean(metricas_acumuladas[m], axis=0).tolist()
        metricas_media = metricas_media
        dicionario_metrica = dict(zip(NOMES_METRICAS, metricas_media))
        dicionario_final[m] = dicionario_metrica
    return dicionario_final


def calcula_media_metricas(acc_metricas, contador, config):
    NOMES_METRICAS = ["DoF",
                "avg_AIC", 
                "avg_BIC",
                "avg_RMSEA",
                "avg_CFI",
                "avg_TLI", 
                "avg_chi_p_value",
                "pct"]
    
    dicionario_final = dict()
    for m, _ in acc_metricas.items():
        metricas_media = np.mean(acc_metricas[m], axis=0).tolist()
        dicionario_final[m] = dict(zip(NOMES_METRICAS, metricas_media))
    return dicionario_final




        



