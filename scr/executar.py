import pandas as pd
import semopy
from typing import Dict, Tuple, Optional
import warnings
from gerar_dados import ConfiguracaoSimulacao, gerar_dados_AB, gerar_dados_ABC, padronizacao
warnings.filterwarnings("ignore")




def ajustar_modelo_sem(dados: pd.DataFrame, formula: str) -> Optional[Dict]:
    try:
        model = semopy.Model(formula)
        model.fit(dados)
        stats = semopy.stats.calc_stats(model)
        params = model.inspect()
        params = params[params["op"] != "~~"][["lval", "rval", "Estimate", "p-value"]]
        metrics = {
            "DoF": stats["DoF"].values[0],
            "AIC": stats["AIC"].values[0],
            "BIC": stats["BIC"].values[0],
            "RMSEA": stats["RMSEA"].values[0],
            "CFI": stats["CFI"].values[0],
            "TLI": stats["TLI"].values[0],
            "p_value": stats["chi2 p-value"].values[0],
            "params": params
        }
        return metrics

    except Exception as e:
        print(e)
        return None
    


def executar_uma_rodada(
    num_vertice: int, R: int, ruido: str, modelo_grafo: str, config: ConfiguracaoSimulacao
) -> Tuple[Dict, Dict]:
    """
    Gera dados e ajusta todos os modelos candidatos para UMA iteração.
    Retorna resultados de métricas e parâmetros.
    """
    amostra_novos_dados = True
    while amostra_novos_dados:
        # 1. Gerar Dados
        dados_brutos = config.modelo_real["dados"](
            num_vertices=num_vertice,
            R=R,
            tipo_ruido=ruido,
            modelo_gerador=modelo_grafo,
            estrutura_causal=config.modelo_real["formula"]
        )


        dados_ranked = dados_brutos.rank()
        dados_ranked_pad = pd.DataFrame(data=padronizacao(dados_ranked), columns=dados_ranked.columns)

        resultados_metrics = {}
        resultados_params = {}
        try:
            for nome, desc in config.modelos_candidatos.items():
                res = ajustar_modelo_sem(dados_ranked_pad, desc["formula"])
                resultados_metrics[nome] = [v for metrica, v in res.items() if metrica != "params"]
                resultados_params[nome] = res["params"]
            amostra_novos_dados = False

        except Exception as e:
            print(f"Erro na iteração: {e}. Gerando nova amostra...")
            amostra_novos_dados = True

    return resultados_metrics, resultados_params