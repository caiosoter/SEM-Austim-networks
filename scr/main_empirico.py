import pandas as pd
import semopy
from modelos_candidatos import get_modelos_completos_5v 
from gerar_dados import padronizacao
from sklearn.utils import resample
from tqdm import tqdm
import json
import os

def seleciona_melhor_modelo(df):
    pool = get_modelos_completos_5v()

    label_best = df["Modelo"].values[0]
    formula = pool[label_best]["formula"]
    arestas = pool[label_best]["Arestas"]
    return formula, arestas



def executar(n_sim, conjunto):
    
    if conjunto == "autismo":
        df = pd.read_csv(r"dado_autismo\ABIDE\dados\raios_espectrais\raios_espectrais_autismo.csv")
        modelos = pd.read_csv(r"simulacoes_selecao_modelos\resultados_empiricos\modelos_autismo.csv")
        formula, arestas = seleciona_melhor_modelo(modelos)

    elif conjunto == "controle":
        df = pd.read_csv(r"dado_autismo\ABIDE\dados\raios_espectrais\raios_espectrais_controle.csv")
        modelos = pd.read_csv(r"simulacoes_selecao_modelos\resultados_empiricos\modelos_controle.csv")
        formula, arestas = seleciona_melhor_modelo(modelos)

    else:
        print("Este conjunto não existe")
        return None

    resultados = {"AIC":[], "BIC":[], "arestas":{v:{"p_valores":[], "estimativa":[]} for v in arestas}}
  
    for i in tqdm(range(n_sim)):
        df_boots = resample(df, replace=True, n_samples=len(df))

        dados_ranked = df_boots.rank()
        dados_ranked_pad = pd.DataFrame(data=padronizacao(df_boots), columns=dados_ranked.columns)
        
        model = semopy.Model(formula)
        model.fit(dados_ranked_pad, obj="MLW")
        aic = semopy.stats.calc_aic(model)
        bic = semopy.stats.calc_bic(model)
        params = model.inspect()
        params = params[params["op"] != "~~"][["lval", "rval", "Estimate", "p-value"]]

        resultados["AIC"].append(aic)
        resultados["BIC"].append(bic)

        for aresta in arestas:
            split = aresta.split("->")
            rval = split[0]
            lval = split[1]
            dados_aresta = params[(params["rval"] == rval) & (params["lval"] == lval)]
            p_valor = dados_aresta["p-value"].values[0]
            estimativa = dados_aresta["Estimate"].values[0]

            resultados["arestas"][aresta]["p_valores"].append(p_valor)
            resultados["arestas"][aresta]["estimativa"].append(estimativa)

    return resultados




try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

output_dir = os.path.join(base_dir, "resultados_empiricos")
os.makedirs(output_dir, exist_ok=True)


conjunto = "autismo"
nome_arquivo = f"{conjunto}_estimativas.json"
resultados = executar(n_sim=10000, conjunto=conjunto)


caminho = os.path.join(output_dir, nome_arquivo)
with open(caminho, "w", encoding='utf-8') as arquivo:
    json.dump(resultados, arquivo, indent=4, ensure_ascii=False)


