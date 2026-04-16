import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import os
from itertools import product
import json
import warnings
from gerar_dados import ConfiguracaoSimulacao, get_caso_real_1, get_caso_real_2, get_caso_real_3, get_caso_real_4
from modelos_candidatos import get_modelos_candidatosAB, get_modelos_candidatosABC
from selecionar_modelo import selecionar_modelo_rmsea, escolhe_modelo_vencedor_geral, calcula_media_metricas
from executar import executar_uma_rodada
warnings.filterwarnings("ignore")

                                                                                                                                                                                                                    

def rodar_simulacao(config: ConfiguracaoSimulacao):
    cenarios = list(product(
        config.modelos_grafos, 
        config.noises, 
        config.vetor_vertices, 
        config.vetor_R
    ))


    
    contadores_globais = {m: 0 for m in config.modelos_candidatos.keys()}
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    output_dir = os.path.join(base_dir, "..", "resultados", config.modelo_real["Caso"])
    os.makedirs(output_dir, exist_ok=True)

    lista_variaveis = config.modelo_real["lista_variaveis"]
   

    print(f"Iniciando simulação com {len(cenarios)} cenários distintos...")
    for (grafo, ruido, n_vert, R) in cenarios:
        dicionario_modelos_final = dict()
        dicionario_modelos_nao_selecionados = dict()

        desc_cenario = f"V={n_vert} | R={R} | noise={ruido} | graph={grafo}"

        output_p_valores = os.path.join(output_dir, grafo, "p_valores", f"{grafo}_{ruido}_V{n_vert}_R{R}")
        os.makedirs(output_p_valores, exist_ok=True)

        acc_params = {m: [] for m in config.modelos_candidatos}
        acc_metricas = {m: [] for m in config.modelos_candidatos}
        contador_cenario = {m:0 for m, v in config.modelos_candidatos.items()}


        # Loop de Monte Carlo (n_sims)
        for _ in tqdm(range(config.n_sims), desc=desc_cenario, leave=False):
            
            metricas, parametros = executar_uma_rodada(n_vert, R, ruido, grafo, config)
            if not metricas: continue

            temp_acc_metricas = {m: [v] for m, v in metricas.items()}
            
            vencedores_desta_rodada, _ = selecionar_modelo_rmsea(temp_acc_metricas, config)
            
            for modelo_vencedor in vencedores_desta_rodada.keys():
                contador_cenario[modelo_vencedor] += 1
                contadores_globais[modelo_vencedor] += 1

            for m in config.modelos_candidatos.keys():
                acc_params[m].append(parametros[m])
                acc_metricas[m].append(metricas[m])

            
        
        dicionario_final = escolhe_modelo_vencedor_geral(acc_metricas, contador_cenario, config)
        dicionario_modelos_total = calcula_media_metricas(acc_metricas, contador_cenario, config)
        
        print("#" * 80)
        print(f"\nCenário: {desc_cenario}")
        print("  > Parsimoniosos: ", [m for m in dicionario_final.keys()])


        lista_info_modelos_selecionados = []
        lista_info_modelos_nao_selecionados = []
        for modelo in config.modelos_candidatos.keys():
            pct_vitoria = contador_cenario[modelo] / config.n_sims

            dicionario_arestas = dict()
            if modelo in dicionario_final.keys():
                print(f"\n  [Análise Detalhada: {modelo}] - Venceu em {pct_vitoria:.2%} das rodadas")
                print(f"  AIC médio: {dicionario_final[modelo]["avg_AIC"]:.4f}")
                try:
                    print(f"  RMSEA médio: {dicionario_final[modelo]["avg_RMSEA"]:.4f}\n")
                except:
                    print(f" Modelo Saturado não possui RMSEA")


            df_params = pd.concat(acc_params[modelo])
            df_p_valores = pd.DataFrame()

            
            for causa in lista_variaveis:
                for efeito in lista_variaveis:
                    if causa == efeito: continue
                        
                    subset = df_params[(df_params['rval'] == causa) & (df_params['lval'] == efeito)]
                    if subset.empty: continue
                    
                    estimates = subset["Estimate"]
                    pvals = subset["p-value"]
                    df_p_valores[f"{causa}_{efeito}"] = pvals.values
                    try:
                        power = (pvals <= config.alpha).mean()
                    except:                            
                        power = "-"
                            
                    mean_est = estimates.mean()
                    ci_lo, ci_hi = np.percentile(estimates, [2.5, 97.5])

                    
                    chave = f"{causa}->{efeito}"
                    dicionario_arestas[chave] = {"mean_est":mean_est, "ci_lo":ci_lo, "ci_hi":ci_hi, "power":power}

                    if modelo in dicionario_final.keys():
                        print(f"    {chave}: Mean={mean_est:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] | Power={power}")

            nome_arquivo = f"{grafo}_{ruido}_V{n_vert}_R{R}_{modelo}.csv"
            caminho_p_valores = os.path.join(output_p_valores, nome_arquivo)
            df_p_valores.to_csv(caminho_p_valores, index=False)


      
            if modelo in dicionario_final.keys():
                dicionario_modelos_com_info_arestas_selecionado = {modelo:{"percentual_vitoria": pct_vitoria,
                                                            "metricas": dicionario_final[modelo], 
                                                            "arestas":dicionario_arestas}}
                
                lista_info_modelos_selecionados.append(dicionario_modelos_com_info_arestas_selecionado)
            else:
                dicionario_modelos_info_nao_selecionados = {modelo:{"percentual_vitoria": pct_vitoria,
                                                            "metricas": dicionario_modelos_total[modelo], 
                                                            "arestas":dicionario_arestas}}
                
                lista_info_modelos_nao_selecionados.append(dicionario_modelos_info_nao_selecionados)
                   
        id = f"{grafo}_{ruido}_V{n_vert}_R{R}"
        dicionario_modelos_final[id] = lista_info_modelos_selecionados
        dicionario_modelos_nao_selecionados[id] = lista_info_modelos_nao_selecionados

        path_modelos = os.path.join(output_dir, grafo, "modelos_selecionados.json")
        path_modelos_nao_selecionados = os.path.join(output_dir, grafo, "modelos_nao_selecionados.json")
        
        # 1. Ler o conteúdo atual do arquivo (se ele existir)
        for path in [path_modelos, path_modelos_nao_selecionados]:
            dados_completos = {}
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding='utf-8') as arquivo:
                        dados_completos = json.load(arquivo)
                except json.JSONDecodeError:
                    # Se o arquivo estiver vazio ou corrompido, começamos um novo
                    dados_completos = {}

            # 2. Adicionar/Atualizar o novo cenário no dicionário carregado
            # O .update() mescla o novo resultado dentro do dicionário existente
            if "modelos_selecionados" in path:
                dados_completos.update(dicionario_modelos_final)
            else:
                dados_completos.update(dicionario_modelos_nao_selecionados)

            # 3. Reescrever o arquivo inteiro com o modo "w" (write)
            with open(path, "w", encoding='utf-8') as arquivo:
                json.dump(dados_completos, arquivo, indent=4, ensure_ascii=False)
          


if __name__ == "__main__":

    """CONFIG = ConfiguracaoSimulacao(
    n_sims=1000,
    noises=["uniform", "gaussian", "student_t", "chi_square", "F"],
    vetor_R=[50, 100, 150, 200],
    vetor_vertices=[100, 200, 500, 1000],
    modelos_grafos=["erdos_renyi"],
    modelo_real=get_caso_real_3(),
    modelos_candidatos=get_modelos_candidatosABC(),
    tol_rmsea=0.08)"""

    CONFIG = ConfiguracaoSimulacao(
    n_sims=10,
    noises=["F"],
    vetor_R=[50],
    vetor_vertices=[1000],
    modelos_grafos=["watts_strogatz"],
    modelo_real=get_caso_real_3(),
    modelos_candidatos=get_modelos_candidatosABC(),
    tol_rmsea=0.08)



    rodar_simulacao(CONFIG)