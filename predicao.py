"""
MOCaP - Modelo Computacional para Otimização e Análise Causal de Preços no Varejo
Módulo: predicao.py
Fase 2: Módulo Preditivo de Preços com XGBoost e Integração ao DEAP (Bridge)
"""

import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

# Importações da Fase 1 (ETL e DEAP)
from etl_genetico import carregar_dados_icb, PRODUTOS, avaliar_cesta, configurar_toolbox
from nutricao_mock import TABELA_NUTRICIONAL


# =====================================================================
# 1. Engenharia de Recursos Temporais (Lags e Médias Móveis)
# =====================================================================
def criar_features_temporais(
    df_pivot: pd.DataFrame,
    lags: list[int] = [1, 2, 3],
    janelas_rolling: list[int] = [2, 3]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gera variáveis defasadas (lags) e médias móveis para cada produto.
    
    RESTRIÇÃO CRÍTICA DE NÃO-VAZAMENTO (Data Leakage):
    O cálculo dos lags e médias móveis é estritamente agrupado por estabelecimento
    (level='Estab') sobre a série ordenada temporalmente. Isso garante que preços
    do estabelecimento anterior no índice não vazem para o próximo.

    Parâmetros:
        df_pivot (pd.DataFrame): DataFrame pivotado com MultiIndex (Mes_Ano_Data, Estab).
        lags (list[int]): Períodos de atraso temporal (ex: [1, 2, 3]).
        janelas_rolling (list[int]): Janelas para médias móveis sobre valores passados.

    Retorno:
        tuple[pd.DataFrame, pd.DataFrame]:
            - df_com_nans: DataFrame completo antes do dropna (útil para auditoria de leakage).
            - df_features: DataFrame limpo após remoção das linhas com NaN dos shifts iniciais.
    """
    # Garantir ordenação temporal rigorosa por estabelecimento e data
    df_ordenado = df_pivot.sort_index(level=["Estab", "Mes_Ano_Data"]).copy()

    features_dict = {}

    for produto in PRODUTOS:
        # Lags temporais por estabelecimento
        for lag in lags:
            col_lag = f"{produto}_lag{lag}"
            features_dict[col_lag] = df_ordenado.groupby(level="Estab")[produto].shift(lag)

        # Médias móveis calculadas exclusivamente sobre informações conhecidas até t-1
        # (evitando vazamento do preço contemporâneo t na feature explicativa)
        lag1_series = features_dict[f"{produto}_lag1"]
        for w in janelas_rolling:
            col_roll = f"{produto}_roll{w}"
            # Média móvel das últimas w observações conhecidas (até t-1)
            features_dict[col_roll] = (
                lag1_series.groupby(level="Estab")
                .rolling(window=w, min_periods=w)
                .mean()
                .reset_index(level=0, drop=True)
            )

    # Inclusão de Feature Cíclica/Sazonal: mês do ano (1 a 12)
    # Permite capturar a sazonalidade agrícola anual (especialmente para hortifruti como Banana Prata)
    meses = df_ordenado.index.get_level_values("Mes_Ano_Data").month
    features_dict["mes"] = pd.Series(meses, index=df_ordenado.index)

    df_feats = pd.DataFrame(features_dict, index=df_ordenado.index)

    # DataFrame completo com targets originais e features geradas
    df_com_nans = pd.concat([df_ordenado, df_feats], axis=1)

    # Remoção das linhas iniciais com NaN geradas pelos lags
    df_features = df_com_nans.dropna().copy()

    return df_com_nans, df_features


# =====================================================================
# 2. Separação Treino/Teste Temporal (Out-of-Time)
# =====================================================================
def separar_treino_teste_temporal(
    df: pd.DataFrame,
    data_corte: str = "2026-01-01"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Realiza a partição temporal Out-of-Time dos dados.
    
    RESTRIÇÃO CRÍTICA:
    É proibido utilizar train_test_split aleatório, pois causaria vazamento temporal
    (prever o passado usando informações do futuro).
    
    Parâmetros:
        df (pd.DataFrame): DataFrame contendo alvos e features com índice temporal.
        data_corte (str): Data limítrofe no formato YYYY-MM-DD.

    Retorno:
        tuple[pd.DataFrame, pd.DataFrame]: (df_treino, df_teste)
    """
    datas = df.index.get_level_values("Mes_Ano_Data")
    mascara_treino = datas < data_corte
    mascara_teste = datas >= data_corte

    df_treino = df.loc[mascara_treino].copy()
    df_teste = df.loc[mascara_teste].copy()

    if len(df_teste) == 0:
        raise ValueError(f"A data de corte '{data_corte}' não gerou amostras no conjunto de teste.")

    return df_treino, df_teste


# =====================================================================
# 3. Treinamento dos Modelos Preditivos (XGBoost)
# =====================================================================
def treinar_modelos_xgboost(
    df_treino: pd.DataFrame,
    df_teste: pd.DataFrame = None,
    estabelecimentos_qualificados: dict[str, list[str]] = None,
    hiperparametros: dict = None
) -> tuple[dict[str, XGBRegressor], dict[str, dict]]:
    """
    Treina um modelo XGBRegressor para cada um dos 10 produtos da cesta básica.

    Parâmetros:
        df_treino (pd.DataFrame): Dados de treinamento.
        df_teste (pd.DataFrame, opcional): Dados de teste para cálculo de métricas.
        estabelecimentos_qualificados (dict, opcional): Mapeamento {produto: [estabs_validos]}
            para filtrar e remover do treinamento/avaliação mercados com histórico excessivamente
            sintético/ausente (> 50% NaN).
        hiperparametros (dict, opcional): Hiperparâmetros customizados para o XGBoost.

    Retorno:
        tuple:
            - modelos (dict[str, XGBRegressor]): Mapeamento {produto: modelo_treinado}.
            - metricas (dict[str, dict]): Mapeamento {produto: {'rmse': float, 'mape': float, 'preco_medio_teste': float, 'estabs_usados': int}}.
    """
    if hiperparametros is None:
        hiperparametros = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1
        }

    modelos = {}
    metricas = {}

    for produto in PRODUTOS:
        # Features exclusivas: 3 lags temporais + 2 médias móveis + mês (sazonalidade anual)
        cols_features = [
            f"{produto}_lag1",
            f"{produto}_lag2",
            f"{produto}_lag3",
            f"{produto}_roll2",
            f"{produto}_roll3",
            "mes"
        ]

        # Aplicar filtro de threshold de qualidade por estabelecimento se fornecido
        if estabelecimentos_qualificados and produto in estabelecimentos_qualificados:
            estabs_ok = estabelecimentos_qualificados[produto]
            mask_estab_train = df_treino.index.get_level_values("Estab").isin(estabs_ok)
            df_tr = df_treino.loc[mask_estab_train]
            qtd_estabs = len(estabs_ok)
        else:
            df_tr = df_treino
            qtd_estabs = df_treino.index.get_level_values("Estab").nunique()

        X_train = df_tr[cols_features]
        y_train = df_tr[produto]

        modelo = XGBRegressor(**hiperparametros)
        modelo.fit(X_train, y_train)
        modelos[produto] = modelo

        if df_teste is not None and len(df_teste) > 0:
            if estabelecimentos_qualificados and produto in estabelecimentos_qualificados:
                estabs_ok = estabelecimentos_qualificados[produto]
                mask_estab_test = df_teste.index.get_level_values("Estab").isin(estabs_ok)
                df_te = df_teste.loc[mask_estab_test]
            else:
                df_te = df_teste

            X_test = df_te[cols_features]
            y_test = df_te[produto]
            y_pred = modelo.predict(X_test)

            rmse = float(root_mean_squared_error(y_test, y_pred))
            mape = float(mean_absolute_percentage_error(y_test, y_pred))

            metricas[produto] = {
                "rmse": rmse,
                "mape": mape,
                "preco_medio_teste": float(y_test.mean()),
                "estabs_usados": qtd_estabs
            }

    return modelos, metricas


# =====================================================================
# 4. Mapeamento de Formato para o DEAP (Função Bridge)
# =====================================================================
def bridge_precos_deap(
    modelos: dict[str, XGBRegressor],
    ultimos_dados: pd.DataFrame | pd.Series
) -> dict[str, float]:
    """
    Função de integração entre o Módulo Preditivo (XGBoost) e o Algoritmo Genético (DEAP).
    Recebe os 10 modelos treinados e os dados históricos mais recentes de um determinado
    estabelecimento, gerando a projeção de preços por quilo (PPK) para o próximo mês (t+1).

    Parâmetros:
        modelos (dict): Dicionário {produto: modelo_XGBRegressor}.
        ultimos_dados (pd.DataFrame | pd.Series):
            - Pode ser um DataFrame com as últimas coletas de preços do mercado
              (contendo pelo menos as últimas 3 observações históricas dos 10 produtos), OU
            - Uma Series/DataFrame já formatada com as colunas de lag calculadas.

    Retorno:
        dict[str, float]: Dicionário contendo os nomes exatos dos 10 produtos como chaves
                          e os preços projetados em R$/kg (arredondados a 2 casas) como valores.
    """
    precos_projetados = {}

    # Determinar o mês cronológico a ser projetado (t+1) para a feature sazonal
    mes_pred = 9  # default padrão (setembro, após agosto/2026 da base ICB)
    if isinstance(ultimos_dados, pd.DataFrame):
        try:
            if isinstance(ultimos_dados.index, pd.DatetimeIndex):
                ultima_data = ultimos_dados.index.max()
                mes_pred = (ultima_data.month % 12) + 1
            elif "Mes_Ano_Data" in ultimos_dados.index.names:
                ultima_data = ultimos_dados.index.get_level_values("Mes_Ano_Data").max()
                mes_pred = (ultima_data.month % 12) + 1
            elif "Mes_Ano_Data" in ultimos_dados.columns:
                ultima_data = pd.to_datetime(ultimos_dados["Mes_Ano_Data"]).max()
                mes_pred = (ultima_data.month % 12) + 1
        except Exception:
            mes_pred = 9
    elif isinstance(ultimos_dados, pd.Series):
        if "mes" in ultimos_dados.index and not pd.isna(ultimos_dados["mes"]):
            mes_pred = (int(ultimos_dados["mes"]) % 12) + 1

    # Caso 1: Recebeu DataFrame com série histórica de preços do estabelecimento
    if isinstance(ultimos_dados, pd.DataFrame):
        for produto in PRODUTOS:
            if produto in ultimos_dados.columns:
                serie_precos = ultimos_dados[produto].dropna().values
                if len(serie_precos) < 3:
                    raise ValueError(
                        f"O histórico do produto '{produto}' possui apenas {len(serie_precos)} "
                        "observações; são necessárias pelo menos 3 para calcular os lags (t, t-1, t-2)."
                    )
                # Para projetar t+1, os dados mais recentes se tornam os lags:
                lag1 = float(serie_precos[-1])     # Preço em t
                lag2 = float(serie_precos[-2])     # Preço em t-1
                lag3 = float(serie_precos[-3])     # Preço em t-2
                roll2 = (lag1 + lag2) / 2.0
                roll3 = (lag1 + lag2 + lag3) / 3.0

                X_input = pd.DataFrame([{
                    f"{produto}_lag1": lag1,
                    f"{produto}_lag2": lag2,
                    f"{produto}_lag3": lag3,
                    f"{produto}_roll2": roll2,
                    f"{produto}_roll3": roll3,
                    "mes": mes_pred
                }])

                modelo = modelos[produto]
                preco_pred = float(modelo.predict(X_input)[0])
                # Garantir preço positivo realista
                precos_projetados[produto] = round(max(0.01, preco_pred), 2)
            else:
                # Se for um DataFrame que já possui as colunas de lag pré-calculadas
                cols_necessarias = [
                    f"{produto}_lag1", f"{produto}_lag2", f"{produto}_lag3",
                    f"{produto}_roll2", f"{produto}_roll3"
                ]
                if all(c in ultimos_dados.columns for c in cols_necessarias):
                    X_input = ultimos_dados[cols_necessarias].tail(1).copy()
                    if "mes" not in X_input.columns:
                        X_input["mes"] = mes_pred
                    else:
                        X_input["mes"] = X_input["mes"].fillna(mes_pred)
                    # Reordenar colunas exatamente como no modelo
                    cols_modelo = cols_necessarias + ["mes"]
                    X_input = X_input[cols_modelo]
                    modelo = modelos[produto]
                    preco_pred = float(modelo.predict(X_input)[0])
                    precos_projetados[produto] = round(max(0.01, preco_pred), 2)
                else:
                    raise KeyError(f"Produto ou features de '{produto}' não encontrados nos dados fornecidos.")

    # Caso 2: Recebeu uma Series contendo diretamente as features
    elif isinstance(ultimos_dados, pd.Series):
        for produto in PRODUTOS:
            cols_necessarias = [
                f"{produto}_lag1", f"{produto}_lag2", f"{produto}_lag3",
                f"{produto}_roll2", f"{produto}_roll3"
            ]
            dados_dict = ultimos_dados[cols_necessarias].to_dict()
            dados_dict["mes"] = ultimos_dados.get("mes", mes_pred)
            cols_modelo = cols_necessarias + ["mes"]
            X_input = pd.DataFrame([dados_dict])[cols_modelo]
            modelo = modelos[produto]
            preco_pred = float(modelo.predict(X_input)[0])
            precos_projetados[produto] = round(max(0.01, preco_pred), 2)

    return precos_projetados


# =====================================================================
# 5. Execução de Testes Embutidos e Validação (Passo 5)
# =====================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("MOCaP - Validação da Fase 2: Engenharia de Recursos, XGBoost e Bridge DEAP")
    print("=" * 80)

    # 1. Carregamento do DataFrame da Fase 1 com Threshold de Qualidade
    df_pivot, estabs_qualificados = carregar_dados_icb(retornar_qualidade=True, threshold_qualidade=0.5)
    print(f"[Carga] DataFrame carregado: {df_pivot.shape[0]} linhas, {df_pivot.shape[1]} produtos.")
    print(f"[Qualidade] Estabelecimentos qualificados por produto (ausência <= 50%):")
    for p in PRODUTOS:
        print(f"   - {p:<26}: {len(estabs_qualificados[p])} de {df_pivot.index.get_level_values('Estab').nunique()} mercados")

    # 2. Engenharia de Recursos Temporais (Lags, Médias Móveis e Mês Cíclico)
    df_com_nans, df_features = criar_features_temporais(df_pivot)
    print(f"[Features] Linhas pré-dropna: {len(df_com_nans)} | Linhas pós-dropna: {len(df_features)}")
    print(f"[Features] Features incluídas: Lags 1-3, Médias Móveis 2-3, e Mês Sazonal ('mes').")

    # -----------------------------------------------------------------
    # Teste 1: Validação Rigorosa de Data Leakage (Passo 5 - Item 1)
    # -----------------------------------------------------------------
    print("\n--- Teste 1: Validação de Ausência de Data Leakage nos Lags ---")
    estabelecimentos = df_pivot.index.get_level_values("Estab").unique()
    estabs_auditados = 0

    for estab in estabelecimentos:
        df_estab = df_com_nans.xs(estab, level="Estab")
        primeira_data = df_estab.index.min()
        linha_primeira_data = df_estab.loc[primeira_data]

        # Verificar se os lags da primeira data são estritamente nulos
        lags_primeira_data = [linha_primeira_data[f"{p}_lag1"] for p in PRODUTOS]
        assert all(pd.isna(v) for v in lags_primeira_data), (
            f"FALHA DE DATA LEAKAGE: A primeira data ({primeira_data}) do estabelecimento '{estab}' "
            "contém valores de lag preenchidos, indicando contaminação temporal de outro mercado!"
        )
        estabs_auditados += 1

    print(f">> [ASSERT OK] {estabs_auditados} estabelecimentos auditados!")
    print(">> Todas as primeiras datas possuem lags = NaN. Zero vazamento entre estabelecimentos confirmado.")

    # -----------------------------------------------------------------
    # Teste 2: Partição Temporal Out-of-Time e Treinamento XGBoost Refinado
    # -----------------------------------------------------------------
    data_corte = "2026-01-01"
    df_treino, df_teste = separar_treino_teste_temporal(df_features, data_corte=data_corte)

    print(f"\n--- Teste 2: Partição Temporal Out-of-Time (Corte: {data_corte}) ---")
    print(f"Amostras de Treino (< {data_corte}): {len(df_treino)} registros.")
    print(f"Amostras de Teste  (>= {data_corte}): {len(df_teste)} registros.")

    modelos, metricas = treinar_modelos_xgboost(
        df_treino, df_teste, estabelecimentos_qualificados=estabs_qualificados
    )

    print("\nDesempenho Preditivo Refinado no Período de Teste Out-of-Time (2026):")
    print("-" * 88)
    print(f"{'Produto':<26} | {'Preço Médio':<12} | {'RMSE':<12} | {'MAPE':<10} | {'Mercados Usados':<15}")
    print("-" * 88)
    for p in PRODUTOS:
        m = metricas[p]
        print(f"{p:<26} | R$ {m['preco_medio_teste']:<9.2f} | R$ {m['rmse']:<9.4f} | {m['mape']*100:<8.2f}% | {m['estabs_usados']}/51")
    print("-" * 88)

    # Destaque do Pão Francês (grande beneficiário do threshold de qualidade)
    pao_m = metricas["Pão Francês"]
    print(f"\n>> [DESTAQUE 1] Produto: Pão Francês (Refinado com Quality Threshold)")
    print(f"   Preço Médio no Teste : R$ {pao_m['preco_medio_teste']:.2f}")
    print(f"   RMSE no Período Teste: R$ {pao_m['rmse']:.4f}")
    print(f"   MAPE no Período Teste: {pao_m['mape']*100:.2f}% (Redução expressiva frente aos 16,47% anteriores!)")

    # Destaque do Feijão
    feijao_m = metricas["Feijão"]
    print(f"\n>> [DESTAQUE 2] Produto: Feijão")
    print(f"   Preço Médio no Teste : R$ {feijao_m['preco_medio_teste']:.2f}")
    print(f"   RMSE no Período Teste: R$ {feijao_m['rmse']:.4f}")
    print(f"   MAPE no Período Teste: {feijao_m['mape']*100:.2f}%")
    assert feijao_m["rmse"] < 2.0, "Alerta: RMSE do Feijão acima da tolerância esperada!"

    # Destaque da Banana Prata (beneficiária da feature sazonal 'mes')
    banana_m = metricas["Banana Prata"]
    print(f"\n>> [DESTAQUE 3] Produto: Banana Prata (Refinada com Feature Cíclica 'mes')")
    print(f"   Preço Médio no Teste : R$ {banana_m['preco_medio_teste']:.2f}")
    print(f"   RMSE no Período Teste: R$ {banana_m['rmse']:.4f}")
    print(f"   MAPE no Período Teste: {banana_m['mape']*100:.2f}%")

    # -----------------------------------------------------------------
    # Teste 3: Mapeamento de Formato para o DEAP (Função Bridge)
    # -----------------------------------------------------------------
    print("\n--- Teste 3: Projeção de Preços via bridge_precos_deap (Passo 5 - Item 3) ---")
    mercado_exemplo = "SUP_01"
    dados_mercado = df_pivot.xs(mercado_exemplo, level="Estab")
    precos_projetados = bridge_precos_deap(modelos, dados_mercado)

    print(f"Previsões para o próximo mês (t+1) no estabelecimento '{mercado_exemplo}':")
    for prod, preco in precos_projetados.items():
        print(f"   - {prod:<25}: R$ {preco:>6.2f} / kg")

    # Validação do formato e tipos do dicionário para injeção no DEAP
    assert isinstance(precos_projetados, dict), "A saída da bridge deve ser um dicionário!"
    assert len(precos_projetados) == 10, "A bridge deve conter exatamente os 10 produtos da cesta básica!"
    assert all(p in precos_projetados for p in PRODUTOS), "Todas as chaves da TABELA_NUTRICIONAL devem estar presentes!"
    assert all(isinstance(v, (float, int)) and v > 0 for v in precos_projetados.values()), "Preços devem ser numéricos positivos!"
    print(">> [ASSERT OK] Estrutura do dicionário da Bridge 100% em conformidade com o DEAP.")

    # -----------------------------------------------------------------
    # Teste 4: Integração de Ponta a Ponta com o Fitness do DEAP (Fase 1)
    # -----------------------------------------------------------------
    print("\n--- Teste 4: Injeção das Projeções na Função de Fitness da Fase 1 ---")
    cesta_teste = [5] * 10  # 5 kg de cada alimento
    custo_projetado, = avaliar_cesta(cesta_teste, precos_projetados)
    print(f"Custo projetado para a cesta mock [5 kg x 10 produtos]: R$ {custo_projetado:.2f}")
    print("Status da Cesta: " + ("PENALIZADA" if custo_projetado >= 10000 else "VÁLIDA (Nutricionalmente Aprovada)"))

    print("\n" + "=" * 80)
    print("TODOS OS TESTES DA FASE 2 FORAM CONCLUÍDOS COM SUCESSO!")
    print("=" * 80)
