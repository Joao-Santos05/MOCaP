"""
MOCaP - Modelo Computacional para Otimização e Análise Causal de Preços no Varejo
Módulo: etl_genetico.py
Fase 1: Pipeline de ETL e Esqueleto do Algoritmo Genético (DEAP)
"""

from __future__ import annotations

import os
import random
import numpy as np
import pandas as pd
from deap import base, creator, tools

# Importação do mock nutricional e constantes de restrição
from nutricao_mock import (
    TABELA_NUTRICIONAL,
    META_CALORIAS,
    META_PROTEINAS,
    LOW_BOUNDS,
    UP_BOUNDS,
    TETO_CALORIAS_VAZIAS,
    PRODUTOS_CALORIAS_VAZIAS,
    RAZAO_ARROZ_FEIJAO,
)

# Lista ordenada de produtos para correspondência biunívoca gene <-> produto
PRODUTOS = list(TABELA_NUTRICIONAL.keys())


# =====================================================================
# 1. Pipeline de ETL (Extração, Transformação e Carga)
# =====================================================================
def identificar_estabelecimentos_qualificados(
    df_pivot_bruto: pd.DataFrame,
    threshold_ausencia: float = 0.5
) -> dict[str, list[str]]:
    """
    Identifica para cada produto os estabelecimentos com histórico consistente de coleta,
    ou seja, cuja taxa de ausência (NaN no pivot bruto antes da interpolação) seja menor
    ou igual ao threshold definido (ex: 50%).
    Evita que séries sintéticas (quase 100% interpoladas) desalinhem os modelos de ML.
    """
    estabs_qualificados = {}
    estabs = df_pivot_bruto.index.get_level_values("Estab").unique()
    for produto in PRODUTOS:
        validos = []
        for e in estabs:
            serie = df_pivot_bruto.xs(e, level="Estab")[produto]
            if serie.isna().mean() <= threshold_ausencia:
                validos.append(e)
        estabs_qualificados[produto] = validos
    return estabs_qualificados


def carregar_dados_icb(
    caminho_arquivo: str = None,
    retornar_qualidade: bool = False,
    threshold_qualidade: float = 0.5
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Carrega a aba ACUMUL_DESAGREG da planilha do ICB, filtra os 10 produtos
    da cesta nutricional, aplica filtro de outliers extremos (PPK >= R$ 1,00),
    calcula a média do PPK por mês/estabelecimento/produto, pivota o DataFrame
    e trata os valores ausentes de forma isolada por estabelecimento (prevenindo data leakage).

    Parâmetros:
        caminho_arquivo (str, opcional): Caminho do arquivo Excel.
        retornar_qualidade (bool): Se True, retorna também o dicionário com os
                                   estabelecimentos qualificados por produto (<= threshold_qualidade de NaNs).
        threshold_qualidade (float): Limite máximo de ausência histórica admitido (padrão: 0.5 = 50%).

    Retorno:
        pd.DataFrame ou tuple[pd.DataFrame, dict[str, list[str]]]
    """
    if caminho_arquivo is None:
        # Busca o arquivo no diretório do script ou na pasta de trabalho
        possiveis_caminhos = [
            os.path.join(os.path.dirname(__file__), "ICB_DADOS_2026.xlsx"),
            os.path.join(os.path.dirname(__file__), "MOCaP", "ICB_DADOS_2026.xlsx"),
            "ICB_DADOS_2026.xlsx",
            os.path.join("MOCaP", "ICB_DADOS_2026.xlsx")
        ]
        for p in possiveis_caminhos:
            if os.path.exists(p):
                caminho_arquivo = p
                break
        if caminho_arquivo is None:
            raise FileNotFoundError("Arquivo 'ICB_DADOS_2026.xlsx' não encontrado nos caminhos padrão.")

    print(f"[ETL] Lendo base de dados em: {caminho_arquivo}")
    df_raw = pd.read_excel(caminho_arquivo, sheet_name="ACUMUL_DESAGREG")

    # 1. Filtragem para os 10 produtos e remoção de outliers extremos de digitação (ex: PPK < R$ 1,00/kg)
    # Ex: registros com PPK = R$ 0,10/kg decorrentes de erros de digitação em Coxão Mole e Pão Francês
    df_filtrado = df_raw[(df_raw["Produto"].isin(PRODUTOS)) & (df_raw["PPK"] >= 1.0)].copy()

    # 2. Agrupamento: média do preço por quilo (PPK) por data, estabelecimento e produto
    df_grp = df_filtrado.groupby(["Mes_Ano_Data", "Estab", "Produto"], as_index=False)["PPK"].mean()

    # 3. Pivot table: Produtos nas colunas, Mes_Ano_Data e Estab como índices
    df_pivot = df_grp.pivot_table(index=["Mes_Ano_Data", "Estab"], columns="Produto", values="PPK")

    # Reordenar colunas exatamente na ordem da TABELA_NUTRICIONAL
    df_pivot = df_pivot[PRODUTOS]

    # Identificar estabelecimentos qualificados ANTES da interpolação
    estabs_qualificados = identificar_estabelecimentos_qualificados(
        df_pivot, threshold_ausencia=threshold_qualidade
    )

    # 4. Tratamento de NaNs SEM Data Leakage:
    # A interpolação temporal deve ser feita isolada por estabelecimento,
    # garantindo que preços de um supermercado não vazem para outro.
    df_interp = df_pivot.groupby(level="Estab", group_keys=False).apply(lambda x: x.ffill().bfill())

    # Para casos anômalos onde um estabelecimento NUNCA coletou um determinado produto
    # em nenhuma das suas datas (ex: lojas de conveniência que não vendem pão francês),
    # preenchemos os NaNs residuais com a média de mercado daquele produto no mesmo mês.
    if df_interp.isnull().sum().sum() > 0:
        df_interp = df_interp.groupby(level="Mes_Ano_Data", group_keys=False).apply(
            lambda x: x.fillna(x.mean())
        )

    # Fallback de segurança (caso algum mês inteiro não tenha coleta de algum item)
    if df_interp.isnull().sum().sum() > 0:
        df_interp = df_interp.bfill().ffill()

    if retornar_qualidade:
        return df_interp, estabs_qualificados

    return df_interp


# =====================================================================
# 2. Função de Avaliação com Penalidade Adaptativa Progressiva
# =====================================================================
def calcular_metricas_nutricionais(
    individuo: list[int],
    meta_calorias: float = META_CALORIAS,
    meta_proteinas: float = META_PROTEINAS
) -> dict:
    """
    Calcula os totais nutricionais de uma cesta e o déficit nutricional relativo
    normalizado (delta_nutricional), que acumula as violações das 4 regras:
      1. Meta calórica mínima:   calorias >= meta_calorias
      2. Meta proteica mínima:   proteinas >= meta_proteinas
      3. Teto de calorias vazias: (cal_Açúcar + cal_Óleo) <= 25% das calorias totais
      4. Proporção cultural:     kg_Arroz >= 0,5 * kg_Feijão
    """
    calorias_total = 0.0
    proteinas_total = 0.0
    calorias_vazias = 0.0
    for i, produto in enumerate(PRODUTOS):
        info = TABELA_NUTRICIONAL[produto]
        calorias_total += individuo[i] * info["calorias"]
        proteinas_total += individuo[i] * info["proteina"]
        if produto in PRODUTOS_CALORIAS_VAZIAS:
            calorias_vazias += individuo[i] * info["calorias"]

    deficit_cal = max(0.0, float(meta_calorias - calorias_total))
    deficit_prot = max(0.0, float(meta_proteinas - proteinas_total))

    # Excesso de calorias vazias relativo ao teto permitido
    teto_vazias = TETO_CALORIAS_VAZIAS * calorias_total
    excesso_vazias = max(0.0, calorias_vazias - teto_vazias)
    rel_vazias = excesso_vazias / calorias_total if calorias_total > 0 else (1.0 if calorias_vazias > 0 else 0.0)

    # Déficit da proporção cultural Arroz >= 0,5 * Feijão
    kg_arroz = individuo[PRODUTOS.index("Arroz")]
    kg_feijao = individuo[PRODUTOS.index("Feijão")]
    minimo_arroz = RAZAO_ARROZ_FEIJAO * kg_feijao
    rel_arroz = max(0.0, minimo_arroz - kg_arroz) / minimo_arroz if minimo_arroz > 0 else 0.0

    delta_nutricional = (
        (deficit_cal / meta_calorias if meta_calorias > 0 else 0.0)
        + (deficit_prot / meta_proteinas if meta_proteinas > 0 else 0.0)
        + rel_vazias
        + rel_arroz
    )

    return {
        "calorias": calorias_total,
        "proteinas": proteinas_total,
        "pct_calorias_vazias": (calorias_vazias / calorias_total) if calorias_total > 0 else 0.0,
        "delta_nutricional": delta_nutricional,
        "valido": delta_nutricional <= 0.0,
    }


def avaliar_cesta(
    individuo: list[int],
    precos_projetados: dict,
    geracao: int = None,
    ngen: int = None,
    p_min: float = 500.0,
    p_max: float = 10000.0,
    gama: float = 2.0,
    retornar_detalhes: bool = False,
    meta_calorias: float = META_CALORIAS,
    meta_proteinas: float = META_PROTEINAS
) -> tuple[float] | tuple[tuple[float], dict]:
    """
    Função de avaliação de fitness para o DEAP com suporte a Penalidade Adaptativa.
    Calcula o custo total da cesta e penaliza caso as restrições nutricionais
    (calorias, proteínas, teto de calorias vazias e proporção Arroz/Feijão)
    não sejam atendidas.

    Mecânica de Despacho (Retrocompatibilidade):
      - Se geracao is None ou ngen is None: Aplica a penalidade estática clássica (+10.000).
      - Se geracao e ngen informados: Aplica a penalidade adaptativa progressiva:
          P(q, g) = [p_min + (p_max - p_min) * (g / ngen)^gama] * (1 + delta_nutricional)

    Parâmetros:
        individuo (list[int]): Vetor de 10 inteiros indicando KGs de cada produto
                               na ordem de PRODUTOS.
        precos_projetados (dict): Mapeamento {nome_produto: preco_por_kg}.
        geracao (int, opcional): Geração atual g (0 a ngen).
        ngen (int, opcional): Número máximo de gerações.
        p_min (float): Penalidade mínima na geração inicial (padrão: R$ 500,00).
        p_max (float): Penalidade máxima na geração final (padrão: R$ 10.000,00).
        gama (float): Expoente de aceleração temporal (padrão: 2.0).
        retornar_detalhes (bool): Se True, retorna também métricas detalhadas.
        meta_calorias (float): Meta calórica mensal mínima (padrão: 60.000 kcal).
        meta_proteinas (float): Meta proteica mensal mínima (padrão: 1.500 g).

    Retorno:
        tuple[float] ou tuple[tuple[float], dict]
    """
    custo_total = 0.0
    for i, produto in enumerate(PRODUTOS):
        custo_total += individuo[i] * precos_projetados.get(produto, 0.0)

    nutri = calcular_metricas_nutricionais(individuo, meta_calorias, meta_proteinas)
    delta_nutricional = nutri["delta_nutricional"]
    violou = not nutri["valido"]

    if violou:
        if geracao is None or ngen is None:
            # Modo Legado (Fase 1): Penalidade externa estática
            penalidade = 10000.0
        else:
            # Modo Adaptativo Progressivo (Fase 7.3)
            progresso = min(1.0, max(0.0, float(geracao) / max(1.0, float(ngen))))
            p_base = p_min + (p_max - p_min) * (progresso ** gama)
            penalidade = p_base * (1.0 + delta_nutricional)
        custo_fitness = custo_total + penalidade
    else:
        penalidade = 0.0
        custo_fitness = custo_total

    if retornar_detalhes:
        detalhes = {
            "custo_real": custo_total,
            "penalidade": penalidade,
            "calorias": nutri["calorias"],
            "proteinas": nutri["proteinas"],
            "pct_calorias_vazias": nutri["pct_calorias_vazias"],
            "delta_nutricional": delta_nutricional,
            "valido": not violou
        }
        return (custo_fitness,), detalhes

    return (custo_fitness,)


# =====================================================================
# 3. Setup do DEAP (Algoritmo Genético)
# =====================================================================
# Configuração do Creator: Minimização de Custo (peso -1.0)
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


def gerar_genes_heterogeneos() -> list[int]:
    """Gera um vetor de 10 genes, cada um sorteado em [LOW_BOUNDS_i, UP_BOUNDS_i]."""
    return [random.randint(lo, up) for lo, up in zip(LOW_BOUNDS, UP_BOUNDS)]


def configurar_toolbox(
    precos_projetados: dict = None,
    meta_calorias: float = META_CALORIAS,
    meta_proteinas: float = META_PROTEINAS
) -> base.Toolbox:
    """
    Configura e retorna o Toolbox do DEAP com todos os operadores genéticos
    devidamente registrados:
      - Genes: inteiros heterogêneos em [LOW_BOUNDS_i, UP_BOUNDS_i] (LIMITES_PRODUTOS)
      - Cruzamento: cxTwoPoint
      - Mutação: mutUniformInt(low=LOW_BOUNDS, up=UP_BOUNDS, indpb=0.2)
      - Seleção: selTournament(tournsize=3)
    """
    toolbox = base.Toolbox()

    # Indivíduo: lista de 10 genes com domínio individual por produto
    toolbox.register("individual", tools.initIterate, creator.Individual, gerar_genes_heterogeneos)

    # População
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operadores Genéticos
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=LOW_BOUNDS, up=UP_BOUNDS, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Função de Avaliação
    if precos_projetados is not None:
        toolbox.register(
            "evaluate", avaliar_cesta, precos_projetados=precos_projetados,
            meta_calorias=meta_calorias, meta_proteinas=meta_proteinas
        )

    return toolbox


# =====================================================================
# 4. Loop Evolutivo com Penalidade Adaptativa (Fase 7.3)
# =====================================================================
def otimizar_cesta_genetica(
    precos_projetados: dict,
    n_populacao: int = 100,
    n_geracoes: int = 50,
    prob_cruzamento: float = 0.7,
    prob_mutacao: float = 0.2,
    usar_penalidade_adaptativa: bool = True,
    gama: float = 2.0,
    seed: int = 42,
    verbose: bool = True,
    meta_calorias: float = META_CALORIAS,
    meta_proteinas: float = META_PROTEINAS
) -> dict:
    """
    Executa a evolução heurística do Algoritmo Genético no DEAP para encontrar
    a cesta básica de menor custo financeiro que atenda rigorosamente às metas nutricionais.

    Implementa os requisitos da Fase 7.3:
      - Penalidade adaptativa progressiva P(q, g) = P_base(g) * (1 + delta_nutricional).
      - Filtro de reavaliação eficiente: a cada geração g, apenas indivíduos inviáveis
        (delta_nutricional > 0) têm seu fitness recalculado, economizando ciclos de CPU.
      - Coleta da série temporal do déficit nutricional relativo (delta_nutricional)
        e do fitness para renderização de gráficos de convergência no Streamlit.

    Retorno:
        dict contendo:
            - melhor_individuo (list[int]): Vetor de 10 KGs
            - cesta_kg (dict): Mapeamento {produto: kg}
            - custo_total (float): Custo financeiro em R$
            - calorias_totais (float): Ingestão calórica obtida
            - proteinas_totais (float): Ingestão proteica obtida
            - valido (bool): True se atendeu a todas as restrições
            - delta_nutricional_final (float): Déficit final (0.0 se válido)
            - historico_fitness_min (list[float]): Melhor fitness por geração
            - historico_fitness_med (list[float]): Fitness médio por geração
            - historico_deficit (list[float]): Série temporal de delta_nutricional
            - historico_custo_real (list[float]): Custo financeiro da melhor solução
            - historico_calorias (list[float]): Calorias da melhor solução
            - historico_proteinas (list[float]): Proteínas da melhor solução
            - historico_calorias_vazias (list[float]): % de calorias vazias da melhor solução
            - pct_calorias_vazias (float): % de calorias vazias da solução final
            - meta_calorias / meta_proteinas (float): metas utilizadas
    """
    random.seed(seed)
    np.random.seed(seed)

    metas = {"meta_calorias": meta_calorias, "meta_proteinas": meta_proteinas}

    def _avaliar(ind, geracao=None, detalhes=False):
        if usar_penalidade_adaptativa or detalhes:
            return avaliar_cesta(
                ind, precos_projetados, geracao=geracao, ngen=n_geracoes, gama=gama,
                retornar_detalhes=detalhes, **metas
            )
        return avaliar_cesta(ind, precos_projetados, **metas)

    toolbox = configurar_toolbox(precos_projetados, **metas)
    populacao = toolbox.population(n=n_populacao)
    hof = tools.HallOfFame(1)

    # Avaliação inicial (Geração 0)
    for ind in populacao:
        ind.fitness.values = _avaliar(ind, geracao=0)

    hof.update(populacao)

    # Rastreamento de métricas para visualização
    historico_fitness_min = []
    historico_fitness_med = []
    historico_deficit = []
    historico_calorias = []
    historico_proteinas = []
    historico_custo_real = []
    historico_calorias_vazias = []

    if verbose:
        print(f"\n[GA] Iniciando otimização evolutiva: {n_populacao} indivíduos, {n_geracoes} gerações.")
        print(f"[GA] Modo: {'Penalidade Adaptativa (Fase 7.3)' if usar_penalidade_adaptativa else 'Penalidade Estática'}")
        print(f"[GA] Metas: {meta_calorias:,.0f} kcal | {meta_proteinas:,.0f} g proteína")

    for g in range(n_geracoes + 1):
        melhor_ind = hof[0]
        _, detalhes_melhor = _avaliar(melhor_ind, geracao=g, detalhes=True)

        fit_min = float(melhor_ind.fitness.values[0])
        fit_med = float(np.mean([ind.fitness.values[0] for ind in populacao]))
        delta_nutr = float(detalhes_melhor["delta_nutricional"])

        historico_fitness_min.append(fit_min)
        historico_fitness_med.append(fit_med)
        historico_deficit.append(delta_nutr)
        historico_calorias.append(float(detalhes_melhor["calorias"]))
        historico_proteinas.append(float(detalhes_melhor["proteinas"]))
        historico_custo_real.append(float(detalhes_melhor["custo_real"]))
        historico_calorias_vazias.append(float(detalhes_melhor["pct_calorias_vazias"]))

        if verbose and (g % 10 == 0 or g == n_geracoes):
            status = "VÁLIDA" if detalhes_melhor["valido"] else f"Déficit: {delta_nutr*100:.1f}%"
            print(f"   Gen {g:02d}/{n_geracoes} | Melhor Fitness: R$ {fit_min:8.2f} | Custo Real: R$ {detalhes_melhor['custo_real']:7.2f} | Status: {status}")

        if g == n_geracoes:
            break

        # 1. Seleção por Torneio
        offspring = toolbox.select(populacao, len(populacao))
        offspring = list(map(toolbox.clone, offspring))

        # 2. Cruzamento (cxTwoPoint)
        for i in range(1, len(offspring), 2):
            if random.random() < prob_cruzamento:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        # 3. Mutação (mutUniformInt)
        for ind in offspring:
            if random.random() < prob_mutacao:
                toolbox.mutate(ind)
                del ind.fitness.values

        # 4. Avaliação dos Descendentes Modificados
        proxima_geracao = g + 1
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = _avaliar(ind, geracao=proxima_geracao)

        # 5. Filtro de Reavaliação Eficiente EXCLUSIVO para Inviáveis
        # Indivíduos que não sofreram cruzamento/mutação mantiveram o fitness da geração g.
        # Se forem inviáveis (qualquer uma das 4 regras), sua penalidade deve aumentar
        # para refletir a geração g+1. Viáveis têm penalidade 0 em qualquer geração.
        if usar_penalidade_adaptativa:
            for ind in offspring:
                if ind.fitness.valid and not calcular_metricas_nutricionais(ind, **metas)["valido"]:
                    ind.fitness.values = _avaliar(ind, geracao=proxima_geracao)

        # 6. Atualização da População e Elitismo
        populacao[:] = offspring
        # A penalidade do campeão inviável também deve refletir a geração corrente
        if usar_penalidade_adaptativa and len(hof) and not calcular_metricas_nutricionais(hof[0], **metas)["valido"]:
            hof[0].fitness.values = _avaliar(hof[0], geracao=proxima_geracao)
        hof.update(populacao)

    # Detalhes Finais da Melhor Solução
    melhor_solucao = hof[0]
    _, detalhes_finais = _avaliar(melhor_solucao, geracao=n_geracoes, detalhes=True)

    return {
        "melhor_individuo": list(melhor_solucao),
        "cesta_kg": {PRODUTOS[i]: int(melhor_solucao[i]) for i in range(len(PRODUTOS))},
        "custo_total": round(detalhes_finais["custo_real"], 2),
        "calorias_totais": round(detalhes_finais["calorias"], 1),
        "proteinas_totais": round(detalhes_finais["proteinas"], 1),
        "pct_calorias_vazias": round(detalhes_finais["pct_calorias_vazias"], 4),
        "valido": bool(detalhes_finais["valido"]),
        "delta_nutricional_final": round(detalhes_finais["delta_nutricional"], 4),
        "historico_fitness_min": historico_fitness_min,
        "historico_fitness_med": historico_fitness_med,
        "historico_deficit": historico_deficit,
        "historico_custo_real": historico_custo_real,
        "historico_calorias": historico_calorias,
        "historico_proteinas": historico_proteinas,
        "historico_calorias_vazias": historico_calorias_vazias,
        "n_geracoes": n_geracoes,
        "n_populacao": n_populacao,
        "meta_calorias": meta_calorias,
        "meta_proteinas": meta_proteinas
    }


# =====================================================================
# 5. Execução de Testes e Validação Embutida
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MOCaP - Teste e Validação da Fase 1: ETL e Setup DEAP")
    print("=" * 70)

    # 1. Execução do ETL
    df_pivotado = carregar_dados_icb()

    print("\n--- Informações do DataFrame Pivotado (.info()) ---")
    df_pivotado.info()

    print("\n--- Primeiras Linhas (.head()) ---")
    print(df_pivotado.head())

    # Validação obrigatória de dados faltantes (Passo 4)
    total_nans = df_pivotado.isnull().sum().sum()
    print(f"\nTotal de valores nulos após pipeline: {total_nans}")
    assert total_nans == 0, f"Erro: DataFrame contém {total_nans} valores nulos!"
    print(">> [ASSERT OK] df.isnull().sum().sum() == 0 verificado com sucesso!")

    # 2. Configuração de Preços Projetados Mock para Teste
    precos_mock = {
        'Carne Bovina Coxão Mole': 45.0,
        'Arroz': 6.0,
        'Feijão': 8.5,
        'Manteiga': 55.0,
        'Pão Francês': 18.0,
        'Farinha de Trigo': 5.5,
        'Açúcar': 4.5,
        'Óleo': 8.0,
        'Café': 35.0,
        'Banana Prata': 7.0
    }

    toolbox = configurar_toolbox(precos_mock)

    # 3. Teste do Indivíduo Mock [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    ind_valido = creator.Individual([5] * 10)
    fitness_valido = avaliar_cesta(ind_valido, precos_mock)

    # Cálculo analítico do custo sem penalidade para verificação:
    custo_base_valido = sum(5 * precos_mock[p] for p in PRODUTOS)
    calorias_valido = sum(5 * TABELA_NUTRICIONAL[p]['calorias'] for p in PRODUTOS)
    proteinas_valido = sum(5 * TABELA_NUTRICIONAL[p]['proteina'] for p in PRODUTOS)

    print("\n--- Teste 1: Indivíduo Mock [5]*10 ---")
    print(f"Indivíduo: {ind_valido}")
    print(f"Calorias totais: {calorias_valido:.0f} kcal (Meta: {META_CALORIAS})")
    print(f"Proteínas totais: {proteinas_valido:.0f} g (Meta: {META_PROTEINAS})")
    print(f"Custo base: R$ {custo_base_valido:.2f}")
    print(f"Fitness retornado: {fitness_valido[0]:.2f}")
    penalizado_valido = fitness_valido[0] >= 10000.0
    print(f"Houve penalidade? {'SIM (+10.000)' if penalizado_valido else 'NÃO (Cesta Válida)'}")

    # 4. Teste 2: Indivíduo Insuficiente para Validar Aplicação de Penalidade
    ind_insuficiente = creator.Individual([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    fitness_insuficiente = avaliar_cesta(ind_insuficiente, precos_mock)
    calorias_insuficiente = TABELA_NUTRICIONAL['Carne Bovina Coxão Mole']['calorias']
    proteinas_insuficiente = TABELA_NUTRICIONAL['Carne Bovina Coxão Mole']['proteina']
    custo_base_insuficiente = precos_mock['Carne Bovina Coxão Mole']

    print("\n--- Teste 2: Indivíduo Insuficiente [1, 0, ..., 0] ---")
    print(f"Indivíduo: {ind_insuficiente}")
    print(f"Calorias totais: {calorias_insuficiente} kcal (Meta: {META_CALORIAS})")
    print(f"Proteínas totais: {proteinas_insuficiente} g (Meta: {META_PROTEINAS})")
    print(f"Custo base: R$ {custo_base_insuficiente:.2f}")
    print(f"Fitness retornado: {fitness_insuficiente[0]:.2f}")
    penalizado_insuficiente = fitness_insuficiente[0] >= 10000.0
    print(f"Houve penalidade? {'SIM (+10.000)' if penalizado_insuficiente else 'NÃO'}")
    assert penalizado_insuficiente, "Erro: Indivíduo insuficiente deveria ter sido penalizado!"

    # 5. Validação dos Operadores Genéticos do DEAP
    print("\n--- Teste 3: Validação dos Operadores Genéticos (DEAP) ---")
    populacao = toolbox.population(n=10)
    print(f"População inicial gerada com sucesso: {len(populacao)} indivíduos.")
    print(f"Exemplo de indivíduo gerado aleatoriamente (0-20 kg): {populacao[0]}")

    # Teste de Cruzamento (cxTwoPoint)
    filho1, filho2 = toolbox.mate(toolbox.clone(populacao[0]), toolbox.clone(populacao[1]))
    print(f"Cruzamento executado: Filho 1 = {filho1}, Filho 2 = {filho2}")

    # Teste de Mutação (mutUniformInt)
    mutante, = toolbox.mutate(toolbox.clone(populacao[0]))
    print(f"Mutação executada: Mutante = {mutante}")

    # Teste de Seleção (selTournament)
    for ind in populacao:
        ind.fitness.values = avaliar_cesta(ind, precos_mock)
    selecionados = toolbox.select(populacao, k=3)
    print(f"Seleção por torneio executada: {len(selecionados)} indivíduos selecionados.")
    print(f"Melhor fitness do torneio: {min(ind.fitness.values[0] for ind in selecionados):.2f}")

    # -----------------------------------------------------------------
    # Teste 4: Validação da Fase 7.3 - Penalidade Adaptativa Progressiva
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VALIDAÇÃO DA FASE 7.3: DINAMIZAÇÃO ADAPTATIVA DA PENALIDADE")
    print("=" * 70)

    # T1: Teste de Retrocompatibilidade Estrita
    print("\n--- T1: Teste de Retrocompatibilidade com Chamada Legada ---")
    fit_legado = avaliar_cesta(ind_insuficiente, precos_mock)
    print(f"Chamada avaliar_cesta(ind_insuficiente, precos_mock) sem geracao:")
    print(f"Fitness retornado: {fit_legado[0]:.2f}")
    assert fit_legado[0] == 10045.0, f"Falha de retrocompatibilidade: esperava 10045.0, obteve {fit_legado[0]}"
    print(">> [ASSERT OK] Retrocompatibilidade comprovada! Penalidade estática de +10.000 preservada.")

    # T2: Teste da Curva Matemática da Penalidade Adaptativa P(q, g)
    print("\n--- T2: Curva da Penalidade Adaptativa por Geração (NGEN=50) ---")
    print(f"{'Geração g':<12} | {'Progresso (g/50)':<18} | {'P_base(g)':<14} | {'Fitness Cesta Insuficiente':<26}")
    print("-" * 72)
    for g_test in [0, 10, 25, 40, 50]:
        fit_g = avaliar_cesta(ind_insuficiente, precos_mock, geracao=g_test, ngen=50)
        prog = (g_test / 50.0) ** 2.0
        p_base = 500.0 + (10000.0 - 500.0) * prog
        print(f"g = {g_test:<8} | {g_test/50.0:<18.2f} | R$ {p_base:<11.2f} | R$ {fit_g[0]:<23.2f}")
    print("-" * 72)

    # T3: Execução Completa da Otimização Evolutiva (Fase 7.3)
    print("\n--- T3: Execução da Otimização Evolutiva com Penalidade Adaptativa ---")
    resultado_ga = otimizar_cesta_genetica(
        precos_projetados=precos_mock,
        n_populacao=100,
        n_geracoes=50,
        prob_cruzamento=0.7,
        prob_mutacao=0.2,
        usar_penalidade_adaptativa=True,
        gama=2.0,
        seed=42,
        verbose=True
    )

    print("\n--- Histórico de Convergência Nutricional e Fitness (Fase 7.3) ---")
    print(f"{'Geração':<10} | {'Melhor Fitness (R$)':<22} | {'Custo Real (R$)':<18} | {'Déficit Relativo':<18} | {'Calorias (kcal)':<18} | {'Proteínas (g)':<14}")
    print("-" * 105)
    for g_idx in range(0, 51, 5):
        fit_val = resultado_ga["historico_fitness_min"][g_idx]
        custo_v = resultado_ga["historico_custo_real"][g_idx]
        def_v = resultado_ga["historico_deficit"][g_idx]
        cal_v = resultado_ga["historico_calorias"][g_idx]
        prot_v = resultado_ga["historico_proteinas"][g_idx]
        print(f"Gen {g_idx:02d}     | R$ {fit_val:<19.2f} | R$ {custo_v:<15.2f} | {def_v*100:<17.2f}% | {cal_v:<18.1f} | {prot_v:<14.1f}")
    print("-" * 105)

    print("\n--- Cesta Otimizada Final ---")
    print(f"Custo Financeiro Total: R$ {resultado_ga['custo_total']:.2f}")
    print(f"Calorias Totais       : {resultado_ga['calorias_totais']:.1f} kcal (Meta: {META_CALORIAS})")
    print(f"Proteínas Totais      : {resultado_ga['proteinas_totais']:.1f} g (Meta: {META_PROTEINAS})")
    print(f"Status de Viabilidade : {'100% VÁLIDA' if resultado_ga['valido'] else 'INVÁLIDA'}")
    print(f"Déficit Final         : {resultado_ga['delta_nutricional_final']}")
    print("\nComposição em Quilos por Alimento:")
    for prod, kg_qtd in resultado_ga["cesta_kg"].items():
        print(f"   - {prod:<26}: {kg_qtd:>2d} kg")

    # T4: Asserção Estrita de Não-Aceitação de Déficits Nutricionais
    assert resultado_ga["valido"] is True, "FALHA CRÍTICA: A solução final não atendeu às restrições nutricionais!"
    assert resultado_ga["calorias_totais"] >= META_CALORIAS, "FALHA: Calorias abaixo da meta!"
    assert resultado_ga["proteinas_totais"] >= META_PROTEINAS, "FALHA: Proteínas abaixo da meta!"
    assert len(resultado_ga["historico_deficit"]) == 51, "FALHA: Série temporal de déficit incompleta!"
    print("\n>> [ASSERT OK] Solução final 100% FACTÍVEL! Zero déficit proteico/calórico confirmado.")

    print("\n" + "=" * 70)
    print("TODOS OS TESTES DAS FASES 1 E 7.3 CONCLUÍDOS COM SUCESSO!")
    print("=" * 70)
