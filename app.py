"""
MOCaP - Modelo Computacional para Otimização e Análise Causal de Preços no Varejo
Módulo: app.py
Fase Final: Dashboard Interativo (Streamlit) — Otimizador de Despensa
             Pipeline Preditivo-Evolutivo: ETL -> XGBoost (t+1) -> DEAP Adaptativo

Execução:
    streamlit run app.py
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from etl_genetico import (
    PRODUTOS,
    carregar_dados_icb,
    calcular_metricas_nutricionais,
    otimizar_cesta_genetica,
)
from nutricao_mock import (
    CESTA_PADRAO_DIEESE,
    PERFIS_NUTRICIONAIS,
    TETO_CALORIAS_VAZIAS,
    UNIDADES,
)
from predicao import (
    bridge_precos_deap,
    calcular_custo_cesta,
    criar_features_temporais,
    separar_treino_teste_temporal,
    treinar_modelos_xgboost,
)

DATA_CORTE = "2026-01-01"
MERCADO_PADRAO = "SUP_01"
PERFIL_PERSONALIZADO = "Personalizado"

st.set_page_config(
    page_title="MOCaP — Otimizador de Despensa",
    page_icon="🛒",
    layout="wide",
)


# =====================================================================
# 1. Camada de Cache (dados e modelos)
# =====================================================================
@st.cache_data(show_spinner="Carregando e tratando a base ICB (ETL)...")
def carregar_base() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """ETL da planilha ICB — executado uma única vez por sessão do servidor."""
    return carregar_dados_icb(retornar_qualidade=True, threshold_qualidade=0.5)


@st.cache_resource(show_spinner="Treinando os 10 modelos XGBoost (Out-of-Time)...")
def treinar_modelos():
    """Treina os 10 XGBRegressor uma única vez; reutilizados em todas as interações."""
    df_pivot, estabs_qualificados = carregar_base()
    _, df_features = criar_features_temporais(df_pivot)
    df_treino, df_teste = separar_treino_teste_temporal(df_features, data_corte=DATA_CORTE)
    modelos, metricas = treinar_modelos_xgboost(
        df_treino, df_teste, estabelecimentos_qualificados=estabs_qualificados
    )
    return modelos, metricas


@st.cache_data(show_spinner=False)
def projetar_precos(estab: str) -> dict[str, float]:
    """Projeção t+1 via bridge_precos_deap para o estabelecimento selecionado."""
    df_pivot, _ = carregar_base()
    modelos, _ = treinar_modelos()
    return bridge_precos_deap(modelos, df_pivot.xs(estab, level="Estab"))


def formatar_brl(valor: float) -> str:
    texto = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {texto}"


def formatar_num(valor: float, casas: int = 0) -> str:
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# =====================================================================
# 2. Carregamento inicial
# =====================================================================
df_pivot, estabs_qualificados = carregar_base()
modelos, metricas = treinar_modelos()

datas_por_estab = df_pivot.groupby(level="Estab").size()
mercados_validos = sorted(datas_por_estab[datas_por_estab >= 3].index.tolist())
ultima_data = df_pivot.index.get_level_values("Mes_Ano_Data").max()
mes_projetado = (ultima_data + pd.DateOffset(months=1)).strftime("%m/%Y")


# =====================================================================
# 3. Barra Lateral — Parâmetros
# =====================================================================
st.sidebar.title("🛒 MOCaP / CestaGen")
st.sidebar.caption("Pipeline Preditivo-Evolutivo de Otimização da Cesta Básica")

st.sidebar.header("1. Estabelecimento")
mercado = st.sidebar.selectbox(
    "Mercado",
    mercados_validos,
    index=mercados_validos.index(MERCADO_PADRAO) if MERCADO_PADRAO in mercados_validos else 0,
    help="Somente estabelecimentos com pelo menos 3 meses de histórico (exigência dos lags t-1..t-3).",
)

st.sidebar.header("2. Perfil Nutricional")
opcoes_perfis = list(PERFIS_NUTRICIONAIS.keys()) + [PERFIL_PERSONALIZADO]


def aplicar_perfil():
    perfil = st.session_state["perfil"]
    if perfil in PERFIS_NUTRICIONAIS:
        st.session_state["meta_cal"] = int(PERFIS_NUTRICIONAIS[perfil]["calorias"])
        st.session_state["meta_prot"] = int(PERFIS_NUTRICIONAIS[perfil]["proteinas"])


def marcar_personalizado():
    perfil = st.session_state["perfil"]
    if perfil in PERFIS_NUTRICIONAIS:
        ref = PERFIS_NUTRICIONAIS[perfil]
        if (st.session_state["meta_cal"], st.session_state["meta_prot"]) != (ref["calorias"], ref["proteinas"]):
            st.session_state["perfil"] = PERFIL_PERSONALIZADO


if "perfil" not in st.session_state:
    st.session_state["perfil"] = opcoes_perfis[0]
    aplicar_perfil()

perfil = st.sidebar.radio("Perfil", opcoes_perfis, key="perfil", on_change=aplicar_perfil)
if perfil in PERFIS_NUTRICIONAIS:
    st.sidebar.caption(PERFIS_NUTRICIONAIS[perfil]["descricao"])
else:
    st.sidebar.caption("Metas definidas manualmente.")

meta_calorias = st.sidebar.number_input(
    "Meta calórica (kcal/mês)", min_value=10000, max_value=200000, step=1000,
    key="meta_cal", on_change=marcar_personalizado,
)
meta_proteinas = st.sidebar.number_input(
    "Meta proteica (g/mês)", min_value=100, max_value=10000, step=50,
    key="meta_prot", on_change=marcar_personalizado,
)

st.sidebar.header("3. Algoritmo Genético")
n_populacao = st.sidebar.slider("Tamanho da população", 20, 500, 100, step=10)
n_geracoes = st.sidebar.slider("Número de gerações", 10, 300, 50, step=10)
gama = st.sidebar.slider("Expoente adaptativo γ", 0.5, 5.0, 2.0, step=0.1,
                         help="Controla a aceleração da penalidade: P_base(g) = 500 + 9.500·(g/NGEN)^γ")
seed = st.sidebar.number_input("Semente aleatória", min_value=0, max_value=10_000, value=42, step=1)

otimizar = st.sidebar.button("🚀 Otimizar Cesta para t+1", type="primary", width="stretch")


# =====================================================================
# 4. Painel Principal
# =====================================================================
st.title("Otimizador de Despensa — MOCaP")
st.markdown(
    f"Projeção de preços para **{mes_projetado}** (t+1) no estabelecimento **{mercado}** "
    f"e otimização evolutiva da cesta básica sob as metas de "
    f"**{formatar_num(meta_calorias)} kcal** e **{formatar_num(meta_proteinas)} g de proteína** por mês."
)

precos = projetar_precos(mercado)

# ---------------------------------------------------------------------
# Bloco 1 — Projeção de Preços XGBoost (t+1)
# ---------------------------------------------------------------------
st.header("1. Projeção de Preços XGBoost (t+1)")
ultimos_precos = df_pivot.xs(mercado, level="Estab").sort_index().iloc[-1]
linhas_precos = []
for p in PRODUTOS:
    m = metricas[p]
    qualificado = mercado in estabs_qualificados.get(p, [])
    linhas_precos.append({
        "Produto": p,
        "Unidade": f"R$/{UNIDADES[p]}",
        "Último preço observado (R$)": float(ultimos_precos[p]),
        "Preço projetado t+1 (R$)": precos[p],
        "Variação (%)": (precos[p] / float(ultimos_precos[p]) - 1) * 100,
        "MAPE teste 2026 (%)": m["mape"] * 100,
        "RMSE teste 2026 (R$)": m["rmse"],
        "Histórico do mercado": "✅ Coleta real" if qualificado else "⚠️ Interpolado (>50% ausente)",
    })
df_precos = pd.DataFrame(linhas_precos)
st.dataframe(
    df_precos,
    hide_index=True,
    width="stretch",
    column_config={
        "Último preço observado (R$)": st.column_config.NumberColumn(format="%.2f"),
        "Preço projetado t+1 (R$)": st.column_config.NumberColumn(format="%.2f"),
        "Variação (%)": st.column_config.NumberColumn(format="%+.2f"),
        "MAPE teste 2026 (%)": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=25),
        "RMSE teste 2026 (R$)": st.column_config.NumberColumn(format="%.4f"),
    },
)
st.caption(
    f"Modelos treinados com dados anteriores a {DATA_CORTE} e avaliados Out-of-Time em 2026. "
    "Itens marcados como ⚠️ têm série majoritariamente interpolada neste mercado, "
    "portanto a projeção deve ser interpretada com cautela."
)

# ---------------------------------------------------------------------
# Execução do Algoritmo Genético (persistida em session_state)
# ---------------------------------------------------------------------
parametros_atuais = (mercado, int(meta_calorias), int(meta_proteinas), n_populacao, n_geracoes, float(gama), int(seed))

if otimizar:
    with st.spinner(f"Evoluindo {n_populacao} indivíduos por {n_geracoes} gerações..."):
        st.session_state["resultado"] = otimizar_cesta_genetica(
            precos,
            n_populacao=n_populacao,
            n_geracoes=n_geracoes,
            gama=float(gama),
            seed=int(seed),
            verbose=False,
            meta_calorias=float(meta_calorias),
            meta_proteinas=float(meta_proteinas),
        )
        st.session_state["parametros"] = parametros_atuais
        st.session_state["precos_resultado"] = precos

resultado = st.session_state.get("resultado")

if resultado is None:
    st.info("Ajuste os parâmetros na barra lateral e clique em **🚀 Otimizar Cesta para t+1** para executar o DEAP.")
    st.stop()

if st.session_state.get("parametros") != parametros_atuais:
    st.warning("Os parâmetros foram alterados desde a última otimização. Clique em **Otimizar** para atualizar os resultados.")

mercado_res, cal_res, prot_res = st.session_state["parametros"][:3]
precos_res = st.session_state["precos_resultado"]

# ---------------------------------------------------------------------
# Bloco 2 — Comparativo Executivo
# ---------------------------------------------------------------------
st.header("2. Comparativo Executivo")
custo_deap = resultado["custo_total"]
custo_dieese = calcular_custo_cesta(CESTA_PADRAO_DIEESE, precos_res)
nutri_dieese = calcular_metricas_nutricionais([CESTA_PADRAO_DIEESE[p] for p in PRODUTOS], cal_res, prot_res)
economia = custo_dieese - custo_deap
economia_pct = economia / custo_dieese * 100 if custo_dieese else 0.0

if resultado["valido"]:
    st.success(f"Cesta 100% válida para {mercado_res}: todas as restrições nutricionais foram atendidas.")
else:
    st.error(
        f"A melhor solução encontrada ainda viola restrições (Δ nutricional = {resultado['delta_nutricional_final']:.4f}). "
        "Aumente a população/gerações ou revise as metas — elas podem ser inatingíveis dentro dos limites por produto."
    )

c1, c2, c3 = st.columns(3)
c1.metric("Cesta Otimizada (DEAP)", formatar_brl(custo_deap),
          delta=f"-{formatar_brl(economia)} vs. DIEESE" if economia >= 0 else f"+{formatar_brl(-economia)} vs. DIEESE",
          delta_color="inverse")
c2.metric("Cesta Padrão DIEESE", formatar_brl(custo_dieese), help="Decreto-Lei nº 399/1938 — 33,75 kg/mês.")
c3.metric("Economia gerada", f"{formatar_num(economia_pct, 2)} %", delta=formatar_brl(economia))

n1, n2, n3 = st.columns(3)
n1.metric("Calorias", f"{formatar_num(resultado['calorias_totais'])} kcal",
          delta=f"{formatar_num(resultado['calorias_totais'] - cal_res)} kcal vs. meta")
n2.metric("Proteínas", f"{formatar_num(resultado['proteinas_totais'])} g",
          delta=f"{formatar_num(resultado['proteinas_totais'] - prot_res)} g vs. meta")
pct_vazias = resultado["pct_calorias_vazias"] * 100
n3.metric("Calorias vazias (Açúcar + Óleo)", f"{formatar_num(pct_vazias, 2)} %",
          delta=f"{formatar_num(pct_vazias - TETO_CALORIAS_VAZIAS * 100, 2)} p.p. vs. teto de 25%",
          delta_color="inverse")

with st.expander("Como interpretar a economia?"):
    st.markdown(
        f"""
A cesta DIEESE fornece **{formatar_num(nutri_dieese['calorias'])} kcal** e **{formatar_num(nutri_dieese['proteinas'])} g**
de proteína, pois foi dimensionada em 1938 para um trabalhador em atividade braçal intensa (~3.112 kcal/dia).
A economia reflete dois efeitos combinados:

1. **Substituição inteligente de itens** guiada pelos preços projetados pelo XGBoost;
2. **Adequação da demanda calórica** ao perfil selecionado (ANVISA/OMS: 2.000 kcal/dia).

Para isolar o efeito puro da otimização, selecione o perfil **Alta Demanda / Laboral (Padrão DIEESE)**
e compare novamente.
"""
    )

# ---------------------------------------------------------------------
# Bloco 3 — Lista de Compras Otimizada
# ---------------------------------------------------------------------
st.header("3. Lista de Compras Otimizada")
linhas_lista = []
for p in PRODUTOS:
    qtd = resultado["cesta_kg"][p]
    linhas_lista.append({
        "Produto": p,
        "Unidade": UNIDADES[p],
        "Qtd. Otimizada DEAP (kg/L)": qtd,
        "Qtd. Referência DIEESE (kg/L)": CESTA_PADRAO_DIEESE[p],
        "Preço Unitário Proj. (R$)": precos_res[p],
        "Subtotal Otimizado (R$)": round(qtd * precos_res[p], 2),
        "Subtotal DIEESE (R$)": round(CESTA_PADRAO_DIEESE[p] * precos_res[p], 2),
    })
df_lista = pd.DataFrame(linhas_lista)
st.dataframe(
    df_lista,
    hide_index=True,
    width="stretch",
    column_config={
        "Qtd. Referência DIEESE (kg/L)": st.column_config.NumberColumn(format="%.2f"),
        "Preço Unitário Proj. (R$)": st.column_config.NumberColumn(format="%.2f"),
        "Subtotal Otimizado (R$)": st.column_config.NumberColumn(format="%.2f"),
        "Subtotal DIEESE (R$)": st.column_config.NumberColumn(format="%.2f"),
    },
)
total_kg_deap = sum(resultado["cesta_kg"].values())
st.caption(
    f"Total DEAP: {formatar_num(total_kg_deap)} kg/L — {formatar_brl(custo_deap)} | "
    f"Total DIEESE: {formatar_num(sum(CESTA_PADRAO_DIEESE.values()), 2)} kg/L — {formatar_brl(custo_dieese)}"
)

csv = df_lista.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
st.download_button(
    "⬇️ Baixar lista de compras (.csv)",
    data=csv,
    file_name=f"lista_compras_mocap_{mercado_res}_{mes_projetado.replace('/', '-')}.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------
# Bloco 4 — Auditoria Evolutiva
# ---------------------------------------------------------------------
st.header("4. Auditoria Evolutiva (Convergência)")
n_ger_res = len(resultado["historico_fitness_min"]) - 1
geracoes = list(range(n_ger_res + 1))


def grafico_convergencia(series: dict, cores: list[str], tracejados: list[bool], titulo_y: str,
                         formato_y: str = ",.2f", log: bool = False, zero: bool = False) -> alt.Chart:
    """Gráfico de linhas Altair com eixo X fixo [0, NGEN], cores e estilos distintos por série."""
    df_long = pd.DataFrame({"Geração": geracoes, **series}).melt(
        "Geração", var_name="Série", value_name="Valor"
    )
    nomes = list(series.keys())
    escala_y = alt.Scale(type="symlog") if log else alt.Scale(zero=zero, nice=True)
    return (
        alt.Chart(df_long)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("Geração:Q", scale=alt.Scale(domain=[0, n_ger_res], nice=False),
                    axis=alt.Axis(tickMinStep=1, format="d")),
            y=alt.Y("Valor:Q", title=titulo_y, scale=escala_y, axis=alt.Axis(format=formato_y)),
            color=alt.Color("Série:N", scale=alt.Scale(domain=nomes, range=cores),
                            legend=alt.Legend(orient="bottom", title=None, labelLimit=300)),
            strokeDash=alt.StrokeDash("Série:N", scale=alt.Scale(domain=nomes,
                                      range=[[6, 4] if t else [1, 0] for t in tracejados]), legend=None),
            size=alt.Size("Série:N", scale=alt.Scale(domain=nomes,
                          range=[2 if t else 4.5 for t in tracejados]), legend=None),
            tooltip=["Geração:Q", "Série:N", alt.Tooltip("Valor:Q", format=formato_y)],
        )
        .properties(height=320)
        .interactive(bind_y=False)
    )


g1, g2 = st.columns(2)

with g1:
    st.subheader("Custo da melhor cesta (elitismo)")
    st.altair_chart(
        grafico_convergencia(
            {
                "Melhor fitness (custo + penalidade)": resultado["historico_fitness_min"],
                "Custo real da melhor cesta": resultado["historico_custo_real"],
            },
            cores=["#3b82f6", "#f28e2b"], tracejados=[False, True], titulo_y="R$",
        ),
        use_container_width=True,
    )
    st.caption(
        "Linha azul (grossa) = fitness do melhor indivíduo (custo + penalidade); linha laranja tracejada = "
        "custo real. Quando a melhor cesta já é válida, a penalidade é zero e as duas linhas se sobrepõem."
    )

with g2:
    st.subheader("Déficit nutricional relativo (Δ)")
    st.altair_chart(
        grafico_convergencia(
            {
                "Δ médio da população": resultado["historico_deficit_medio"],
                "Δ do melhor indivíduo": resultado["historico_deficit"],
            },
            cores=["#3b82f6", "#e15759"], tracejados=[False, True], titulo_y="Δ nutricional",
            formato_y=".3f", zero=True,
        ),
        use_container_width=True,
    )
    st.caption(
        "Soma normalizada das violações (calorias, proteínas, calorias vazias e Arroz ≥ 0,5 × Feijão). "
        "A penalidade adaptativa empurra a população inteira para a região viável; o melhor indivíduo fica em Δ = 0."
    )

with st.expander("Diagnóstico da população (fração de inviáveis e fitness médio)"):
    e1, e2 = st.columns(2)
    with e1:
        st.altair_chart(
            grafico_convergencia(
                {"Indivíduos inviáveis (%)": [v * 100 for v in resultado["historico_pct_inviaveis"]]},
                cores=["#e15759"], tracejados=[False], titulo_y="% da população", formato_y=".0f", zero=True,
            ),
            use_container_width=True,
        )
    with e2:
        st.altair_chart(
            grafico_convergencia(
                {"Fitness médio da população": resultado["historico_fitness_med"]},
                cores=["#3b82f6"], tracejados=[False], titulo_y="R$ (escala log)", formato_y=",.0f", log=True,
            ),
            use_container_width=True,
        )
    st.caption(
        "O fitness médio pode subir nas gerações finais: a penalidade P_base(g) cresce de R$ 500 a R$ 10.000, "
        "então os poucos inviáveis gerados por mutação passam a pesar muito mais na média — comportamento esperado."
    )

st.divider()
st.caption(
    "MOCaP / CestaGen — PUC-Campinas · Engenharia de Computação · Aprendizado de Máquina. "
    "Dados: Índice da Cesta Básica (ICB), Set/2022 a Ago/2026."
)
