"""
MOCaP - Geração de resultados, tabelas e figuras para o Relatório Final
Execução (na raiz do projeto):  python relatorio/gerar_resultados.py

Saídas:
    relatorio/figuras/*.png      -> figuras do relatório
    relatorio/resultados.json    -> todos os números citados no relatório
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error  # noqa: E402

from etl_genetico import PRODUTOS, carregar_dados_icb, otimizar_cesta_genetica  # noqa: E402
from nutricao_mock import (  # noqa: E402
    CESTA_PADRAO_DIEESE, LOW_BOUNDS, UP_BOUNDS, PERFIS_NUTRICIONAIS, TABELA_NUTRICIONAL,
    TETO_CALORIAS_VAZIAS, RAZAO_ARROZ_FEIJAO,
)
from predicao import (  # noqa: E402
    bridge_precos_deap, calcular_custo_cesta, criar_features_temporais,
    separar_treino_teste_temporal, treinar_modelos_xgboost,
)

FIG = os.path.join(RAIZ, "relatorio", "figuras")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
R: dict = {}
PERFIL_URB, PERFIL_LAB = list(PERFIS_NUTRICIONAIS.keys())
METAS = {
    "urbano": (PERFIS_NUTRICIONAIS[PERFIL_URB]["calorias"], PERFIS_NUTRICIONAIS[PERFIL_URB]["proteinas"]),
    "laboral": (PERFIS_NUTRICIONAIS[PERFIL_LAB]["calorias"], PERFIS_NUTRICIONAIS[PERFIL_LAB]["proteinas"]),
}
t0 = time.time()


def salvar(nome):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, nome), bbox_inches="tight")
    plt.close()
    print(f"   figura: relatorio/figuras/{nome}")


# =====================================================================
# 1. Descrição da base
# =====================================================================
print("[1] Descrição da base")
df_raw = pd.read_excel("ICB_DADOS_2026.xlsx", sheet_name="ACUMUL_DESAGREG")
df10 = df_raw[df_raw["Produto"].isin(PRODUTOS)]
df_pivot, estabs_q = carregar_dados_icb(retornar_qualidade=True)
df_grp = df10[df10["PPK"] >= 1.0].groupby(["Mes_Ano_Data", "Estab", "Produto"])["PPK"].mean().unstack()
meses_por_estab = df_pivot.groupby(level="Estab").size()
R["base"] = {
    "registros_brutos": int(len(df_raw)),
    "produtos_distintos": int(df_raw["Produto"].nunique()),
    "estabelecimentos": int(df_raw["Estab"].nunique()),
    "periodo_inicio": str(pd.to_datetime(df_raw["Mes_Ano_Data"]).min().date()),
    "periodo_fim": str(pd.to_datetime(df_raw["Mes_Ano_Data"]).max().date()),
    "meses": int(pd.to_datetime(df_raw["Mes_Ano_Data"]).nunique()),
    "registros_10_produtos": int(len(df10)),
    "outliers_removidos_ppk_menor_1": int((df10["PPK"] < 1.0).sum()),
    "linhas_pivot": int(df_pivot.shape[0]),
    "nans_antes_interpolacao": int(df_grp.reindex(columns=PRODUTOS).isna().sum().sum()),
    "celulas_pivot": int(df_pivot.size),
    "estabs_com_3_ou_mais_meses": int((meses_por_estab >= 3).sum()),
    "meses_por_estab_mediana": float(meses_por_estab.median()),
    "estabs_qualificados_por_produto": {p: len(estabs_q[p]) for p in PRODUTOS},
}

# Figura 1 — índice de preço médio (base 100 no 1º mês)
media_mensal = df_pivot.groupby(level="Mes_Ano_Data").mean()
indice = media_mensal / media_mensal.iloc[0] * 100
fig, ax = plt.subplots(figsize=(9, 4.5))
for p in PRODUTOS:
    ax.plot(indice.index, indice[p], label=p, lw=1.4)
ax.axvline(pd.Timestamp("2026-01-01"), color="k", ls="--", lw=1)
ax.text(pd.Timestamp("2026-01-15"), ax.get_ylim()[1] * 0.97, "Início do teste\n(Out-of-Time)", fontsize=8, va="top")
ax.set_title("Evolução do preço médio por kg (índice base 100 = set/2022)")
ax.set_ylabel("Índice (base 100)")
ax.legend(ncol=2, fontsize=7, loc="upper left")
salvar("fig1_indice_precos.png")
R["base"]["variacao_acumulada_pct"] = {p: round(float(indice[p].iloc[-1] - 100), 1) for p in PRODUTOS}

# =====================================================================
# 2. Questão 1 — Previsão de preços (XGBoost vs. baselines)
# =====================================================================
print("[2] Questão 1 — XGBoost vs. baselines")
_, df_feat = criar_features_temporais(df_pivot)
df_tr, df_te = separar_treino_teste_temporal(df_feat, "2026-01-01")
modelos, metricas = treinar_modelos_xgboost(df_tr, df_te, estabelecimentos_qualificados=estabs_q)
R["particao"] = {"treino": int(len(df_tr)), "teste": int(len(df_te))}

linhas = []
importancias = {}
HP = dict(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
          random_state=42, n_jobs=-1)


def feats_retorno(d, p):
    """Features adimensionais para o XGBoost sobre variação relativa (evita limite de extrapolação das árvores)."""
    l1 = d[f"{p}_lag1"]
    return pd.DataFrame({
        "r12": l1 / d[f"{p}_lag2"] - 1, "r23": d[f"{p}_lag2"] / d[f"{p}_lag3"] - 1,
        "dev2": l1 / d[f"{p}_roll2"] - 1, "dev3": l1 / d[f"{p}_roll3"] - 1, "mes": d["mes"],
    })


for p in PRODUTOS:
    te = df_te[df_te.index.get_level_values("Estab").isin(estabs_q[p])]
    tr = df_tr[df_tr.index.get_level_values("Estab").isin(estabs_q[p])]
    y = te[p]
    cols = [f"{p}_lag1", f"{p}_lag2", f"{p}_lag3", f"{p}_roll2", f"{p}_roll3", "mes"]
    pred = modelos[p].predict(te[cols])
    # Variante: XGBoost prevendo a variação relativa y_t / y_{t-1} - 1
    from xgboost import XGBRegressor
    m_ret = XGBRegressor(**HP).fit(feats_retorno(tr, p), tr[p] / tr[f"{p}_lag1"] - 1)
    pred_ret = te[f"{p}_lag1"].values * (1 + m_ret.predict(feats_retorno(te, p)))
    # Apenas pontos efetivamente coletados (não interpolados) no mês t
    obs = df_grp[p].reindex(te.index).notna().values
    linhas.append({
        "produto": p, "n_teste": int(len(te)), "n_teste_observado": int(obs.sum()), "preco_medio": float(y.mean()),
        "mape_xgb": float(mean_absolute_percentage_error(y, pred)),
        "rmse_xgb": float(root_mean_squared_error(y, pred)),
        "mape_naive": float(mean_absolute_percentage_error(y, te[f"{p}_lag1"])),
        "rmse_naive": float(root_mean_squared_error(y, te[f"{p}_lag1"])),
        "mape_mm3": float(mean_absolute_percentage_error(y, te[f"{p}_roll3"])),
        "rmse_mm3": float(root_mean_squared_error(y, te[f"{p}_roll3"])),
        "mape_xgb_ret": float(mean_absolute_percentage_error(y, pred_ret)),
        "mape_xgb_obs": float(mean_absolute_percentage_error(y[obs], pred[obs])),
        "mape_naive_obs": float(mean_absolute_percentage_error(y[obs], te[f"{p}_lag1"][obs])),
        "mape_mm3_obs": float(mean_absolute_percentage_error(y[obs], te[f"{p}_roll3"][obs])),
        "mape_xgb_ret_obs": float(mean_absolute_percentage_error(y[obs], pred_ret[obs])),
        "rmse_xgb_obs": float(root_mean_squared_error(y[obs], pred[obs])),
        "rmse_naive_obs": float(root_mean_squared_error(y[obs], te[f"{p}_lag1"][obs])),
    })
    imp = modelos[p].get_booster().get_score(importance_type="gain")
    tot = sum(imp.values()) or 1.0
    importancias[p] = {c.replace(f"{p}_", ""): imp.get(c, 0.0) / tot for c in cols}
tab_q1 = pd.DataFrame(linhas)
R["q1_tabela"] = linhas
R["q1_resumo"] = {
    "mape_medio_xgb": float(tab_q1["mape_xgb"].mean()),
    "mape_medio_naive": float(tab_q1["mape_naive"].mean()),
    "mape_medio_mm3": float(tab_q1["mape_mm3"].mean()),
    "produtos_xgb_melhor_que_naive_mape": int((tab_q1["mape_xgb"] < tab_q1["mape_naive"]).sum()),
    "produtos_xgb_melhor_que_naive_rmse": int((tab_q1["rmse_xgb"] < tab_q1["rmse_naive"]).sum()),
    "produtos_xgb_melhor_que_mm3_mape": int((tab_q1["mape_xgb"] < tab_q1["mape_mm3"]).sum()),
    "mape_medio_xgb_ret": float(tab_q1["mape_xgb_ret"].mean()),
    "produtos_xgb_ret_melhor_que_naive_mape": int((tab_q1["mape_xgb_ret"] < tab_q1["mape_naive"]).sum()),
    "obs_mape_medio_xgb": float(tab_q1["mape_xgb_obs"].mean()),
    "obs_mape_medio_naive": float(tab_q1["mape_naive_obs"].mean()),
    "obs_mape_medio_mm3": float(tab_q1["mape_mm3_obs"].mean()),
    "obs_mape_medio_xgb_ret": float(tab_q1["mape_xgb_ret_obs"].mean()),
    "obs_produtos_xgb_melhor_que_naive_mape": int((tab_q1["mape_xgb_obs"] < tab_q1["mape_naive_obs"]).sum()),
    "obs_produtos_xgb_melhor_que_naive_rmse": int((tab_q1["rmse_xgb_obs"] < tab_q1["rmse_naive_obs"]).sum()),
    "pct_teste_observado": float(tab_q1["n_teste_observado"].sum() / tab_q1["n_teste"].sum()),
}
imp_df = pd.DataFrame(importancias).T
R["q1_importancia_media"] = {k: round(float(v), 3) for k, v in imp_df.mean().items()}

# Figura 2 — MAPE por produto: XGBoost vs. baselines
x = np.arange(len(PRODUTOS))
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.bar(x - 0.27, tab_q1["mape_naive"] * 100, 0.27, label="Ingênuo (preço do mês anterior)", color="#bab0ac")
ax.bar(x, tab_q1["mape_mm3"] * 100, 0.27, label="Média móvel 3 meses", color="#76b7b2")
ax.bar(x + 0.27, tab_q1["mape_xgb"] * 100, 0.27, label="XGBoost", color="#1f3b73")
ax.set_xticks(x, [p.replace("Carne Bovina ", "") for p in PRODUTOS], rotation=30, ha="right")
ax.set_ylabel("MAPE (%) — teste jan–ago/2026")
ax.set_title("Erro de previsão t+1 por produto: XGBoost vs. baselines")
ax.legend(fontsize=8)
salvar("fig2_mape_xgb_vs_baselines.png")

# Figura 3 — importância das features (gain)
fig, ax = plt.subplots(figsize=(7.5, 4))
im = ax.imshow(imp_df.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(imp_df.shape[1]), imp_df.columns)
ax.set_yticks(range(len(PRODUTOS)), [p.replace("Carne Bovina ", "") for p in PRODUTOS])
for i in range(imp_df.shape[0]):
    for j in range(imp_df.shape[1]):
        ax.text(j, i, f"{imp_df.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if imp_df.values[i, j] > 0.5 else "black")
ax.grid(False)
ax.set_title("Importância relativa das features (gain) por modelo XGBoost")
fig.colorbar(im, ax=ax, fraction=0.03)
salvar("fig3_importancia_features.png")

# Figura 4 — real vs. previsto (média entre mercados qualificados) no teste
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
for ax, p in zip(axes, ["Arroz", "Café", "Banana Prata"]):
    te = df_te[df_te.index.get_level_values("Estab").isin(estabs_q[p])].copy()
    cols = [f"{p}_lag1", f"{p}_lag2", f"{p}_lag3", f"{p}_roll2", f"{p}_roll3", "mes"]
    te["pred"] = modelos[p].predict(te[cols])
    g = te.groupby(level="Mes_Ano_Data")[[p, "pred", f"{p}_lag1"]].mean()
    ax.plot(g.index, g[p], "o-", label="Real", color="k")
    ax.plot(g.index, g["pred"], "s--", label="XGBoost", color="#1f3b73")
    ax.plot(g.index, g[f"{p}_lag1"], ":", label="Ingênuo", color="#999")
    ax.set_title(p)
    ax.tick_params(axis="x", rotation=45)
axes[0].set_ylabel("R$/kg (média dos mercados)")
axes[0].legend(fontsize=7)
fig.suptitle("Preço real vs. previsto (t+1) no período de teste — 2026")
salvar("fig4_real_vs_previsto.png")

# =====================================================================
# 3. Questão 2 — Otimização da cesta (GA vs. ótimo exato)
# =====================================================================
print("[3] Questão 2 — GA vs. força bruta")
grade = np.array(list(itertools.product(*[range(lo, up + 1) for lo, up in zip(LOW_BOUNDS, UP_BOUNDS)])), dtype=float)
cal_v = np.array([TABELA_NUTRICIONAL[p]["calorias"] for p in PRODUTOS], dtype=float)
prot_v = np.array([TABELA_NUTRICIONAL[p]["proteina"] for p in PRODUTOS], dtype=float)
vaz_v = np.array([cal_v[i] if p in ("Açúcar", "Óleo") else 0 for i, p in enumerate(PRODUTOS)], dtype=float)
with np.errstate(all="ignore"):
    cal_g, prot_g, vaz_g = grade @ cal_v, grade @ prot_v, grade @ vaz_v
regra_arroz = grade[:, PRODUTOS.index("Arroz")] >= RAZAO_ARROZ_FEIJAO * grade[:, PRODUTOS.index("Feijão")]
R["espaco_busca"] = int(len(grade))


def otimo_exato(precos, meta_cal, meta_prot):
    viavel = (cal_g >= meta_cal) & (prot_g >= meta_prot) & (vaz_g <= TETO_CALORIAS_VAZIAS * cal_g) & regra_arroz
    with np.errstate(all="ignore"):  # aviso espúrio do matmul (Accelerate/macOS)
        custo = grade @ np.array([precos[p] for p in PRODUTOS])
    custo[~viavel] = np.inf
    i = int(np.argmin(custo))
    return float(custo[i]), grade[i].astype(int).tolist(), int(viavel.sum())


precos_sup01 = bridge_precos_deap(modelos, df_pivot.xs("SUP_01", level="Estab"))
R["precos_sup01"] = precos_sup01
custo_dieese_sup01 = calcular_custo_cesta(CESTA_PADRAO_DIEESE, precos_sup01)
R["q2_sup01"] = {"custo_dieese": custo_dieese_sup01}
for nome, (mc, mp) in METAS.items():
    ce, ie, nviaveis = otimo_exato(precos_sup01, mc, mp)
    res = otimizar_cesta_genetica(precos_sup01, seed=42, verbose=False, meta_calorias=mc, meta_proteinas=mp)
    R["q2_sup01"][nome] = {
        "meta_calorias": mc, "meta_proteinas": mp,
        "custo_ga": res["custo_total"], "cesta_ga": res["cesta_kg"],
        "custo_otimo_exato": round(ce, 2), "cesta_otima_exata": dict(zip(PRODUTOS, ie)),
        "solucoes_viaveis": nviaveis,
        "calorias": res["calorias_totais"], "proteinas": res["proteinas_totais"],
        "pct_calorias_vazias": res["pct_calorias_vazias"],
        "economia_pct_vs_dieese": (custo_dieese_sup01 - res["custo_total"]) / custo_dieese_sup01 * 100,
        "ga_igual_otimo": bool(abs(res["custo_total"] - ce) < 0.01),
    }
    if nome == "urbano":
        hist42 = res

# Robustez: 30 sementes, penalidade adaptativa (γ variado) vs. estática
print("[3b] Robustez multi-semente (pode levar ~2 min)")
SEEDS = list(range(30))
ce_urb = R["q2_sup01"]["urbano"]["custo_otimo_exato"]
config = {"Estática (+10.000)": dict(usar_penalidade_adaptativa=False)}
for gm in [0.5, 1.0, 2.0, 3.0]:
    config[f"Adaptativa γ={gm}"] = dict(usar_penalidade_adaptativa=True, gama=gm)
robustez = {}
curvas = {}
for nome, kw in config.items():
    custos, ger_otimo = [], []
    curva = np.zeros(51)
    for s in SEEDS:
        r = otimizar_cesta_genetica(precos_sup01, seed=s, verbose=False, **kw)
        custos.append(r["custo_total"] if r["valido"] else np.nan)
        hc = np.array(r["historico_custo_real"])
        curva += hc
        hit = np.where(np.abs(hc - ce_urb) < 0.01)[0]
        ger_otimo.append(int(hit[0]) if len(hit) else np.nan)
    custos = np.array(custos)
    robustez[nome] = {
        "taxa_otimo_global": float(np.mean(np.abs(custos - ce_urb) < 0.01)),
        "custo_medio": float(np.nanmean(custos)), "custo_desvio": float(np.nanstd(custos)),
        "gap_medio_pct": float(np.nanmean((custos - ce_urb) / ce_urb * 100)),
        "geracao_media_otimo": float(np.nanmean(ger_otimo)) if not np.all(np.isnan(ger_otimo)) else None,
        "taxa_validas": float(np.mean(~np.isnan(custos))),
    }
    curvas[nome] = curva / len(SEEDS)
R["q2_robustez"] = robustez

# Perfil laboral (região viável menor): efeito do tamanho do AG — 30 sementes em SUP_01
ce_lab = R["q2_sup01"]["laboral"]["custo_otimo_exato"]
rob_lab = {}
for pop, ng in [(100, 50), (200, 100), (300, 200)]:
    cs = np.array([
        otimizar_cesta_genetica(precos_sup01, n_populacao=pop, n_geracoes=ng, seed=s, verbose=False,
                                meta_calorias=METAS["laboral"][0], meta_proteinas=METAS["laboral"][1])["custo_total"]
        for s in SEEDS
    ])
    rob_lab[f"pop{pop}_gen{ng}"] = {
        "taxa_otimo_global": float(np.mean(np.abs(cs - ce_lab) < 0.01)),
        "custo_medio": float(cs.mean()), "gap_medio_pct": float(((cs - ce_lab) / ce_lab * 100).mean()),
        "avaliacoes_aprox": pop * (ng + 1),
    }
R["q2_robustez_laboral"] = rob_lab

# Figura 5 — convergência (semente 42): custo + % inviáveis
fig, ax1 = plt.subplots(figsize=(8, 3.8))
g = np.arange(len(hist42["historico_custo_real"]))
ax1.plot(g, hist42["historico_fitness_min"], color="#1f3b73", lw=2.5, label="Melhor fitness")
ax1.plot(g, hist42["historico_custo_real"], color="#f28e2b", lw=1.8, ls="--", label="Custo real da melhor cesta")
ax1.axhline(ce_urb, color="green", lw=1, ls=":", label=f"Ótimo exato (força bruta) R$ {ce_urb:.2f}")
ax1.set_xlabel("Geração")
ax1.set_ylabel("R$")
ax2 = ax1.twinx()
ax2.plot(g, np.array(hist42["historico_pct_inviaveis"]) * 100, color="#e15759", lw=1.2, alpha=0.8, label="% população inviável")
ax2.set_ylabel("% inviáveis", color="#e15759")
ax2.grid(False)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
ax1.set_title("Convergência do AG em SUP_01 (perfil urbano, semente 42)")
salvar("fig5_convergencia_ga.png")

# Figura 6 — curva média de 30 sementes por configuração de penalidade
fig, ax = plt.subplots(figsize=(8, 3.8))
for nome, c in curvas.items():
    ax.plot(np.arange(51), c, lw=1.6, label=nome)
ax.axhline(ce_urb, color="k", ls=":", lw=1, label="Ótimo exato")
ax.set_xlabel("Geração")
ax.set_ylabel("Custo médio da melhor cesta (R$)")
ax.set_title("Sensibilidade à penalidade: média de 30 sementes (SUP_01, perfil urbano)")
ax.legend(fontsize=7)
salvar("fig6_sensibilidade_penalidade.png")

# Figura 7 — composição DEAP vs. DIEESE
fig, ax = plt.subplots(figsize=(9, 3.8))
q_deap = [R["q2_sup01"]["urbano"]["cesta_ga"][p] for p in PRODUTOS]
q_lab = [R["q2_sup01"]["laboral"]["cesta_ga"][p] for p in PRODUTOS]
q_die = [CESTA_PADRAO_DIEESE[p] for p in PRODUTOS]
ax.bar(x - 0.27, q_die, 0.27, label="Cesta DIEESE (1938)", color="#bab0ac")
ax.bar(x, q_lab, 0.27, label="DEAP — perfil laboral", color="#76b7b2")
ax.bar(x + 0.27, q_deap, 0.27, label="DEAP — perfil urbano", color="#1f3b73")
ax.set_xticks(x, [p.replace("Carne Bovina ", "") for p in PRODUTOS], rotation=30, ha="right")
ax.set_ylabel("Quantidade mensal (kg ou L)")
ax.set_title("Composição das cestas em SUP_01")
ax.legend(fontsize=8)
salvar("fig7_composicao_cestas.png")

# =====================================================================
# 4. Generalização para todos os mercados
# =====================================================================
print("[4] Todos os mercados elegíveis")
mercados = sorted(meses_por_estab[meses_por_estab >= 3].index)
linhas_m = []
for m in mercados:
    pr = bridge_precos_deap(modelos, df_pivot.xs(m, level="Estab"))
    cd = calcular_custo_cesta(CESTA_PADRAO_DIEESE, pr)
    lin = {"mercado": m, "custo_dieese": cd}
    for nome, (mc, mp) in METAS.items():
        r = otimizar_cesta_genetica(pr, seed=42, verbose=False, meta_calorias=mc, meta_proteinas=mp)
        ce, _, _ = otimo_exato(pr, mc, mp)
        lin[f"custo_ga_{nome}"] = r["custo_total"]
        lin[f"otimo_{nome}"] = ce
        lin[f"economia_{nome}_pct"] = (cd - r["custo_total"]) / cd * 100
        lin[f"ga_otimo_{nome}"] = bool(abs(r["custo_total"] - ce) < 0.01)
    r = otimizar_cesta_genetica(pr, n_populacao=300, n_geracoes=200, seed=42, verbose=False,
                                meta_calorias=METAS["laboral"][0], meta_proteinas=METAS["laboral"][1])
    lin["custo_ga_laboral_grande"] = r["custo_total"]
    lin["ga_otimo_laboral_grande"] = bool(abs(r["custo_total"] - lin["otimo_laboral"]) < 0.01)
    lin["economia_laboral_otimo_pct"] = (cd - lin["otimo_laboral"]) / cd * 100
    linhas_m.append(lin)
tab_m = pd.DataFrame(linhas_m)
R["mercados"] = {
    "n": len(mercados),
    "tabela": linhas_m,
    "economia_urbano": tab_m["economia_urbano_pct"].describe().round(2).to_dict(),
    "economia_laboral": tab_m["economia_laboral_pct"].describe().round(2).to_dict(),
    "custo_dieese": tab_m["custo_dieese"].describe().round(2).to_dict(),
    "custo_ga_urbano": tab_m["custo_ga_urbano"].describe().round(2).to_dict(),
    "ga_atingiu_otimo_urbano": int(tab_m["ga_otimo_urbano"].sum()),
    "ga_atingiu_otimo_laboral": int(tab_m["ga_otimo_laboral"].sum()),
    "ga_grande_atingiu_otimo_laboral": int(tab_m["ga_otimo_laboral_grande"].sum()),
    "gap_medio_laboral_pct": float(((tab_m["custo_ga_laboral"] - tab_m["otimo_laboral"]) / tab_m["otimo_laboral"] * 100).mean()),
    "gap_medio_laboral_grande_pct": float(((tab_m["custo_ga_laboral_grande"] - tab_m["otimo_laboral"]) / tab_m["otimo_laboral"] * 100).mean()),
    "economia_laboral_otimo": tab_m["economia_laboral_otimo_pct"].describe().round(2).to_dict(),
    "gap_medio_urbano_pct": float(((tab_m["custo_ga_urbano"] - tab_m["otimo_urbano"]) / tab_m["otimo_urbano"] * 100).mean()),
    "mercado_mais_barato_urbano": tab_m.loc[tab_m["custo_ga_urbano"].idxmin(), ["mercado", "custo_ga_urbano"]].to_dict(),
    "mercado_mais_caro_urbano": tab_m.loc[tab_m["custo_ga_urbano"].idxmax(), ["mercado", "custo_ga_urbano"]].to_dict(),
}

# Figura 8 — custo por mercado
tab_s = tab_m.sort_values("custo_ga_urbano")
fig, ax = plt.subplots(figsize=(10, 4))
xm = np.arange(len(tab_s))
ax.bar(xm, tab_s["custo_dieese"], color="#bab0ac", label="Cesta DIEESE")
ax.bar(xm, tab_s["custo_ga_laboral"], color="#76b7b2", label="DEAP — perfil laboral")
ax.bar(xm, tab_s["custo_ga_urbano"], color="#1f3b73", label="DEAP — perfil urbano")
ax.set_xticks(xm, tab_s["mercado"], rotation=90, fontsize=7)
ax.set_ylabel("Custo projetado t+1 (R$)")
ax.set_title(f"Custo da cesta por estabelecimento ({len(mercados)} mercados elegíveis)")
ax.legend(fontsize=8)
salvar("fig8_custo_por_mercado.png")

R["tempo_execucao_s"] = round(time.time() - t0, 1)
with open(os.path.join(RAIZ, "relatorio", "resultados.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2, default=float)
print(f"\nConcluído em {R['tempo_execucao_s']} s -> relatorio/resultados.json")
