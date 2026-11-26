# MOCaP: Modelo Computacional para Otimização e Análise Causal de Preços no Varejo
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![DEAP](https://img.shields.io/badge/Genetics-DEAP-green.svg)](https://github.com/DEAP/deap)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

Sistema integrado de inteligência de compras desenvolvido como atividade de extensão acadêmica do curso de Engenharia de Computação da Pontifícia Universidade Católica de Campinas (PUC-Campinas), na disciplina de Aprendizado de Máquina.

O **MOCaP / CestaGen** protege o poder de compra do consumidor diante da flutuação de preços por meio de um **Pipeline Preditivo-Evolutivo**: projeta os preços do próximo mês com XGBoost e, a partir dessas projeções, otimiza via Algoritmo Genético a cesta básica de menor custo que ainda atende às metas nutricionais mínimas.

> **Escopo desta entrega:** ETL + XGBoost + DEAP Adaptativo + Dashboard Streamlit. A análise de repasse de preços entre estabelecimentos (Causalidade de Granger) foi reposicionada como **Trabalho Futuro / Extensão para TCC** (ver seção [Trabalhos Futuros](#-trabalhos-futuros)).

---

## 🏗️ Arquitetura Técnica

```text
ICB_DADOS_2026.xlsx ──► ETL (etl_genetico.py) ──► XGBoost (predicao.py) ──► bridge_precos_deap() ──► DEAP (etl_genetico.py) ──► Cesta ótima
                                                                                                                   ▲
                                                                                    nutricao_mock.py (metas, limites, baseline DIEESE)
```

1. **ETL & Saneamento de Dados (`etl_genetico.py` → `carregar_dados_icb`):**
   * Leitura da base histórica `ICB_DADOS_2026.xlsx` (aba `ACUMUL_DESAGREG`, 46.264 registros, 51 estabelecimentos, 48 meses — Set/2022 a Ago/2026).
   * Filtragem dos 10 itens da cesta básica e remoção de *outliers* extremos de coleta (`PPK < R$ 1,00`).
   * **Prevenção de *Data Leakage*:** Pivotamento indexado por `(Mes_Ano_Data, Estab)` seguido de interpolação em 3 camadas (`ffill`/`bfill` **isolado por estabelecimento** → média mensal de mercado → *fallback* residual). Resultado: `1.281 x 10`, zero `NaN`.
   * **Threshold de Qualidade (`identificar_estabelecimentos_qualificados`):** Estabelecimentos com $> 50\%$ de ausência histórica em um produto são excluídos do treino/teste daquele item.

2. **Base Nutricional (`nutricao_mock.py`):**
   * `TABELA_NUTRICIONAL` dos 10 alimentos padronizada em **peso seco** (Tabela TACO; ex.: Arroz cru = 3.580 kcal/kg e 72 g prot/kg).
   * Metas padrão `META_CALORIAS = 60.000 kcal/mês` e `META_PROTEINAS = 1.500 g/mês`, fundamentadas no VDR da **ANVISA (RDC nº 429/2020)** e da **OMS** (2.000 kcal/dia e 50 g proteína/dia × 30 dias).
   * Limites individuais por produto `LIMITES_PRODUTOS` (`LOW_BOUNDS` / `UP_BOUNDS`) baseados no consumo mensal DIEESE.
   * Baseline `CESTA_PADRAO_DIEESE` (Decreto-Lei nº 399/1938 — 33,75 kg/mês, 93.360 kcal/mês, 3.792 g proteína/mês).

3. **Módulo Preditivo (`predicao.py`):**
   * `criar_features_temporais`: *lags* $t-1, t-2, t-3$, médias móveis de 2 e 3 meses sobre valores passados e *feature* sazonal `mes`, agrupados estritamente por estabelecimento.
   * `separar_treino_teste_temporal`: validação *Out-of-Time* (treino $< \text{2026-01-01}$, teste $\ge \text{2026-01-01}$).
   * `treinar_modelos_xgboost`: um `XGBRegressor` dedicado por produto (MAPE de 3,25% no Pão Francês a 20,05% na Banana Prata; média ≈ 9,5%).
   * **Função *Bridge* (`bridge_precos_deap`):** Converte as projeções $\hat{y}_{t+1}$ de um estabelecimento no dicionário exato exigido pelo Algoritmo Genético.

4. **Otimizador Genético (`etl_genetico.py` → `avaliar_cesta` / `otimizar_cesta_genetica`):**
   * Implementado via **DEAP** com genes inteiros heterogêneos em `[LOW_BOUNDS_i, UP_BOUNDS_i]` (`initIterate`), cruzamento `cxTwoPoint`, mutação `mutUniformInt` respeitando os limites, seleção por torneio (`tournsize=3`) e elitismo (`HallOfFame`).
   * **Restrições:** meta calórica mínima, meta proteica mínima, teto de 25% de calorias vazias (Açúcar + Óleo) e proporção cultural $\text{Arroz} \ge 0{,}5 \times \text{Feijão}$.
   * **Penalidade Adaptativa Progressiva (Fase 7.3):** $P(\mathbf{q}, g) = \left[500 + 9.500 \cdot (g/\text{NGEN})^2\right] \times (1 + \Delta_{\text{nutricional}})$, com reavaliação seletiva exclusiva para indivíduos inviáveis e registro de séries temporais de convergência.
   * **Resultado de referência (`SUP_01`, projeção $t+1$):** Cesta DEAP de **R\$ 278,91** vs. `CESTA_PADRAO_DIEESE` de **R\$ 631,29** (economia de **55,82%**). Sob o perfil laboral (93.360 kcal), a economia atribuível apenas à otimização é de **32,52%** (R\$ 425,99).

---

## 📁 Estrutura de Diretórios e Status

```text
MOCaP/
├── ICB_DADOS_2026.xlsx             # ✅ Base histórica de preços do ICB (48 meses / 51 mercados)
├── nutricao_mock.py                # ✅ Concluído — Tabela nutricional TACO, metas, limites e baseline DIEESE
├── etl_genetico.py                 # ✅ Concluído — Pipeline ETL, setup DEAP e penalidade adaptativa
├── predicao.py                     # ✅ Concluído — Feature engineering, XGBoost Out-of-Time e Bridge
├── resumo_metodologico_mocap.md    # ✅ Concluído — Documentação acadêmica para arguição
├── app.py                          # ✅ Concluído — Dashboard interativo em Streamlit
├── requirements.txt                # ✅ Dependências do projeto
├── tests/test_dashboard.py         # ✅ Teste automatizado do dashboard (ver TESTE_DASHBOARD.md)
├── TESTE_DASHBOARD.md              # ✅ Guia de testes do dashboard
├── .gitignore                      # ✅ Ignora venv/ e caches locais
└── README.md                       # ✅ Documentação do repositório
```

---

## 🚀 Como Executar e Testar

Os módulos de ETL, Predição e Algoritmo Genético possuem suítes de validação embutidas e podem ser executados diretamente no terminal utilizando um ambiente virtual com **Python 3.9+**.

### 1. Criar e Ativar o Ambiente Virtual

No **Linux / macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

No **Windows (PowerShell)**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Instale as dependências:
```bash
pip install -r requirements.txt
```

> **macOS:** o XGBoost requer o runtime OpenMP. Caso apareça o erro `libomp.dylib could not be loaded`, instale-o com `brew install libomp`.

### 2. Testar o Pipeline de ETL e o Algoritmo Genético Adaptativo
```bash
python etl_genetico.py
```
**O que esperar:** Carregamento e pivotamento do Excel (`1281 x 10`, zero `NaN`), teste de retrocompatibilidade da penalidade estática (+10.000), exibição da curva quadrática da penalidade adaptativa, evolução em 50 gerações e impressão da cesta ótima válida ($\ge 60.000$ kcal, $\ge 1.500$ g de proteína, $\le 25\%$ de calorias vazias e Arroz $\ge 0{,}5 \times$ Feijão).

### 3. Testar a Predição XGBoost Out-of-Time e a Função Bridge
```bash
python predicao.py
```
**O que esperar:** Auditoria automática confirmando zero *data leakage* nas primeiras datas dos 51 estabelecimentos, tabela de RMSE/MAPE no conjunto de teste de 2026 (com destaque para Pão Francês, Feijão e Banana Prata), projeção $\hat{y}_{t+1}$ para o estabelecimento `SUP_01` via `bridge_precos_deap` e comparação integrada entre a `CESTA_PADRAO_DIEESE` (R\$ 631,29) e a cesta otimizada pelo DEAP (R\$ 278,91).

### 4. Executar o Dashboard Streamlit
```bash
streamlit run app.py
```
O navegador abrirá em `http://localhost:8501`. Na primeira execução, o ETL e o treino dos 10 modelos XGBoost são executados uma única vez e mantidos em cache (`@st.cache_data` / `@st.cache_resource`).

**Funcionalidades:**
* **Barra lateral:** seleção do estabelecimento (padrão `SUP_01`), perfil nutricional (**Urbano Padrão — ANVISA/OMS**: 60.000 kcal / 1.500 g; **Alta Demanda / Laboral — DIEESE**: 93.360 kcal / 2.250 g; ou metas personalizadas) e hiperparâmetros do AG (população, gerações, γ e semente).
* **Bloco 1 — Projeção XGBoost (t+1):** preços projetados via `bridge_precos_deap()` com MAPE/RMSE de cada item e alerta para séries interpoladas.
* **Bloco 2 — Comparativo executivo:** custo DEAP vs. `CESTA_PADRAO_DIEESE`, economia em R\$ e %, calorias, proteínas e % de calorias vazias frente ao teto de 25%.
* **Bloco 3 — Lista de compras:** quantidades DEAP vs. DIEESE, preços e subtotais, com exportação em `.csv`.
* **Bloco 4 — Auditoria evolutiva:** curvas do melhor fitness/custo e do déficit nutricional relativo $\Delta_{\text{nutricional}}$ por geração.

---

## 🔭 Trabalhos Futuros

Extensões mapeadas para continuidade do projeto (detalhadas na Seção 7 do `resumo_metodologico_mocap.md`):

* **Radar de Repasse (Causalidade de Granger):** identificar estabelecimentos formadores de preço e a defasagem de repasse aos mercados de bairro — reposicionado como **Extensão para TCC**.
* **NSGA-II:** otimização multiobjetivo (custo × variedade × qualidade nutricional) com Fronteira de Pareto.
* **Deep Learning para Séries Temporais:** comparação do XGBoost com LSTM, N-BEATS e TFT.
* **VRP + Basket Optimization:** compra fracionada entre múltiplos mercados considerando custo de deslocamento.

---

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python 3.9+
* **Manipulação de Dados:** Pandas, NumPy, OpenPyXL
* **Modelagem Preditiva:** XGBoost, Scikit-learn
* **Otimização Heurística:** DEAP (Distributed Evolutionary Algorithms in Python)
* **Interface / Dashboard:** Streamlit
