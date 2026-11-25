# MOCaP: Modelo Computacional para Otimização e Análise Causal de Preços no Varejo
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![DEAP](https://img.shields.io/badge/Genetics-DEAP-green.svg)](https://github.com/DEAP/deap)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

Sistema integrado de inteligência de compras desenvolvido como atividade de extensão acadêmica do curso de Engenharia de Computação da Pontifícia Universidade Católica de Campinas (PUC-Campinas).

O **MOCaP** atua em duas frentes para proteger o poder de compra do consumidor diante da flutuação de preços: otimiza o orçamento da cesta básica garantindo metas nutricionais mínimas e mapeia o "efeito dominó" da inflação entre os grandes e pequenos estabelecimentos.

---

## 🏗️ Arquitetura Técnica Atual

O núcleo preditivo e de otimização (**Otimizador de Despensa**) já está implementado e validado em um pipeline ponta a ponta:

1. **ETL & Saneamento de Dados (`etl_genetico.py` + `nutricao_mock.py`):**
   * Leitura da base histórica `ICB_DADOS_2026.xlsx` (aba `ACUMUL_DESAGREG`, 46.264 registros, 51 estabelecimentos, 48 meses).
   * Filtragem dos 10 itens da cesta básica e remoção de *outliers* extremos de coleta (`PPK < R$ 1,00`).
   * **Prevenção de *Data Leakage*:** Pivotamento indexado por `(Mes_Ano_Data, Estab)` seguido de interpolação (`ffill`/`bfill`) **isolada por estabelecimento**, com *fallback* pela média mensal de mercado.
   * **Threshold de Qualidade:** Identificação de estabelecimentos com $\le 50\%$ de ausência histórica por produto, evitando que séries puramente sintéticas contaminem o treinamento.

2. **Módulo Preditivo (`predicao.py`):**
   * Engenharia de *features* temporais agrupada estritamente por estabelecimento (`lags` $t-1, t-2, t-3$, médias móveis de 2 e 3 meses sobre valores passados e *feature* sazonal `mes`).
   * Validação estritamente temporal (*Out-of-Time*, treino $< \text{2026-01-01}$ e teste $\ge \text{2026-01-01}$) com modelos `XGBRegressor` dedicados por produto.
   * **Função *Bridge* (`bridge_precos_deap`):** Converte as projeções $\hat{y}_{t+1}$ de um estabelecimento no dicionário exato exigido pelo Algoritmo Genético.

3. **Otimizador Genético (`etl_genetico.py`):**
   * Implementado via **DEAP** com representação inteira de 0 a 20 kg por produto, cruzamento `cxTwoPoint`, mutação `mutUniformInt`, seleção por torneio e elitismo (`HallOfFame`).
   * **Penalidade Adaptativa Progressiva (Fase 7.3):** Penalidade quadrática $P(\mathbf{q}, g)$ modulada pela geração atual e pelo déficit nutricional relativo ($\Delta_{\text{nutricional}}$), com reavaliação seletiva exclusiva para indivíduos inviáveis e registro de séries temporais de convergência.

---

## 📁 Estrutura de Diretórios e Status

```text
MOCaP/
├── ICB_DADOS_2026.xlsx             # ✅ Base histórica de preços do ICB (48 meses / 51 mercados)
├── nutricao_mock.py                # ✅ Concluído — Tabela nutricional (10 itens) e metas mínimas
├── etl_genetico.py                 # ✅ Concluído — Pipeline ETL, setup DEAP e penalidade adaptativa
├── predicao.py                     # ✅ Concluído — Feature engineering, XGBoost Out-of-Time e Bridge
├── plano_fase_7_3.md               # ✅ Concluído — Especificação da penalidade adaptativa
├── resumo_metodologico_mocap.md    # ✅ Concluído — Documentação acadêmica para arguição
├── causalidade.py                  # 🚧 Em desenvolvimento — Radar de Repasse (Causalidade de Granger)
├── app.py                          # 🚧 Em desenvolvimento — Dashboard interativo em Streamlit
├── .gitignore                      # ✅ Ignora venv/ e caches locais
└── README.md                       # ✅ Documentação do repositório
```

---

## 🚀 Como Executar e Testar (Estado Atual)

Os módulos de ETL, Predição e Algoritmo Genético possuem suítes de validação embutidas e podem ser executados diretamente no terminal utilizando o ambiente virtual (`venv`).

### 1. Ativar o Ambiente Virtual

No **Windows (PowerShell)**:
```powershell
.\venv\Scripts\Activate.ps1
```

No **Linux / macOS**:
```bash
source venv/bin/activate
```

*(Caso ainda não tenha as dependências instaladas no ambiente, execute `pip install pandas openpyxl scikit-learn xgboost deap streamlit statsmodels matplotlib`.)*

### 2. Testar o Pipeline de ETL e o Algoritmo Genético Adaptativo
```bash
python etl_genetico.py
```
**O que esperar:** Carregamento e pivotamento do Excel (`1281 x 10`, zero `NaN`), teste de retrocompatibilidade da penalidade estática, exibição da curva quadrática da penalidade adaptativa, evolução em 50 gerações e impressão da cesta ótima válida ($\ge 60.000$ kcal e $\ge 1.500$ g de proteína).

### 3. Testar a Predição XGBoost Out-of-Time e a Função Bridge
```bash
python predicao.py
```
**O que esperar:** Auditoria automática confirmando zero *data leakage* nas primeiras datas dos 51 estabelecimentos, tabela de RMSE/MAPE no conjunto de teste de 2026 (com destaque para Pão Francês, Feijão e Banana Prata), projeção $\hat{y}_{t+1}$ para o estabelecimento `SUP_01` via `bridge_precos_deap` e validação de custo na função de *fitness*.

### 4. Executar o Dashboard Streamlit *(Em breve)*
> 🚧 O arquivo `app.py` ainda será implementado na próxima etapa para integrar visualmente o Otimizador de Despensa e o Radar de Repasse (`streamlit run app.py`).

---

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python 3.10+
* **Manipulação de Dados:** Pandas, NumPy, OpenPyXL
* **Modelagem Preditiva:** XGBoost, Scikit-learn
* **Otimização Heurística:** DEAP (Distributed Evolutionary Algorithms in Python)
* **Séries Temporais / Causalidade:** Statsmodels *(em integração no módulo `causalidade.py`)*
* **Interface / Dashboard:** Streamlit *(em integração no arquivo `app.py`)*
