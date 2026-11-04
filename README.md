# MOCaP: Modelo Computacional para Otimização e Análise Causal de Preços no Varejo
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![DEAP](https://img.shields.io/badge/Genetics-DEAP-green.svg)](https://github.com/DEAP/deap)

Sistema integrado de inteligência de compras desenvolvido como atividade de extensão acadêmica do curso de Engenharia de Computação da Pontifícia Universidade Católica de Campinas (PUC-Campinas). 

O **MOCaP** atua em duas frentes para proteger o poder de compra do consumidor diante da flutuação de preços: otimiza o orçamento da cesta básica garantindo metas nutricionais mínimas e mapeia o "efeito dominó" da inflação entre os grandes e pequenos estabelecimentos.

## ✨ Funcionalidades

O sistema é dividido em dois módulos principais consolidados em um dashboard interativo:

### 1. Otimizador de Despensa (Algoritmos Genéticos + XGBoost)
* **Previsão:** Utiliza modelos de aprendizado supervisionado para projetar o preço por quilo (PPK) dos alimentos essenciais para o mês seguinte.
* **Otimização:** Emprega Algoritmos Genéticos (via biblioteca DEAP) para cruzar e mutar combinações de compras.
* **Resultado:** Sugere uma lista de compras (em KG) que minimiza o impacto financeiro no bolso da família, sujeita a restrições rígidas (penalidades na função de *fitness*) para garantir a ingestão mínima de calorias e proteínas estipulada no mês.

### 2. Radar de Repasse (Estatística e Causalidade)
* **Mapeamento:** Trata uma base histórica de preços de mais de 50 estabelecimentos para identificar padrões de reajuste.
* **Causalidade de Granger:** Aplica testes estatísticos para descobrir quais mercados atuam como "líderes" (trendsetters).
* **Resultado:** Gera um alerta antecipado informando o tempo exato (lag/defasagem) que um aumento de preços nos hipermercados levará para atingir os mercados de bairro.

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python
* **Manipulação de Dados:** Pandas, NumPy
* **Modelagem Preditiva:** XGBoost, Scikit-learn
* **Otimização Heurística:** DEAP (Distributed Evolutionary Algorithms in Python)
* **Séries Temporais:** Statsmodels
* **Interface / Dashboard:** Streamlit
