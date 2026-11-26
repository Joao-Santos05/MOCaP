# 🧪 Como Testar o Dashboard do MOCaP

Guia rápido para validar a versão final: `app.py`, as restrições do algoritmo genético e as metas nutricionais dinâmicas.

---

## 📦 O que mudou nesta versão

| Arquivo | Alteração |
|---|---|
| `nutricao_mock.py` | Arroz padronizado em **peso seco** (3.580 kcal/kg, 72 g prot/kg), `LIMITES_PRODUTOS` / `LOW_BOUNDS` / `UP_BOUNDS`, `CESTA_PADRAO_DIEESE` (33,75 kg, 93.360 kcal, 3.792 g), `UNIDADES` e `PERFIS_NUTRICIONAIS`. |
| `etl_genetico.py` | Genes heterogêneos (`initIterate`), 4 restrições (calorias, proteínas, teto de 25% de calorias vazias, Arroz ≥ 0,5 × Feijão) via `calcular_metricas_nutricionais()`, parâmetros `meta_calorias` / `meta_proteinas` (padrão 60.000 / 1.500) e `historico_calorias_vazias`. |
| `predicao.py` | Teste 4 agora compara a `CESTA_PADRAO_DIEESE` com a cesta DEAP em `SUP_01`. Nova função `calcular_custo_cesta()`. |
| `app.py` | **Novo:** dashboard Streamlit completo. |
| `tests/test_dashboard.py` | **Novo:** teste automatizado do dashboard (sem navegador). |
| `requirements.txt` | **Novo:** dependências do projeto. |

> [!NOTE]
> O `LIMITES_PRODUTOS` foi definido de forma que a cesta ótima reproduza a composição documentada no resumo metodológico (2 kg Carne, 6 kg Arroz, 2 kg Feijão…). **Confira se os limites batem com o que você tinha planejado.**

---

## 1. Preparar o ambiente

Na raiz do projeto:

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

> [!WARNING]
> **macOS:** se aparecer o erro `XGBoost Library (libxgboost.dylib) could not be loaded ... libomp.dylib`, instale o OpenMP com `brew install libomp` e rode novamente.

Requisitos: Python **3.9+**.

---

## 2. Testar os módulos base (terminal)

```bash
python etl_genetico.py
python predicao.py
```

**Resultado esperado:**
- `etl_genetico.py` termina com `TODOS OS TESTES DAS FASES 1 E 7.3 CONCLUÍDOS COM SUCESSO!` e uma cesta válida de **R\$ 251,50** (preços mock).
- `predicao.py` termina com `TODOS OS TESTES DA FASE 2 FORAM CONCLUÍDOS COM SUCESSO!` e o comparativo em `SUP_01`:

| Métrica | Cesta DIEESE | Cesta DEAP |
|---|---|---|
| Custo projetado | R\$ 631,29 | **R\$ 278,91** |
| Calorias | 93.360 kcal | 60.060 kcal |
| Proteínas | 3.792 g | 1.872 g |
| Calorias vazias | 20,69% | 20,98% |
| **Economia** | | **55,82%** |

> Variações na segunda casa decimal são normais se as versões do XGBoost/scikit-learn forem diferentes (validado com `xgboost 2.1.4`, `scikit-learn 1.6.1`, `pandas 2.3.3`, `streamlit 1.50`).

---

## 3. Teste automatizado do dashboard (sem abrir o navegador)

```bash
python tests/test_dashboard.py
```

O script simula um usuário usando o `streamlit.testing` e valida, em sequência:

1. Carregamento inicial (ETL + treino dos 10 XGBoost em cache), com `SUP_01` como padrão;
2. Otimização no perfil **Urbano Padrão** (60.000 kcal / 1.500 g) com cesta válida;
3. Troca para o perfil **Laboral (DIEESE)**, que preenche 93.360 kcal / 2.250 g automaticamente e mostra o aviso de parâmetros alterados;
4. Ajuste manual de meta, que muda o perfil para **Personalizado**;
5. Otimização em outro estabelecimento;
6. Presença do botão de download do CSV.

**Saída esperada (resumida):**
```text
[OK] Carga inicial — 31 mercados disponíveis, padrão SUP_01
[OK] Perfil Urbano — DEAP R$ 278,91 | DIEESE R$ 631,29 | Economia 55,82 % | 60.060 kcal | 1.872 g
[OK] Perfil Laboral — DEAP R$ 425,99 | Economia 32,52 % | 93.960 kcal
[OK] Ajuste manual das metas — perfil alterado para 'Personalizado'
[OK] Mercado SUP_04 — DEAP R$ 289,19 | DIEESE R$ 628,74
[OK] Botão de download da lista de compras presente

TODOS OS TESTES DO DASHBOARD PASSARAM!
```

---

## 4. Teste manual no navegador

```bash
streamlit run app.py
```

Abra `http://localhost:8501`. A primeira carga leva alguns segundos (ETL + treino); depois disso fica em cache.

### Checklist

- [ ] **Bloco 1 — Projeção XGBoost:** a tabela mostra os 10 produtos com preço projetado, variação, MAPE e RMSE. No `SUP_01`, o **Pão Francês** aparece como ⚠️ *Interpolado*, já que o mercado tem 98% de ausência nesse item.
- [ ] **Clique em "🚀 Otimizar Cesta para t+1"** (perfil Urbano, `SUP_01`, padrões 100 / 50 / γ=2,0):
  - [ ] Mensagem verde "Cesta 100% válida";
  - [ ] Cartões: **R\$ 278,91** (DEAP) vs. **R\$ 631,29** (DIEESE), economia de **55,82%**;
  - [ ] Calorias vazias abaixo do teto de 25%.
- [ ] **Bloco 3 — Lista de compras:** colunas DEAP vs. DIEESE lado a lado; o botão **⬇️ Baixar lista de compras (.csv)** gera um arquivo que abre corretamente no Excel (separador `;`, decimal `,`).
- [ ] **Bloco 4 — Convergência:** o gráfico do melhor fitness cai e estabiliza, e o gráfico do Δ nutricional chega a zero.
- [ ] **Trocar para "Alta Demanda / Laboral (Padrão DIEESE)":** as metas mudam para 93.360 / 2.250 e aparece um aviso amarelo pedindo nova otimização. Após otimizar: **R\$ 425,99**, economia de **32,52%** (ganho atribuível apenas à otimização).
- [ ] **Editar uma meta manualmente:** o perfil muda para **Personalizado**.
- [ ] **Metas inatingíveis** (ex.: 200.000 kcal): deve aparecer uma mensagem vermelha explicando que a solução viola restrições, sem travar o app.
- [ ] **Trocar o mercado** (ex.: `SUP_11`) e otimizar de novo: os preços e a cesta mudam.

---

## 5. Problemas comuns

| Sintoma | Solução |
|---|---|
| `libomp.dylib could not be loaded` (macOS) | `brew install libomp` |
| `ModuleNotFoundError: etl_genetico` | Rode os comandos **na raiz do projeto** |
| `FileNotFoundError: ICB_DADOS_2026.xlsx` | Confirme que a planilha está na raiz do projeto |
| Lista de mercados com 31 itens (e não 51) | Esperado: só aparecem mercados com ≥ 3 meses de histórico, exigência dos lags t-1..t-3 |
| Avisos `use_container_width` | Atualize o Streamlit (`pip install -U streamlit`) |
