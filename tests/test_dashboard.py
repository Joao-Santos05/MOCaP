"""
MOCaP - Teste automatizado (headless) do dashboard app.py
Simula as interações de um usuário sem abrir o navegador, usando streamlit.testing.

Execução (na raiz do projeto, com o venv ativo):
    python tests/test_dashboard.py
"""

import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)

from streamlit.testing.v1 import AppTest  # noqa: E402

PERFIL_LABORAL = "Alta Demanda / Laboral (Padrão DIEESE)"


def metricas(at):
    return {m.label: m.value for m in at.metric}


def verificar(at, etapa):
    assert not at.exception, f"[{etapa}] Exceção no app: {at.exception}"
    assert not at.error, f"[{etapa}] Erro exibido: {[e.value for e in at.error]}"


at = AppTest.from_file(os.path.join(RAIZ, "app.py"), default_timeout=300).run()

# 1. Carregamento inicial (ETL + treino XGBoost em cache)
verificar(at, "carga inicial")
mercados = at.sidebar.selectbox[0]
assert mercados.value == "SUP_01", "Mercado padrão deveria ser SUP_01"
print(f"[OK] Carga inicial — {len(mercados.options)} mercados disponíveis, padrão {mercados.value}")

# 2. Perfil Urbano Padrão (ANVISA/OMS)
assert (at.sidebar.number_input[0].value, at.sidebar.number_input[1].value) == (60000, 1500)
at.sidebar.button[0].click().run()
verificar(at, "perfil urbano")
assert at.success, "A cesta otimizada deveria ser válida no perfil urbano"
m = metricas(at)
print(f"[OK] Perfil Urbano — DEAP {m['Cesta Otimizada (DEAP)']} | DIEESE {m['Cesta Padrão DIEESE']} "
      f"| Economia {m['Economia gerada']} | {m['Calorias']} | {m['Proteínas']}")

# 3. Perfil Laboral (DIEESE) preenche as metas automaticamente
at.sidebar.radio[0].set_value(PERFIL_LABORAL).run()
assert (at.sidebar.number_input[0].value, at.sidebar.number_input[1].value) == (93360, 2250)
assert at.warning, "Deveria avisar que os parâmetros mudaram desde a última otimização"
at.sidebar.button[0].click().run()
verificar(at, "perfil laboral")
assert at.success, "A cesta otimizada deveria ser válida no perfil laboral"
m = metricas(at)
print(f"[OK] Perfil Laboral — DEAP {m['Cesta Otimizada (DEAP)']} | Economia {m['Economia gerada']} | {m['Calorias']}")

# 4. Ajuste manual muda o perfil para "Personalizado"
at.sidebar.number_input[0].set_value(70000).run()
assert at.sidebar.radio[0].value == "Personalizado"
print("[OK] Ajuste manual das metas — perfil alterado para 'Personalizado'")

# 5. Outro estabelecimento
outro = [o for o in mercados.options if o != "SUP_01"][0]
at.sidebar.selectbox[0].set_value(outro).run()
at.sidebar.button[0].click().run()
verificar(at, f"mercado {outro}")
m = metricas(at)
print(f"[OK] Mercado {outro} — DEAP {m['Cesta Otimizada (DEAP)']} | DIEESE {m['Cesta Padrão DIEESE']}")

# 6. Download do CSV disponível
assert len(at.get("download_button")) == 1, "Botão de download do CSV não encontrado"
print("[OK] Botão de download da lista de compras presente")

print("\nTODOS OS TESTES DO DASHBOARD PASSARAM!")
