"""
MOCaP - Modelo Computacional para Otimização e Análise Causal de Preços no Varejo
Módulo: nutricao_mock.py
Descrição: Tabela nutricional (peso seco, Tabela TACO), metas de restrição,
           limites individuais por produto para o Algoritmo Genético e
           baseline oficial da Cesta Básica DIEESE (Decreto-Lei nº 399/1938).
"""

# =====================================================================
# 1. Metas Nutricionais Mensais
# =====================================================================
# Perfil Urbano Padrão — VDR ANVISA (RDC nº 429/2020) e OMS:
# 2.000 kcal/dia x 30 dias e 50 g proteína/dia x 30 dias.
META_CALORIAS = 60000    # kcal/mês
META_PROTEINAS = 1500    # g/mês

# Teto de calorias vazias (Açúcar + Óleo) sobre o total calórico da cesta
TETO_CALORIAS_VAZIAS = 0.25
PRODUTOS_CALORIAS_VAZIAS = ('Açúcar', 'Óleo')

# Proporção cultural mínima: kg de Arroz >= RAZAO_ARROZ_FEIJAO * kg de Feijão
RAZAO_ARROZ_FEIJAO = 0.5

# =====================================================================
# 2. Tabela Nutricional (valores por KG/L do alimento, peso seco — TACO)
# =====================================================================
TABELA_NUTRICIONAL = {
    'Carne Bovina Coxão Mole': {
        'calorias': 2100,
        'proteina': 320,
        'carboidrato': 0
    },
    'Arroz': {
        # Arroz tipo 1 CRU (grão seco), coerente com o PPK coletado no ICB
        'calorias': 3580,
        'proteina': 72,
        'carboidrato': 788
    },
    'Feijão': {
        'calorias': 3400,
        'proteina': 210,
        'carboidrato': 630
    },
    'Manteiga': {
        'calorias': 7100,
        'proteina': 8,
        'carboidrato': 0
    },
    'Pão Francês': {
        'calorias': 3000,
        'proteina': 80,
        'carboidrato': 500
    },
    'Farinha de Trigo': {
        'calorias': 3600,
        'proteina': 100,
        'carboidrato': 760
    },
    'Açúcar': {
        'calorias': 3800,
        'proteina': 0,
        'carboidrato': 1000
    },
    'Óleo': {
        'calorias': 8800,
        'proteina': 0,
        'carboidrato': 0
    },
    'Café': {
        'calorias': 0,
        'proteina': 0,
        'carboidrato': 0
    },
    'Banana Prata': {
        'calorias': 890,
        'proteina': 10,
        'carboidrato': 220
    }
}

# Unidade de medida de cada item (Óleo é comercializado em litros)
UNIDADES = {p: 'kg' for p in TABELA_NUTRICIONAL}
UNIDADES['Óleo'] = 'L'

# =====================================================================
# 3. Limites Individuais por Produto (domínio heterogêneo do AG)
# =====================================================================
# Intervalos inteiros [min, max] de consumo mensal, inspirados nas quantidades
# da Cesta Básica DIEESE. Pisos de Açúcar = 1 e Óleo = 1 viabilizam o teto de
# 25% de calorias vazias; pisos de Carne, Pão, Café e Banana preservam a
# variedade cultural mínima da dieta brasileira e evitam cestas degeneradas.
LIMITES_PRODUTOS = {
    'Carne Bovina Coxão Mole': (2, 8),
    'Arroz': (2, 6),
    'Feijão': (1, 5),
    'Manteiga': (0, 1),
    'Pão Francês': (2, 6),
    'Farinha de Trigo': (1, 2),
    'Açúcar': (1, 3),
    'Óleo': (1, 2),
    'Café': (1, 2),
    'Banana Prata': (2, 8),
}

LOW_BOUNDS = [LIMITES_PRODUTOS[p][0] for p in TABELA_NUTRICIONAL]
UP_BOUNDS = [LIMITES_PRODUTOS[p][1] for p in TABELA_NUTRICIONAL]

# =====================================================================
# 4. Baseline Oficial — Cesta Básica DIEESE (Região Sudeste / SP)
# =====================================================================
# Decreto-Lei nº 399/1938 — dimensionada para trabalhador adulto em atividade
# braçal intensa. Totais: 33,75 kg/mês, 93.360 kcal/mês, 3.792 g proteína/mês
# (~3.112 kcal/dia), 20,69% de calorias vazias.
CESTA_PADRAO_DIEESE = {
    'Carne Bovina Coxão Mole': 6.0,
    'Arroz': 3.0,
    'Feijão': 4.5,
    'Manteiga': 0.75,
    'Pão Francês': 6.0,
    'Farinha de Trigo': 1.5,
    'Açúcar': 3.0,
    'Óleo': 0.9,
    'Café': 0.6,
    'Banana Prata': 7.5,
}

# =====================================================================
# 5. Perfis Nutricionais Selecionáveis no Dashboard
# =====================================================================
PERFIS_NUTRICIONAIS = {
    'Urbano Padrão (ANVISA RDC 429/2020 & OMS)': {
        'calorias': META_CALORIAS,
        'proteinas': META_PROTEINAS,
        'descricao': '2.000 kcal/dia e 50 g de proteína/dia (adulto médio, ciclo de 30 dias).',
    },
    'Alta Demanda / Laboral (Padrão DIEESE)': {
        'calorias': 93360,
        'proteinas': 2250,
        'descricao': '~3.112 kcal/dia (Decreto-Lei nº 399/1938) e 75 g de proteína/dia '
                     '(trabalhador em atividade física intensa).',
    },
}
