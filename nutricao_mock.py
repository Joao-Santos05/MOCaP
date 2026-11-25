"""
MOCaP - Modelo Computacional para Otimização e Análise Causal de Preços no Varejo
Módulo: nutricao_mock.py
Descrição: Tabela nutricional e metas de restrição para o Algoritmo Genético.
"""

# Constantes de Restrição Nutricional Mensal
META_CALORIAS = 60000    # kcal
META_PROTEINAS = 1500    # gramas

# Tabela Nutricional (valores por KG do alimento)
TABELA_NUTRICIONAL = {
    'Carne Bovina Coxão Mole': {
        'calorias': 2100,
        'proteina': 320,
        'carboidrato': 0
    },
    'Arroz': {
        'calorias': 1300,
        'proteina': 25,
        'carboidrato': 280
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
