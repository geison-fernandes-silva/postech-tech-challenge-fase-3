"""Deterministic synthetic clinical-text data for the academic demonstration."""

from __future__ import annotations

import random

import pandas as pd

NORMAL = [
    "Radiografia de torax sem infiltrados ou derrame pleural.",
    "Exame sem alteracoes agudas; acompanhamento ambulatorial de rotina.",
    "Tomografia sem evidencia de sangramento ou lesao aguda.",
    "Hemograma dentro dos limites de referencia e paciente estavel.",
]
ATTENTION = [
    "Dor persistente com discreta elevacao de marcadores inflamatorios; avaliar em 24 horas.",
    "Nodulo pulmonar pequeno identificado; recomenda-se investigacao ambulatorial prioritaria.",
    "Desidratacao moderada e alteracao eletrolitica leve, necessita reavaliacao clinica.",
    "Possivel infeccao urinaria com febre baixa; iniciar protocolo de observacao.",
]
URGENT = [
    "Dor toracica intensa com alteracao isquemica no ECG; acionar cardiologia imediatamente.",
    "Tomografia evidencia hemorragia intracraniana aguda; prioridade maxima para neurocirurgia.",
    "Saturacao de oxigenio critica e infiltrado bilateral; suporte respiratorio imediato.",
    "Sinais de sepse com hipotensao persistente e lactato elevado; iniciar protocolo urgente.",
]


def make_dataset(samples_per_class: int = 800, seed: int = 42) -> pd.DataFrame:
    """Create 2,400 non-identifiable text records for training and tests."""
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for label, templates in (("normal", NORMAL), ("atencao", ATTENTION), ("urgente", URGENT)):
        for index in range(samples_per_class):
            qualifier = rng.choice(
                [
                    "Paciente adulto.",
                    "Paciente idoso.",
                    "Reavaliacao solicitada.",
                    "Sem dados identificaveis.",
                ]
            )
            rows.append(
                {
                    "report_text": f"{rng.choice(templates)} {qualifier} Registro {index}.",
                    "urgency": label,
                }
            )
    rng.shuffle(rows)
    return pd.DataFrame(rows)
