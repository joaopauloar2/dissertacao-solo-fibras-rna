# -*- coding: utf-8 -*-
"""
Verificacao e regeneracao da Figura 3 da dissertacao
(Exemplos de trajetorias tensao-deformacao sinteticas geradas pelo modelo analitico).

O script executa duas tarefas:

1. VERIFICACAO FISICA do arquivo dados_sinteticos_tier1_filtrado.csv, cenario a
   cenario, checando:
   (a) trajetoria de tensoes do ensaio CID convencional: dq/dp = 3, verificada
       de forma acumulada, (q - q0)/(p - p0) = 3 em todos os pontos;
   (b) consistencia p' = sigma3 + q/3;
   (c) ponto inicial de cada cenario: q = 0 e p' = sigma3;
   (d) consistencia do indice de vazios: e = e0 - ev (1 + e0), com a convencao
       de deformacao volumetrica positiva na compressao;
   (e) deformacao axial estritamente crescente;
   (f) ordem fisica dos picos: q_max cresce com o confinamento para uma mesma
       configuracao de fibra, e cresce com o teor de fibra para um mesmo
       confinamento e comprimento.

2. REGENERACAO da figura de dois paineis, com os mesmos cenarios da versao da
   dissertacao: (a) efeito do confinamento (Pf = 0,5%, L = 25 mm) e
   (b) efeito do reforco (sigma3 = 200 kPa; sem fibra, 0,5% e 1,0% com L = 25 mm).

Uso:
    python verificar_trajetorias_sinteticas.py [caminho_do_csv]

Por padrao, procura 'dados_sinteticos_tier1_filtrado.csv' no diretorio atual.
Saidas: relatorio no terminal e 'curvas_sinteticas_verificacao.png'.
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
})

CSV = sys.argv[1] if len(sys.argv) > 1 else "dados_sinteticos_tier1_filtrado.csv"
TOL_REL = 5e-3   # tolerancia relativa das checagens (0,5%)
TOL_ABS = 0.75   # tolerancia absoluta em kPa para os primeiros pontos

df = pd.read_csv(CSV)
grupos = df.groupby(["po_kpa", "teor_de_fibra", "comp_de_fibra"], sort=True)
print(f"Arquivo: {CSV}")
print(f"Pontos: {len(df)} | Cenarios: {grupos.ngroups}\n")

falhas = []

def checa(cond, msg):
    if not cond:
        falhas.append(msg)

resumo_picos = {}
for (s3, pf, L), g in grupos:
    g = g.sort_values("ea_deformacao_axial").reset_index(drop=True)
    nome = f"sigma3={s3} kPa, Pf={pf}%, L={L} mm"

    q = g["q_tensao_desviadora"].to_numpy()
    p = g["p_tensao_media"].to_numpy()
    ea = g["ea_deformacao_axial"].to_numpy()
    ev = g["ev_deformacao_volumetrica"].to_numpy()
    e = g["e_indice_de_vazios"].to_numpy()
    e0 = g["e_inicial_vazio_inicial"].iloc[0]

    # (c) ponto inicial
    checa(abs(q[0]) < TOL_ABS and abs(p[0] - s3) < TOL_ABS,
          f"{nome}: ponto inicial q={q[0]:.3f}, p={p[0]:.3f} (esperado 0 e {s3})")

    # (a) trajetoria dq/dp = 3 (forma acumulada, robusta a ruido incremental)
    dp_ac = p - s3
    mask = dp_ac > 1.0
    razao = q[mask] / dp_ac[mask]
    checa(np.allclose(razao, 3.0, rtol=TOL_REL),
          f"{nome}: (q - 0)/(p - sigma3) fora de 3 "
          f"(min={razao.min():.4f}, max={razao.max():.4f})")

    # (b) p' = sigma3 + q/3
    checa(np.allclose(p, s3 + q / 3.0, rtol=TOL_REL, atol=TOL_ABS),
          f"{nome}: p' difere de sigma3 + q/3")

    # (d) indice de vazios: a planilha geradora atualiza e de forma
    # incremental, de = -dev (1 + e), cujo equivalente acumulado e
    # e = (1 + e0) exp(-ev) - 1. A forma linear e0 - ev (1 + e0) e uma
    # aproximacao de primeira ordem, valida para |ev| pequeno.
    checa(np.allclose(e, (1.0 + e0) * np.exp(-ev) - 1.0,
                      rtol=1e-3, atol=1e-4),
          f"{nome}: e != (1 + e0) exp(-ev) - 1")

    # (e) deformacao axial estritamente crescente
    checa(np.all(np.diff(ea) > 0), f"{nome}: eps_a nao e crescente")

    resumo_picos[(s3, pf, L)] = q.max()

# (f) ordens fisicas dos picos
for (pf, L) in sorted({(k[1], k[2]) for k in resumo_picos}):
    s3s = sorted({k[0] for k in resumo_picos if k[1:] == (pf, L)})
    picos = [resumo_picos[(s, pf, L)] for s in s3s]
    checa(all(a < b for a, b in zip(picos, picos[1:])),
          f"Pf={pf}%, L={L} mm: q_max nao cresce com sigma3: "
          + ", ".join(f"{s}:{p:.0f}" for s, p in zip(s3s, picos)))

for s3 in sorted({k[0] for k in resumo_picos}):
    for L in sorted({k[2] for k in resumo_picos if k[2] > 0}):
        trio = [(0.0, 0.0), (0.5, L), (1.0, L)]
        if all((s3, pf, l) in resumo_picos for pf, l in trio):
            v = [resumo_picos[(s3, pf, l)] for pf, l in trio]
            checa(v[0] < v[1] < v[2],
                  f"sigma3={s3}, L={L}: ordem de reforco violada "
                  f"(sem fibra {v[0]:.0f}, 0,5% {v[1]:.0f}, 1,0% {v[2]:.0f})")

print("=== RELATORIO DA VERIFICACAO ===")
if falhas:
    for f in falhas:
        print("FALHA:", f)
else:
    print("Todas as checagens passaram: trajetorias dq/dp = 3, "
          "p' = sigma3 + q/3, pontos iniciais, indice de vazios, "
          "monotonicidade de eps_a e ordem fisica dos picos.")

print("\nPicos de q (kPa) por cenario (Pf = 0,5%, L = 25 mm):")
for s3 in sorted({k[0] for k in resumo_picos}):
    k = (s3, 0.5, 25.0)
    if k in resumo_picos:
        print(f"  sigma3 = {s3:3d} kPa -> q_max = {resumo_picos[k]:6.1f}")

# ------------------------- regeneracao da figura -------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), dpi=300, sharey=False)
tons = ["0.0", "0.15", "0.3", "0.45", "0.6", "0.72"]

ax = axes[0]
s3_niveis = sorted(df["po_kpa"].unique())
for cor, s3 in zip(tons, s3_niveis):
    g = df[(df.po_kpa == s3) & (df.teor_de_fibra == 0.5)
           & (df.comp_de_fibra == 25.0)].sort_values("ea_deformacao_axial")
    ax.plot(g.ea_deformacao_axial * 100, g.q_tensao_desviadora,
            color=cor, lw=1.6, label=rf"$\sigma_3$ = {s3} kPa")
ax.set_title(r"(a) Efeito do confinamento ($P_f$ = 0,5%, $L$ = 25 mm)",
             fontsize=11)
ax.legend(fontsize=9, frameon=False)

ax = axes[1]
series = [((0.0, 0.0), "sem fibra", "0.55", "--"),
          ((0.5, 25.0), r"$P_f$ = 0,5% ($L$ = 25 mm)", "0.3", "-"),
          ((1.0, 25.0), r"$P_f$ = 1,0% ($L$ = 25 mm)", "0.0", "-")]
for (pf, L), rot, cor, ls in series:
    g = df[(df.po_kpa == 200) & (df.teor_de_fibra == pf)
           & (df.comp_de_fibra == L)].sort_values("ea_deformacao_axial")
    ax.plot(g.ea_deformacao_axial * 100, g.q_tensao_desviadora,
            color=cor, lw=1.6, ls=ls, label=rot)
ax.set_title(r"(b) Efeito do reforço ($\sigma_3$ = 200 kPa)", fontsize=11)
ax.legend(fontsize=9, frameon=False)

for ax in axes:
    ax.set_xlabel(r"Deformação axial, $\varepsilon_a$ (%)", fontsize=11)
    ax.set_ylabel(r"Tensão desviadora, $q$ (kPa)", fontsize=11)
    ax.grid(color="0.9", lw=0.6)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

fig.tight_layout()
fig.savefig("curvas_sinteticas_verificacao.png", facecolor="white")
print("\nFigura regenerada: curvas_sinteticas_verificacao.png")
