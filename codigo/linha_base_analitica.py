# -*- coding: utf-8 -*-
"""
LINHA DE BASE: desempenho do proprio modelo analitico de Machado et al. (2024)
sobre os 1.207 pontos experimentais de Conceicao et al. (2022).

Para cada um dos 24 ensaios, toma-se a trajetoria sintetica gerada pelo modelo
analitico para o mesmo cenario (sigma3, Pf, L) e a interpola nos valores de
deformacao axial em que ha medida experimental. Compara-se entao a previsao do
modelo com a medida, exatamente com as mesmas metricas e a mesma agregacao
usadas para a rede (concatenacao dos 1.207 pares previsto-observado).

Advertencia metodologica, que deve constar do texto: esta comparacao FAVORECE o
modelo analitico, pois ele foi calibrado por Machado et al. (2024) com esses
mesmos ensaios, ao passo que a rede e avaliada em validacao cruzada
leave-one-out, prevendo cada ensaio sem nunca te-lo visto.
"""
import numpy as np, pandas as pd, pickle
from sklearn.metrics import r2_score

sin = pd.read_csv('/mnt/project/dados_sinteticos_tier1_filtrado.csv')
exp = pd.read_csv('/mnt/project/dados_experimentais_tier1.csv')
sin.loc[sin.teor_de_fibra == 0, 'comp_de_fibra'] = 0.0
exp.loc[exp.teor_de_fibra == 0, 'comp_de_fibra'] = 0.0
sin = sin[(sin.ea_deformacao_axial >= 0) & (sin.ea_deformacao_axial <= 0.25)]
exp = exp[~(exp.deltaEa < -0.05)].reset_index(drop=True)

COLS = [('q_tensao_desviadora','q'), ('p_tensao_media','p'),
        ('ev_deformacao_volumetrica','ev')]
agg = {v: [[], []] for _, v in COLS}
por_fold = []

for (s3, pf, L), g in exp.groupby(['po_kpa','teor_de_fibra','comp_de_fibra']):
    sg = sin[(sin.po_kpa==s3)&(sin.teor_de_fibra==pf)&(sin.comp_de_fibra==L)] \
         .sort_values('ea_deformacao_axial')
    ea = g.ea_deformacao_axial.values
    r = dict(po_kpa=s3, teor_de_fibra=pf, comp_de_fibra=L, n=len(g))
    for col, v in COLS:
        hat = np.interp(ea, sg.ea_deformacao_axial, sg[col])
        agg[v][0].extend(g[col]); agg[v][1].extend(hat)
        r['R2_'+v] = r2_score(g[col], hat)
    por_fold.append(r)

df = pd.DataFrame(por_fold)
df.to_csv('/home/claude/run/linha_base_analitica.csv', index=False)

print(f'pontos experimentais: {len(exp)}')
print('\n=== MODELO ANALITICO (linha de base) ===')
for _, v in COLS:
    y, yh = np.array(agg[v][0]), np.array(agg[v][1])
    r2 = r2_score(y, yh); rmse = np.sqrt(np.mean((y-yh)**2))
    if v == 'ev': rmse *= 100
    print(f'  {v:3s}: R2 agregado = {r2:7.4f}   RMSE = {rmse:8.4g}'
          f'{" %" if v=="ev" else " kPa"}   mediana por ensaio = {df["R2_"+v].median():.3f}')

print('\n=== REDE (LOOCV, dissertacao) ===')
d = pickle.load(open('/home/claude/run/predicoes.pkl','rb'))
pr = d['predicoes']; res = d['resultados']
for v in ('q','p','ev'):
    y = np.concatenate([p[f'{v}_real'] for p in pr.values()])
    yh = np.concatenate([p[f'{v}_pred'] for p in pr.values()])
    rmse = np.sqrt(np.mean((y-yh)**2))
    if v == 'ev': rmse *= 100
    med = res[{'q':'R2_q','p':'R2_p','ev':'R2_ev'}[v]].median()
    print(f'  {v:3s}: R2 agregado = {r2_score(y,yh):7.4f}   RMSE = {rmse:8.4g}'
          f'{" %" if v=="ev" else " kPa"}   mediana por ensaio = {med:.3f}')
