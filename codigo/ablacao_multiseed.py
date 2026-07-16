# -*- coding: utf-8 -*-
"""
ABLACAO MULTISSEMENTE: pre-treinamento sintetico x treino do zero.

Para cada uma de 5 sementes (42, 7, 123, 2024, 31), executa os dois bracos com
o MESMO protocolo LOOCV de 24 folds:

  BRACO A (com pre-treinamento): reproduz exatamente o pipeline da dissertacao
    (`modelo_constitutivo_MLP_v9.py`): pre-treinamento nos 19.513 pontos
    sinteticos, scaler ajustado nos sinteticos, ajuste fino com descongelamento
    progressivo em 4 fases, paciencia 100, teto 400 epocas.

  BRACO B (sem pre-treinamento): inicializacao aleatoria, treino apenas com os
    23 ensaios experimentais do fold. Sem congelamento (nao faz sentido sem
    transferencia). Adam lr = 1e-3 com ReduceLROnPlateau, teto 500 epocas
    (maior que o do braco A), paciencia 100. Scaler ajustado nos 23 ensaios de
    treino do fold, pois sem modelo analitico nao ha sinteticos para ajusta-lo.

Tudo o mais e identico nos dois bracos: arquitetura 5-256-128-64-3 com ELU,
perda ponderada (wq=2, wp'=1, wev=3), batch 32 no ajuste, mesma limpeza dos
dados, mesmos 24 folds, mesmas metricas, mesma agregacao (concatenacao dos
1.207 pares previsto-observado).

Com a semente 42, o braco A deve reproduzir o resultado da dissertacao
(R2 agregado: q = 0,9765; p' = 0,9914; ev = 0,9143), o que serve de aferição.

Saida: ablacao_multiseed.csv (metricas por semente e braco) e
       ablacao_multiseed_folds.csv (metricas por fold, semente e braco).
"""

import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cpu')
SEMENTES = [42, 7, 123, 2024, 31]

PATIENCE_PRE, MAX_EPOCAS_PRE, BATCH_PRE = 40, 500, 256
PATIENCE_FT, MAX_EPOCAS_FT, BATCH_FT = 100, 400, 32
MAX_EPOCAS_ZERO = 500

FEATURES = ['po_kpa', 'teor_de_fibra', 'tem_fibra', 'comp_de_fibra',
            'ea_deformacao_axial']
TARGETS = ['q_tensao_desviadora', 'p_tensao_media', 'ev_deformacao_volumetrica']

# ---------------------------------------------------------------------------
# Dados (limpeza identica ao v9)
# ---------------------------------------------------------------------------
sin = pd.read_csv('/mnt/project/dados_sinteticos_tier1_filtrado.csv')
exp = pd.read_csv('/mnt/project/dados_experimentais_tier1.csv')
for df in (sin, exp):
    df.loc[df['teor_de_fibra'] == 0, 'comp_de_fibra'] = 0.0
    df['tem_fibra'] = (df['teor_de_fibra'] > 0).astype(float)
sin = sin[~((sin['deltaEa'] == 0) & (sin['dq'] == 0))].reset_index(drop=True)
sin = sin[(sin['ea_deformacao_axial'] >= 0) &
          (sin['ea_deformacao_axial'] <= 0.25)].reset_index(drop=True)
exp = exp[~(exp['deltaEa'] < -0.05)].reset_index(drop=True)
print(f'sinteticos: {len(sin)} | experimentais: {len(exp)}', flush=True)

ENSAIOS = (exp.groupby(['po_kpa', 'teor_de_fibra', 'comp_de_fibra'])
           .size().reset_index(name='n'))

# scaler do braco A: ajustado SOMENTE nos sinteticos (como no v9)
sxA, syA = StandardScaler(), StandardScaler()
X_sin = sxA.fit_transform(sin[FEATURES].values)
y_sin = syA.fit_transform(sin[TARGETS].values)
X_expA = sxA.transform(exp[FEATURES].values)
y_expA = syA.transform(exp[TARGETS].values)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        camadas, prev = [], len(FEATURES)
        for h in (256, 128, 64):
            camadas += [nn.Linear(prev, h), nn.ELU(alpha=1.0)]
            prev = h
        camadas.append(nn.Linear(prev, len(TARGETS)))
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        return self.rede(x)


PESOS = torch.FloatTensor([2.0, 1.0, 3.0])


def perda(pred, alvo):
    return (((pred - alvo) ** 2).mean(dim=0) * PESOS).mean()


mse = nn.MSELoss()


def dl(X, y, bs):
    return DataLoader(TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)),
                      batch_size=bs, shuffle=True)


def epoca(modelo, loader, opt):
    modelo.train()
    tot = n = 0
    for Xb, yb in loader:
        opt.zero_grad()
        L = perda(modelo(Xb), yb)
        L.backward(); opt.step()
        tot += L.item(); n += 1
    return tot / n


def val_loss(modelo, Xt, yt):
    modelo.eval()
    with torch.no_grad():
        return mse(modelo(Xt), yt).item()


def agregar(pred):
    """R2 agregado sobre a concatenacao dos 1.207 pares."""
    out = {}
    for j, v in enumerate(('q', 'p', 'ev')):
        y = np.concatenate([p[0][:, j] for p in pred])
        yh = np.concatenate([p[1][:, j] for p in pred])
        out[v] = r2_score(y, yh)
        out['rmse_' + v] = float(np.sqrt(np.mean((y - yh) ** 2)))
    return out


linhas, linhas_fold = [], []
t_ini = time.time()

for semente in SEMENTES:
    print('\n' + '=' * 60, flush=True)
    print(f'SEMENTE {semente}', flush=True)
    print('=' * 60, flush=True)

    # -----------------------------------------------------------------------
    # BRACO A: pre-treinamento
    # -----------------------------------------------------------------------
    torch.manual_seed(semente); np.random.seed(semente)
    m_pre = MLP()
    loader = dl(X_sin, y_sin, BATCH_PRE)
    Xv = torch.FloatTensor(X_expA); yv = torch.FloatTensor(y_expA)
    opt = torch.optim.Adam(m_pre.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=20)
    melhor, estado, sem_melhora = float('inf'), None, 0
    for ep in range(MAX_EPOCAS_PRE):
        epoca(m_pre, loader, opt)
        lv = val_loss(m_pre, Xv, yv)
        sch.step(lv)
        if lv < melhor:
            melhor, estado, sem_melhora = lv, deepcopy(m_pre.state_dict()), 0
        else:
            sem_melhora += 1
        if sem_melhora >= PATIENCE_PRE:
            break
    print(f'  pre-treino: {ep + 1} epocas, perda exp = {melhor:.4f}', flush=True)

    predA, predB = [], []

    for fi, row in ENSAIOS.iterrows():
        po, pf, lf = int(row.po_kpa), float(row.teor_de_fibra), float(row.comp_de_fibra)
        k = fi + 1
        mask = ((exp['po_kpa'] == po) & (exp['teor_de_fibra'] == pf) &
                (exp['comp_de_fibra'] == lf)).values
        i_ft, i_val = np.where(~mask)[0], np.where(mask)[0]

        # ---- BRACO A: ajuste fino com descongelamento progressivo ----------
        torch.manual_seed(semente); np.random.seed(semente)
        mA = MLP(); mA.load_state_dict(estado)
        lo = dl(X_expA[i_ft], y_expA[i_ft], BATCH_FT)
        Xv = torch.FloatTensor(X_expA[i_val]); yv = torch.FloatTensor(y_expA[i_val])
        melhorA, estA, semA = float('inf'), None, 0
        for ep in range(MAX_EPOCAS_FT):
            if ep < 30:
                for n_, p_ in mA.named_parameters():
                    p_.requires_grad = 'rede.6' in n_
                lr = 1e-3
            elif ep < 60:
                for n_, p_ in mA.named_parameters():
                    p_.requires_grad = ('rede.4' in n_ or 'rede.6' in n_)
                lr = 5e-4
            elif ep < 100:
                for n_, p_ in mA.named_parameters():
                    p_.requires_grad = ('rede.2' in n_ or 'rede.4' in n_ or 'rede.6' in n_)
                lr = 1e-4
            else:
                for p_ in mA.parameters():
                    p_.requires_grad = True
                lr = 5e-5
            o = torch.optim.Adam(filter(lambda p: p.requires_grad, mA.parameters()), lr=lr)
            epoca(mA, lo, o)
            lv = val_loss(mA, Xv, yv)
            if lv < melhorA:
                melhorA, estA, semA = lv, deepcopy(mA.state_dict()), 0
            else:
                semA += 1
            if semA >= PATIENCE_FT:
                break
        mA.load_state_dict(estA); mA.eval()
        with torch.no_grad():
            pn = mA(Xv).numpy()
        yA = syA.inverse_transform(y_expA[i_val]); pA = syA.inverse_transform(pn)
        predA.append((yA, pA))

        # ---- BRACO B: do zero, so com os 23 ensaios ------------------------
        sxB, syB = StandardScaler(), StandardScaler()
        XftB = sxB.fit_transform(exp.iloc[i_ft][FEATURES].values)
        yftB = syB.fit_transform(exp.iloc[i_ft][TARGETS].values)
        XvB = sxB.transform(exp.iloc[i_val][FEATURES].values)
        yvB = syB.transform(exp.iloc[i_val][TARGETS].values)
        torch.manual_seed(semente); np.random.seed(semente)
        mB = MLP()
        lo = dl(XftB, yftB, BATCH_FT)
        XvBt = torch.FloatTensor(XvB); yvBt = torch.FloatTensor(yvB)
        o = torch.optim.Adam(mB.parameters(), lr=1e-3)
        sc = torch.optim.lr_scheduler.ReduceLROnPlateau(o, factor=0.5, patience=20)
        melhorB, estB, semB = float('inf'), None, 0
        for ep in range(MAX_EPOCAS_ZERO):
            epoca(mB, lo, o)
            lv = val_loss(mB, XvBt, yvBt)
            sc.step(lv)
            if lv < melhorB:
                melhorB, estB, semB = lv, deepcopy(mB.state_dict()), 0
            else:
                semB += 1
            if semB >= PATIENCE_FT:
                break
        mB.load_state_dict(estB); mB.eval()
        with torch.no_grad():
            pn = mB(XvBt).numpy()
        yB = syB.inverse_transform(yvB); pB = syB.inverse_transform(pn)
        predB.append((yB, pB))

        for braco, yy, pp in (('com', yA, pA), ('sem', yB, pB)):
            linhas_fold.append(dict(
                semente=semente, braco=braco, fold=k, po_kpa=po,
                teor_de_fibra=pf, comp_de_fibra=lf,
                R2_q=r2_score(yy[:, 0], pp[:, 0]),
                R2_p=r2_score(yy[:, 1], pp[:, 1]),
                R2_ev=r2_score(yy[:, 2], pp[:, 2])))

        print(f'  [{k:2d}/24] com: q={r2_score(yA[:,0],pA[:,0]):+.3f}  '
              f'sem: q={r2_score(yB[:,0],pB[:,0]):+.3f}', flush=True)

    for braco, pred in (('com', predA), ('sem', predB)):
        a = agregar(pred)
        df_f = pd.DataFrame([l for l in linhas_fold
                             if l['semente'] == semente and l['braco'] == braco])
        linhas.append(dict(semente=semente, braco=braco,
                           R2ag_q=a['q'], R2ag_p=a['p'], R2ag_ev=a['ev'],
                           RMSE_q=a['rmse_q'], RMSE_p=a['rmse_p'],
                           RMSE_ev=a['rmse_ev'],
                           med_q=df_f.R2_q.median(), med_p=df_f.R2_p.median(),
                           med_ev=df_f.R2_ev.median()))
        print(f'  >> {braco.upper()}: R2ag q={a["q"]:.4f} p={a["p"]:.4f} '
              f'ev={a["ev"]:.4f} | mediana q={df_f.R2_q.median():.3f} '
              f'p={df_f.R2_p.median():.3f}', flush=True)

    pd.DataFrame(linhas).to_csv('/home/claude/run/ablacao_multiseed.csv', index=False)
    pd.DataFrame(linhas_fold).to_csv('/home/claude/run/ablacao_multiseed_folds.csv', index=False)

print(f'\nTempo total: {(time.time() - t_ini) / 60:.1f} min')
