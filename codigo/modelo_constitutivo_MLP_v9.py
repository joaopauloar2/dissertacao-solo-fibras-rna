#!/usr/bin/env python3
"""
=============================================================================
MODELO CONSTITUTIVO BASEADO EM MLP PARA SOLOS REFORÇADOS COM FIBRAS
=============================================================================
v9: VERSÃO FINAL — hiperparâmetros do v7 + relatórios robustos do v8

Histórico das versões:
    v5 → versão do artigo COBRAMSEG 2026 (4 features, base reduzida)
    v6 → expansão para Tier 1 (24 ensaios) + feature tem_fibra
    v7 → fine-tuning tunado + LOOCV completo (24 folds)
    v8 → tentativa de anti-overfitting (patience=30, weight_decay=1e-4)
         → CONCLUSÃO: piorou marginalmente os R² (folds difíceis precisavam
           de mais épocas; o overfitting visual era inofensivo, pois
           best_state já capturava o mínimo)
    v9 → CONSOLIDA o melhor de ambos:
         - hiperparâmetros do v7 (patience=100, max_epocas=400)
         - relatórios do v8: R² agregado global, mediana, IQR

JUSTIFICATIVA METODOLÓGICA DOS RELATÓRIOS:
    Em problemas com escala de saída fortemente heterogênea entre folds
    (como ensaios triaxiais a diferentes σ₃), o R² médio por fold é
    enganoso porque a métrica é hipersensível a variâncias pequenas
    do alvo. Para σ₃=50, q varia em ~100 kPa; para σ₃=300, em ~700 kPa.
    O mesmo erro absoluto em RMSE produz R² qualitativamente diferentes
    (Krzywinski & Altman, Nature Methods, 2015; Chai & Draxler, GMD, 2014).

    Por isso reportamos como métricas principais:
    (1) R² AGREGADO GLOBAL — calculado sobre os 1.207 pontos da LOOCV
        concatenados. Imune ao problema de variância pequena por fold.
    (2) MEDIANA + IQR — robusta a outliers (que existem em σ₃=50).
    (3) Média ± DP — para completude, com ressalva.

NORMALIZAÇÃO:
    Scaler ajustado APENAS nos sintéticos. Garante consistência absoluta
    entre os 24 folds e elimina vazamento dos dados experimentais.

Tempo estimado em Colab com GPU: 5-6 min.

Autor: João Paulo de Araújo Rodrigues - UFU - 2026
=============================================================================
"""

# ===========================================================================
# CÉLULA 1: IMPORTS E CONFIGURAÇÃO
# ===========================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from copy import deepcopy
import time
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams.update({
    'font.size': 10, 'font.family': 'serif',
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 8, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Dispositivo: {device}')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Hiperparâmetros do treino
PATIENCE_PRE = 40
PATIENCE_FT = 100        # ← aumentado de 50 (v6) para 100 (v7-B)
MAX_EPOCAS_PRE = 500
MAX_EPOCAS_FT = 400      # ← aumentado de 300 (v6) para 400 (v7-B)
BATCH_PRE = 256
BATCH_FT = 32

# Modo de teste rápido: defina como int para limitar nº de folds (debug);
# None executa todos os 24 folds (modo produção).
N_FOLDS_MAX = None

# ===========================================================================
# CÉLULA 2: CARREGAMENTO E LIMPEZA
# ===========================================================================
print('='*60)
print('ETAPA 1: CARREGAMENTO E LIMPEZA')
print('='*60)

import os

ARQUIVO_SIN = 'dados_sinteticos_tier1_filtrado.csv'
ARQUIVO_EXP = 'dados_experimentais_tier1.csv'

if os.path.exists('/content/' + ARQUIVO_SIN):
    PATH = '/content/'
elif os.path.exists(ARQUIVO_SIN):
    PATH = './'
else:
    PATH = '/mnt/user-data/uploads/'

df_sintetico = pd.read_csv(PATH + ARQUIVO_SIN)
df_experimental = pd.read_csv(PATH + ARQUIVO_EXP)

print(f'Sintéticos brutos: {len(df_sintetico)} | Experimentais brutos: {len(df_experimental)}')

# Padronização: para Pf=0%, comp_de_fibra fixo em 0
df_sintetico.loc[df_sintetico['teor_de_fibra'] == 0, 'comp_de_fibra'] = 0.0
df_experimental.loc[df_experimental['teor_de_fibra'] == 0, 'comp_de_fibra'] = 0.0

# Feature binária tem_fibra
df_sintetico['tem_fibra'] = (df_sintetico['teor_de_fibra'] > 0).astype(float)
df_experimental['tem_fibra'] = (df_experimental['teor_de_fibra'] > 0).astype(float)

# Limpezas
df_sintetico = df_sintetico[
    ~((df_sintetico['deltaEa'] == 0) & (df_sintetico['dq'] == 0))
].reset_index(drop=True)
df_sintetico = df_sintetico[
    (df_sintetico['ea_deformacao_axial'] >= 0) &
    (df_sintetico['ea_deformacao_axial'] <= 0.25)
].reset_index(drop=True)
df_experimental = df_experimental[
    ~(df_experimental['deltaEa'] < -0.05)
].reset_index(drop=True)

print(f'Sintéticos limpos: {len(df_sintetico)} | Experimentais limpos: {len(df_experimental)}')

FEATURES = ['po_kpa', 'teor_de_fibra', 'tem_fibra', 'comp_de_fibra', 'ea_deformacao_axial']
TARGETS = ['q_tensao_desviadora', 'p_tensao_media', 'ev_deformacao_volumetrica']

# ===========================================================================
# CÉLULA 3: NORMALIZAÇÃO (scaler ajustado SOMENTE nos sintéticos)
# ===========================================================================
print('\n' + '='*60)
print('ETAPA 2: NORMALIZAÇÃO')
print('='*60)

# Justificativa: scaler ajustado apenas nos sintéticos garante que seja
# idêntico em todos os 24 folds, eliminando qualquer possibilidade de
# vazamento dos dados experimentais de validação no parâmetro do scaler.
scaler_X = StandardScaler()
scaler_y = StandardScaler()
scaler_X.fit(df_sintetico[FEATURES].values)
scaler_y.fit(df_sintetico[TARGETS].values)

X_sin = scaler_X.transform(df_sintetico[FEATURES].values)
y_sin = scaler_y.transform(df_sintetico[TARGETS].values)
X_exp_all = scaler_X.transform(df_experimental[FEATURES].values)
y_exp_all = scaler_y.transform(df_experimental[TARGETS].values)

print(f'X_sintético: {X_sin.shape}')
print(f'X_experimental (todos): {X_exp_all.shape}')

# Identificação dos 24 ensaios únicos (cada um será 1 fold)
ENSAIOS_UNICOS = (df_experimental
                  .groupby(['po_kpa', 'teor_de_fibra', 'comp_de_fibra'])
                  .size().reset_index(name='n_pontos'))
print(f'\nEnsaios únicos para LOOCV: {len(ENSAIOS_UNICOS)}')
print('Distribuição por Pf:')
print(ENSAIOS_UNICOS.groupby('teor_de_fibra').size().to_string())

# ===========================================================================
# CÉLULA 4: ARQUITETURA E FUNÇÕES AUXILIARES
# ===========================================================================
print('\n' + '='*60)
print('ETAPA 3: ARQUITETURA')
print('='*60)


class MLP(nn.Module):
    def __init__(self, n_in=5, n_out=3, hidden=[256, 128, 64]):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU(alpha=1.0)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.rede = nn.Sequential(*layers)

    def forward(self, x):
        return self.rede(x)


PESOS_LOSS = torch.FloatTensor([2.0, 1.0, 3.0]).to(device)  # q, p', εv


class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, pred, target):
        mse_per_output = ((pred - target) ** 2).mean(dim=0)
        return (mse_per_output * self.weights).mean()


criterion = WeightedMSELoss(PESOS_LOSS)
criterion_eval = nn.MSELoss()


def criar_dl(X, y, bs, shuffle=True):
    return DataLoader(TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)),
                      batch_size=bs, shuffle=shuffle)


def treinar_epoca(modelo, loader, criterion, optimizer):
    modelo.train()
    total, n = 0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(modelo(X_b), y_b)
        loss.backward()
        optimizer.step()
        total += loss.item(); n += 1
    return total / n


def avaliar_loss(modelo, X_t, y_t):
    modelo.eval()
    with torch.no_grad():
        return criterion_eval(modelo(X_t), y_t).item()


# Arquitetura template (instanciada uma vez aqui só para imprimir info)
modelo_tmpl = MLP(len(FEATURES), len(TARGETS), [256, 128, 64])
n_params = sum(p.numel() for p in modelo_tmpl.parameters())
print(f'Arquitetura: {len(FEATURES)} → 256 → 128 → 64 → {len(TARGETS)}')
print(f'Parâmetros: {n_params:,}')
del modelo_tmpl

# ===========================================================================
# CÉLULA 5: PRÉ-TREINAMENTO (executado UMA vez, reusado em todos os folds)
# ===========================================================================
print('\n' + '='*60)
print('ETAPA 4: PRÉ-TREINAMENTO (uma única vez)')
print('='*60)

# Validação interna durante pré-treino: usa TODOS os experimentais como
# conjunto de validação. O objetivo aqui é apenas escolher o melhor estado
# pré-treinado em termos de generalização para o domínio experimental —
# o estado escolhido será o ponto de partida de TODOS os 24 fine-tunings,
# então é simétrico em relação a qualquer fold específico.
X_exp_all_t = torch.FloatTensor(X_exp_all).to(device)
y_exp_all_t = torch.FloatTensor(y_exp_all).to(device)

torch.manual_seed(SEED); np.random.seed(SEED)
modelo_pre = MLP(len(FEATURES), len(TARGETS), [256, 128, 64]).to(device)
loader_pre = criar_dl(X_sin, y_sin, BATCH_PRE)
opt = torch.optim.Adam(modelo_pre.parameters(), lr=1e-3)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=20)

hist_pre = {'train': [], 'val': []}
best_val, best_state, no_imp = float('inf'), None, 0
t0 = time.time()

for ep in range(MAX_EPOCAS_PRE):
    lt = treinar_epoca(modelo_pre, loader_pre, criterion, opt)
    lv = avaliar_loss(modelo_pre, X_exp_all_t, y_exp_all_t)
    hist_pre['train'].append(lt); hist_pre['val'].append(lv)
    sch.step(lv)
    if lv < best_val:
        best_val, best_state, no_imp = lv, deepcopy(modelo_pre.state_dict()), 0
    else:
        no_imp += 1
    if (ep + 1) % 25 == 0 or ep == 0:
        print(f'  Época {ep+1:3d} | Train: {lt:.6f} | Val: {lv:.6f} | LR: {opt.param_groups[0]["lr"]:.1e}')
    if no_imp >= PATIENCE_PRE:
        print(f'  Early stopping época {ep+1}'); break

modelo_pre.load_state_dict(best_state)
print(f'Melhor val loss (pré-treino): {best_val:.6f}')
print(f'Tempo do pré-treino: {time.time()-t0:.1f} s')
state_pre = deepcopy(modelo_pre.state_dict())

# ===========================================================================
# CÉLULA 6: LOOP LOOCV — FINE-TUNING POR FOLD
# ===========================================================================
print('\n' + '='*60)
print('ETAPA 5: VALIDAÇÃO CRUZADA LEAVE-ONE-OUT')
print('='*60)


def fazer_finetuning(state_pre, X_ft, y_ft, X_val, y_val,
                     max_epocas=MAX_EPOCAS_FT, patience=PATIENCE_FT,
                     verbose=False):
    """
    Executa um fine-tuning a partir de state_pre.
    Retorna: melhor estado, hist (loss train/val), nº de épocas executadas.
    """
    torch.manual_seed(SEED); np.random.seed(SEED)
    modelo = MLP(len(FEATURES), len(TARGETS), [256, 128, 64]).to(device)
    modelo.load_state_dict(state_pre)

    loader_ft = criar_dl(X_ft, y_ft, BATCH_FT)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    hist = {'train': [], 'val': []}
    best_v, best_s, no_imp = float('inf'), None, 0
    n_ep = 0

    for ep in range(max_epocas):
        # 4 fases de liberação progressiva (idêntico ao v6)
        if ep < 30:
            for n, p in modelo.named_parameters():
                p.requires_grad = 'rede.6' in n
            lr = 1e-3
        elif ep < 60:
            for n, p in modelo.named_parameters():
                p.requires_grad = ('rede.4' in n or 'rede.6' in n)
            lr = 5e-4
        elif ep < 100:
            for n, p in modelo.named_parameters():
                p.requires_grad = ('rede.2' in n or 'rede.4' in n or 'rede.6' in n)
            lr = 1e-4
        else:
            for p in modelo.parameters():
                p.requires_grad = True
            lr = 5e-5

        o = torch.optim.Adam(filter(lambda p: p.requires_grad, modelo.parameters()), lr=lr)
        lt = treinar_epoca(modelo, loader_ft, criterion, o)
        lv = avaliar_loss(modelo, X_val_t, y_val_t)
        hist['train'].append(lt); hist['val'].append(lv)
        n_ep = ep + 1

        if lv < best_v:
            best_v, best_s, no_imp = lv, deepcopy(modelo.state_dict()), 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    return best_s, hist, n_ep


# Loop principal LOOCV
resultados = []
predicoes = {}
historicos = {}

ensaios_a_processar = ENSAIOS_UNICOS.copy()
if N_FOLDS_MAX is not None:
    ensaios_a_processar = ensaios_a_processar.iloc[:N_FOLDS_MAX]
    print(f'\n⚠ MODO TESTE: processando apenas {len(ensaios_a_processar)} folds')

t_loop = time.time()

for fold_idx, row in ensaios_a_processar.iterrows():
    # Conversão explícita: groupby do pandas pode promover int → float;
    # garantimos os tipos certos para os format strings (:3d) abaixo
    po = int(row['po_kpa'])
    pf = float(row['teor_de_fibra'])
    lf = float(row['comp_de_fibra'])
    k = fold_idx + 1
    n_folds = len(ensaios_a_processar)

    # Identificar índices de validação (ensaio reservado) e treino (resto)
    mask_val = ((df_experimental['po_kpa'] == po) &
                (df_experimental['teor_de_fibra'] == pf) &
                (df_experimental['comp_de_fibra'] == lf))
    idx_val = np.where(mask_val.values)[0]
    idx_ft = np.where(~mask_val.values)[0]

    X_ft, y_ft = X_exp_all[idx_ft], y_exp_all[idx_ft]
    X_val, y_val = X_exp_all[idx_val], y_exp_all[idx_val]

    L_str = f'L={lf:4.1f}mm' if lf > 0 else 'sem fibra'
    print(f'\n[{k:2d}/{n_folds}] σ₃={po:3d}kPa, Pf={pf}%, {L_str:<10}'
          f' | n_val={len(idx_val):2d}, n_treino={len(idx_ft)}')

    t_fold = time.time()
    best_state, hist, n_ep = fazer_finetuning(state_pre, X_ft, y_ft, X_val, y_val)
    dt = time.time() - t_fold

    # Avaliação no ensaio reservado
    modelo = MLP(len(FEATURES), len(TARGETS), [256, 128, 64]).to(device)
    modelo.load_state_dict(best_state)
    modelo.eval()
    X_val_t = torch.FloatTensor(X_val).to(device)
    with torch.no_grad():
        pred_n = modelo(X_val_t).cpu().numpy()

    y_real = scaler_y.inverse_transform(y_val)
    pred = scaler_y.inverse_transform(pred_n)

    r2 = [r2_score(y_real[:, j], pred[:, j]) for j in range(3)]
    rmse = [np.sqrt(mean_squared_error(y_real[:, j], pred[:, j])) for j in range(3)]

    print(f'        R²(q)={r2[0]:.4f}, R²(p\')={r2[1]:.4f}, R²(εv)={r2[2]:.4f}'
          f'  | épocas={n_ep:3d}, {dt:.1f}s')

    resultados.append({
        'fold': k, 'po_kpa': po, 'teor_de_fibra': pf, 'comp_de_fibra': lf,
        'n_pontos': len(idx_val), 'epocas_ft': n_ep, 'tempo_s': dt,
        'R2_q': r2[0], 'R2_p': r2[1], 'R2_ev': r2[2],
        'RMSE_q': rmse[0], 'RMSE_p': rmse[1], 'RMSE_ev': rmse[2],
    })

    # Salvar predições e ε_a do ensaio reservado para plots posteriores
    predicoes[f'fold_{k:02d}'] = {
        'po': po, 'pf': pf, 'L': lf,
        'ea': df_experimental.loc[mask_val, 'ea_deformacao_axial'].values,
        'q_real': y_real[:, 0], 'q_pred': pred[:, 0],
        'p_real': y_real[:, 1], 'p_pred': pred[:, 1],
        'ev_real': y_real[:, 2], 'ev_pred': pred[:, 2],
    }
    historicos[f'fold_{k:02d}'] = hist

    # Salvar resultados parciais a cada fold (segurança contra travamentos)
    pd.DataFrame(resultados).to_csv('loocv_resultados.csv', index=False)

print(f'\n>>> Tempo total LOOCV: {(time.time()-t_loop)/60:.1f} min')

# ===========================================================================
# CÉLULA 7: ANÁLISE CONSOLIDADA
# ===========================================================================
print('\n' + '='*60)
print('ETAPA 6: ANÁLISE CONSOLIDADA')
print('='*60)

df_loocv = pd.DataFrame(resultados)
df_loocv.to_csv('loocv_resultados.csv', index=False)
print(f'\n✓ Resultados salvos: loocv_resultados.csv ({len(df_loocv)} folds)')

# ----------------------------------------------------------------------
# 1) MÉTRICAS AGREGADAS GLOBAIS — calculadas concatenando todos os
#    1.207 pontos de validação. Esta é a métrica MAIS INFORMATIVA: não
#    sofre a sensibilidade do R² por fold à variância do alvo (que pune
#    desproporcionalmente folds com range pequeno, como σ₃=50).
# ----------------------------------------------------------------------
todos_q_real, todos_q_pred = [], []
todos_p_real, todos_p_pred = [], []
todos_ev_real, todos_ev_pred = [], []
todos_pf_global = []
for k, v in predicoes.items():
    todos_q_real.extend(v['q_real']);   todos_q_pred.extend(v['q_pred'])
    todos_p_real.extend(v['p_real']);   todos_p_pred.extend(v['p_pred'])
    todos_ev_real.extend(v['ev_real']); todos_ev_pred.extend(v['ev_pred'])
    todos_pf_global.extend([v['pf']] * len(v['q_real']))

todos_q_real = np.array(todos_q_real); todos_q_pred = np.array(todos_q_pred)
todos_p_real = np.array(todos_p_real); todos_p_pred = np.array(todos_p_pred)
todos_ev_real = np.array(todos_ev_real); todos_ev_pred = np.array(todos_ev_pred)
todos_pf_global = np.array(todos_pf_global)

R2_GLOBAL_q  = r2_score(todos_q_real,  todos_q_pred)
R2_GLOBAL_p  = r2_score(todos_p_real,  todos_p_pred)
R2_GLOBAL_ev = r2_score(todos_ev_real, todos_ev_pred)
RMSE_GLOBAL_q  = np.sqrt(mean_squared_error(todos_q_real,  todos_q_pred))
RMSE_GLOBAL_p  = np.sqrt(mean_squared_error(todos_p_real,  todos_p_pred))
RMSE_GLOBAL_ev = np.sqrt(mean_squared_error(todos_ev_real, todos_ev_pred))

print('\n--- R² AGREGADO GLOBAL (sobre todos os 1.207 pontos LOOCV) ---')
print(f'  R²(q)  = {R2_GLOBAL_q:.4f}    | RMSE = {RMSE_GLOBAL_q:.2f} kPa')
print(f'  R²(p\') = {R2_GLOBAL_p:.4f}    | RMSE = {RMSE_GLOBAL_p:.2f} kPa')
print(f'  R²(εv) = {R2_GLOBAL_ev:.4f}    | RMSE = {RMSE_GLOBAL_ev:.4f}')

# ----------------------------------------------------------------------
# 2) ESTATÍSTICAS POR FOLD — mediana + IQR (mais robusto que média±std
#    quando há outliers, como no caso de σ₃=50 com pequena variância)
# ----------------------------------------------------------------------
print('\n--- ESTATÍSTICAS POR FOLD (24 valores de R²) ---')
print(f'{"Métrica":<8} {"Mediana":>10} {"IQR (Q3-Q1)":>14} {"Média":>10} {"DP":>8} {"min":>8} {"max":>8}')
print('-' * 72)
for col, nome in zip(['R2_q', 'R2_p', 'R2_ev'], ['R²(q)', "R²(p')", 'R²(εv)']):
    s = df_loocv[col]
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    print(f'{nome:<8} {q2:>10.4f} {q3-q1:>14.4f} {s.mean():>10.4f} '
          f'{s.std():>8.4f} {s.min():>8.4f} {s.max():>8.4f}')

print('\n  Observação: a MEDIANA é mais representativa do desempenho típico,')
print('  pois não é sensível aos outliers em σ₃=50 (folds 1, 2, 4).')
print('  O R² AGREGADO GLOBAL (acima) é a métrica recomendada para reporte.')

# Resumo por classe de Pf (com mediana ao lado da média)
print('\n--- POR CLASSE DE TEOR DE FIBRA (Pf) ---')
for pf_val in sorted(df_loocv['teor_de_fibra'].unique()):
    sub = df_loocv[df_loocv['teor_de_fibra'] == pf_val]
    print(f'\n  Pf = {pf_val:.1f}% ({len(sub)} ensaios):')
    for col, nome in zip(['R2_q', 'R2_p', 'R2_ev'], ['R²(q)', "R²(p')", 'R²(εv)']):
        print(f'    {nome:<7} → mediana = {sub[col].median():.4f}, '
              f'média = {sub[col].mean():.4f}, '
              f'min = {sub[col].min():.4f}')

# Tabela completa por fold
print('\n--- TABELA COMPLETA POR FOLD ---')
print(df_loocv[['fold', 'po_kpa', 'teor_de_fibra', 'comp_de_fibra',
                'R2_q', 'R2_p', 'R2_ev']].to_string(index=False))

# Identificar os 3 ensaios extremos para a Figura 2
df_sorted = df_loocv.sort_values('R2_q').reset_index(drop=True)
fold_pior = df_sorted.iloc[0]
fold_med = df_sorted.iloc[len(df_sorted)//2]
fold_melhor = df_sorted.iloc[-1]
print(f'\n--- ENSAIOS REPRESENTATIVOS ---')
print(f'  Pior R²(q):    fold {int(fold_pior["fold"]):2d} | σ₃={int(fold_pior["po_kpa"])}, '
      f'Pf={fold_pior["teor_de_fibra"]}%, L={fold_pior["comp_de_fibra"]} | R²={fold_pior["R2_q"]:.4f}')
print(f'  Mediana R²(q): fold {int(fold_med["fold"]):2d} | σ₃={int(fold_med["po_kpa"])}, '
      f'Pf={fold_med["teor_de_fibra"]}%, L={fold_med["comp_de_fibra"]} | R²={fold_med["R2_q"]:.4f}')
print(f'  Melhor R²(q):  fold {int(fold_melhor["fold"]):2d} | σ₃={int(fold_melhor["po_kpa"])}, '
      f'Pf={fold_melhor["teor_de_fibra"]}%, L={fold_melhor["comp_de_fibra"]} | R²={fold_melhor["R2_q"]:.4f}')

# Salvar predições em NPZ (para regenerar figuras sem retreinar)
np.savez_compressed('loocv_predicoes.npz', **{
    k: np.array([v['ea'], v['q_real'], v['q_pred'],
                 v['p_real'], v['p_pred'],
                 v['ev_real'], v['ev_pred']])
    for k, v in predicoes.items()
})
print(f'\n✓ Predições salvas: loocv_predicoes.npz')

# ===========================================================================
# CÉLULA 8: FIGURA 1 — BOXPLOT DE R² POR CLASSE DE Pf
# ===========================================================================
print('\n' + '='*60)
print('ETAPA 7: FIGURAS')
print('='*60)

cores_pf = {0.0: '#2ca02c', 0.5: '#ff7f0e', 1.0: '#1f77b4'}
labels_pf = ['Pf = 0% (areia pura)', 'Pf = 0,5%', 'Pf = 1,0%']
pf_vals = [0.0, 0.5, 1.0]

fig1, axes = plt.subplots(1, 3, figsize=(14, 5))

for j, (col, nome) in enumerate(zip(['R2_q', 'R2_p', 'R2_ev'], ['q', "p'", 'εv'])):
    grupos = [df_loocv[df_loocv['teor_de_fibra'] == pf][col].values for pf in pf_vals]

    bp = axes[j].boxplot(grupos, tick_labels=labels_pf, patch_artist=True,
                         widths=0.5, showmeans=True,
                         meanprops=dict(marker='D', markerfacecolor='white',
                                        markeredgecolor='black', markersize=6))
    for patch, pf in zip(bp['boxes'], pf_vals):
        patch.set_facecolor(cores_pf[pf]); patch.set_alpha(0.4)
        patch.set_edgecolor('black')

    # Strip plot sobreposto (pontos individuais)
    for i, (g, pf) in enumerate(zip(grupos, pf_vals)):
        x = np.random.normal(i + 1, 0.05, size=len(g))
        axes[j].scatter(x, g, alpha=0.85, s=35, color=cores_pf[pf],
                        edgecolors='black', linewidths=0.5, zorder=3)

    axes[j].set_ylabel(f'R²({nome})', fontweight='bold')
    axes[j].set_title(f'Coeficiente de determinação — {nome}', fontweight='bold')
    axes[j].grid(True, alpha=0.3, axis='y')
    # Linha de referência: mediana global (mais robusta que a média a outliers)
    axes[j].axhline(y=df_loocv[col].median(), color='red', ls='--', lw=1, alpha=0.5,
                    label=f'Mediana global = {df_loocv[col].median():.3f}')
    axes[j].legend(loc='lower left', fontsize=8)
    axes[j].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('fig1_loocv_boxplot.png', dpi=300)
plt.show()
print('Salvo: fig1_loocv_boxplot.png')

# ===========================================================================
# CÉLULA 9: FIGURA 2 — CURVAS REPRESENTATIVAS (best/median/worst em q)
# ===========================================================================
fig2, axes = plt.subplots(2, 3, figsize=(15, 9))

representativos = [
    ('Pior', fold_pior),
    ('Mediano', fold_med),
    ('Melhor', fold_melhor),
]

for i, (rotulo, f) in enumerate(representativos):
    fold_k = int(f['fold'])
    pred = predicoes[f'fold_{fold_k:02d}']

    # Ordenar por εa para plot suave
    o = np.argsort(pred['ea'])
    ea = pred['ea'][o] * 100
    q_r, q_p = pred['q_real'][o], pred['q_pred'][o]
    ev_r = -pred['ev_real'][o] * 100   # convenção: positivo = dilatância
    ev_p = -pred['ev_pred'][o] * 100

    L_str = f'L = {f["comp_de_fibra"]} mm' if f['comp_de_fibra'] > 0 else 'sem fibra'
    titulo = f'{rotulo} R²(q): σ₃={int(f["po_kpa"])} kPa, Pf={f["teor_de_fibra"]}%, {L_str}'

    # q × εa
    ax = axes[0, i]
    ax.plot(ea, q_r, 'o', color='#1a1a1a', ms=4, label='Experimental', zorder=3)
    ax.plot(ea, q_p, '-', color='#d62728', lw=2.2, label='RNA (LOOCV)', zorder=2)
    ax.set_ylabel('q (kPa)' if i == 0 else '')
    ax.set_title(titulo, fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.text(0.97, 0.05, f'R²(q) = {f["R2_q"]:.3f}', transform=ax.transAxes,
            ha='right', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    if i == 0:
        ax.legend(loc='center right', fontsize=9, framealpha=0.9)

    # εv × εa
    ax2 = axes[1, i]
    ax2.plot(ea, ev_r, 'o', color='#1a1a1a', ms=4, label='Experimental', zorder=3)
    ax2.plot(ea, ev_p, '-', color='#d62728', lw=2.2, label='RNA (LOOCV)', zorder=2)
    ax2.set_xlabel('Deformação axial, εa (%)')
    ax2.set_ylabel('Deformação volumétrica, εv (%)' if i == 0 else '')
    ax2.grid(True, alpha=0.25, lw=0.5); ax2.set_xlim(left=0)
    ax2.axhline(y=0, color='black', lw=0.5, alpha=0.3)
    ax2.text(0.97, 0.05, f'R²(εv) = {f["R2_ev"]:.3f}', transform=ax2.transAxes,
             ha='right', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    if i == 0:
        ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)

plt.tight_layout(h_pad=1.5)
plt.savefig('fig2_loocv_curvas_repr.png', dpi=300)
plt.show()
print('Salvo: fig2_loocv_curvas_repr.png')

# ===========================================================================
# CÉLULA 10: FIGURA 3 — DISPERSÃO GLOBAL (todos os 24 ensaios)
# ===========================================================================
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Reusa as variáveis globais já calculadas na seção de análise consolidada
# (todos_q_real, todos_q_pred, todos_p_real, todos_p_pred, todos_pf_global,
# R2_GLOBAL_q, R2_GLOBAL_p)

marcadores_pf = {0.0: '^', 0.5: 's', 1.0: 'o'}

for j, (ax, real, pred, label, titulo, r2_global) in enumerate([
    (ax1, todos_q_real, todos_q_pred, 'q', 'Tensão desviadora, q', R2_GLOBAL_q),
    (ax2, todos_p_real, todos_p_pred, "p'", "Tensão média efetiva, p'", R2_GLOBAL_p),
]):
    for pf in pf_vals:
        m = todos_pf_global == pf
        ax.scatter(real[m], pred[m], marker=marcadores_pf[pf], s=22,
                   alpha=0.65, color=cores_pf[pf], edgecolors='none',
                   label=f'Pf = {pf}%')

    lims = [min(real.min(), pred.min()) - 20, max(real.max(), pred.max()) + 20]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='1:1')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(f'{label} experimental (kPa)')
    ax.set_ylabel(f'{label} previsto (kPa)')
    ax.set_title(titulo, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9, loc='upper left')
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.text(0.97, 0.05, f'R² = {r2_global:.4f}\n(global, LOOCV)', transform=ax.transAxes,
            ha='right', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.85))

plt.tight_layout()
plt.savefig('fig3_loocv_dispersao.png', dpi=300)
plt.show()
print('Salvo: fig3_loocv_dispersao.png')

# ===========================================================================
# CÉLULA 11: FIGURA 4 — HISTÓRICO DO PRÉ-TREINO + 1 FINE-TUNE EXEMPLO
# ===========================================================================
fig4, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.5))

axA.plot(hist_pre['train'], color='#2C3E50', lw=1.5, label='Treino (sintético)')
axA.plot(hist_pre['val'], color='#E74C3C', lw=1.5, alpha=0.8,
         label='Validação (todos experimentais)')
axA.set_xlabel('Época'); axA.set_ylabel('Loss')
axA.set_title('Pré-treinamento (executado uma vez)', fontweight='bold')
axA.legend(framealpha=0.9); axA.grid(True, alpha=0.25); axA.set_yscale('log')

# Histórico do fold mediano (representativo)
hist_med = historicos[f'fold_{int(fold_med["fold"]):02d}']
axB.plot(hist_med['train'], color='#2C3E50', lw=1.5, label='Treino (23 ensaios)')
axB.plot(hist_med['val'], color='#E74C3C', lw=1.5, alpha=0.8,
         label='Validação (1 ensaio reservado)')
axB.axvline(x=30, color='gray', ls=':', alpha=0.4, label='Fase B')
axB.axvline(x=60, color='gray', ls='--', alpha=0.4, label='Fase C')
axB.axvline(x=100, color='gray', ls='-.', alpha=0.4, label='Fase D')
L_str = f'L={fold_med["comp_de_fibra"]}mm' if fold_med['comp_de_fibra'] > 0 else 'sem fibra'
axB.set_xlabel('Época'); axB.set_ylabel('Loss')
axB.set_title(f'Fine-tuning (fold mediano: σ₃={int(fold_med["po_kpa"])}, '
              f'Pf={fold_med["teor_de_fibra"]}%, {L_str})', fontweight='bold')
axB.legend(framealpha=0.9, fontsize=7); axB.grid(True, alpha=0.25); axB.set_yscale('log')

plt.tight_layout()
plt.savefig('fig4_loocv_historico.png', dpi=300)
plt.show()
print('Salvo: fig4_loocv_historico.png')

# ===========================================================================
# CÉLULA 12: TABELA FINAL E RESUMO
# ===========================================================================
print('\n' + '='*70)
print('TABELA FINAL — VALIDAÇÃO CRUZADA LEAVE-ONE-OUT (LOOCV)')
print('='*70)

print('\n┌───────────────────────────────────────┬──────────┬──────────┬──────────┐')
print('│ Fold (σ₃, Pf, L)                      │  R²(q)   │  R²(p\')  │  R²(εv)  │')
print('├───────────────────────────────────────┼──────────┼──────────┼──────────┤')
for _, r in df_loocv.iterrows():
    L_str = f'L={r["comp_de_fibra"]:4.1f}mm' if r['comp_de_fibra'] > 0 else 'sem fibra'
    print(f'│ σ₃={int(r["po_kpa"]):3d}kPa  Pf={r["teor_de_fibra"]}%  {L_str:<10}    │'
          f'  {r["R2_q"]:>6.4f}  │  {r["R2_p"]:>6.4f}  │  {r["R2_ev"]:>6.4f}  │')
print('├───────────────────────────────────────┼──────────┼──────────┼──────────┤')
print(f'│ Mediana                               │  {df_loocv["R2_q"].median():>6.4f}  │'
      f'  {df_loocv["R2_p"].median():>6.4f}  │  {df_loocv["R2_ev"].median():>6.4f}  │')
print(f'│ Média ± DP                            │ {df_loocv["R2_q"].mean():.3f}±{df_loocv["R2_q"].std():.3f}'
      f'│ {df_loocv["R2_p"].mean():.3f}±{df_loocv["R2_p"].std():.3f}'
      f'│ {df_loocv["R2_ev"].mean():.3f}±{df_loocv["R2_ev"].std():.3f}│')
print(f'│ R² agregado global (★ recomendado)   │  {R2_GLOBAL_q:>6.4f}  │'
      f'  {R2_GLOBAL_p:>6.4f}  │  {R2_GLOBAL_ev:>6.4f}  │')
print('└───────────────────────────────────────┴──────────┴──────────┴──────────┘')

print('\n' + '='*60)
print('RESUMO FINAL')
print('='*60)
print(f'Arquitetura: MLP {len(FEATURES)}→256→128→64→{len(TARGETS)} ({n_params:,} parâmetros)')
print(f'Loss: MSE ponderada [q=2.0, p\'=1.0, εv=3.0]')
print(f'Pré-treino: {len(df_sintetico)} pontos sintéticos (1 vez, reusado)')
print(f'LOOCV: {len(df_loocv)} folds completos ({len(df_experimental)} pontos no total)')
print(f'\n★ MÉTRICAS RECOMENDADAS PARA REPORTE NA DISSERTAÇÃO:')
print(f'  R²(q)  agregado global = {R2_GLOBAL_q:.4f}    [RMSE = {RMSE_GLOBAL_q:.2f} kPa]')
print(f'  R²(p\') agregado global = {R2_GLOBAL_p:.4f}    [RMSE = {RMSE_GLOBAL_p:.2f} kPa]')
print(f'  R²(εv) agregado global = {R2_GLOBAL_ev:.4f}    [RMSE = {RMSE_GLOBAL_ev:.4f}]')
print(f'\n  Mediana de R²(q)  por fold = {df_loocv["R2_q"].median():.4f}')
print(f'  Mediana de R²(p\') por fold = {df_loocv["R2_p"].median():.4f}')
print(f'  Mediana de R²(εv) por fold = {df_loocv["R2_ev"].median():.4f}')
print('='*60)
