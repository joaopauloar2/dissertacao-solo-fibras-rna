"""
============================================================================
GERAÇÃO AUTOMATIZADA DE DADOS SINTÉTICOS — TIER 1 (versão xlwings)
Modelo Constitutivo MLP para Areias Reforçadas com Fibras Poliméricas

Uso ALTERNATIVO ao script principal: requer Microsoft Excel instalado
(Windows ou Mac). É geralmente mais rápido e estável que LibreOffice.

Para instalar: pip install xlwings pandas openpyxl
============================================================================
"""

import os
import shutil
import tempfile
import pandas as pd
import xlwings as xw

# ============================================================================
# CONFIGURAÇÃO (idêntica ao script principal)
# ============================================================================
PLANILHA_MESTRE = "FRS_2002_mod_conceicao_Pf1.xlsx"
CSV_SAIDA = "dados_sinteticos_tier1.csv"
RAZAO_PF_PO = 2.17

# ============================================================================
# DEFINIÇÃO DOS 42 CENÁRIOS
# ============================================================================
def gerar_cenarios():
    cenarios = []
    pos = [50, 100, 150, 200, 250, 300]
    for po in pos:
        cenarios.append({
            "id": f"Pf0_L51_po{po}",
            "Pf_pct": 0.0, "L_mm": 51,
            "po_kPa": po, "pf_kPa": round(po * RAZAO_PF_PO, 0),
        })
    for pf_pct in [0.5, 1.0]:
        for L in [12.5, 25, 51]:
            for po in pos:
                cenarios.append({
                    "id": f"Pf{pf_pct}_L{L}_po{po}",
                    "Pf_pct": pf_pct, "L_mm": L,
                    "po_kPa": po, "pf_kPa": round(po * RAZAO_PF_PO, 0),
                })
    return cenarios

# ============================================================================
# PIPELINE
# ============================================================================
def main():
    cenarios = gerar_cenarios()
    print(f"Total de cenários: {len(cenarios)}")
    todos_dados = []
    log = []

    # Abrir Excel UMA VEZ (muito mais rápido que abrir/fechar)
    app = xw.App(visible=False, add_book=False)
    app.display_alerts = False

    try:
        for i, cenario in enumerate(cenarios, 1):
            print(f"\n[{i:>2}/{len(cenarios)}] {cenario['id']}: "
                  f"Pf={cenario['Pf_pct']}%, L={cenario['L_mm']}mm, "
                  f"po={cenario['po_kPa']}, pf={cenario['pf_kPa']}")

            wb = app.books.open(os.path.abspath(PLANILHA_MESTRE))
            ws = wb.sheets["A"]

            # Definir parâmetros — Excel recalcula AUTOMATICAMENTE
            ws.range("C12").value = cenario["Pf_pct"]
            ws.range("C22").value = cenario["L_mm"]
            ws.range("D3").value = cenario["po_kPa"]
            ws.range("D7").value = cenario["pf_kPa"]
            app.calculate()  # garante recálculo

            # Ler bloco de dados (linhas 58-850, colunas A:W)
            dados = ws.range("A58:W850").value

            wb.close()

            # Processar dados
            linhas = []
            for row in dados:
                if row is None or row[0] is None:
                    continue
                try:
                    p = float(row[0])
                    q = float(row[1]) if row[1] is not None else 0
                    v_msw = float(row[18]) if row[18] is not None else None
                    ev = float(row[19]) if row[19] is not None else None
                    ea = float(row[22]) if row[22] is not None else None
                except (TypeError, ValueError):
                    continue
                if v_msw is None or ev is None or ea is None:
                    continue
                linhas.append({
                    "p_tensao_media": p,
                    "q_tensao_desviadora": q,
                    "e_indice_de_vazios": v_msw - 1,
                    "ea_deformacao_axial": ea,
                    "ev_deformacao_volumetrica": ev,
                })

            df = pd.DataFrame(linhas)
            if len(df) == 0:
                log.append({**cenario, "n_pontos": 0, "status": "vazio"})
                continue

            df["po_kpa"] = cenario["po_kPa"]
            df["teor_de_fibra"] = cenario["Pf_pct"]
            df["comp_de_fibra"] = cenario["L_mm"]
            df["e_inicial_vazio_inicial"] = df["e_indice_de_vazios"].iloc[0]
            df["deltaEa"] = df["ea_deformacao_axial"].diff().fillna(0)
            df["deltaev"] = df["ev_deformacao_volumetrica"].diff().fillna(0)
            df["dq"] = df["q_tensao_desviadora"].diff().fillna(0)
            df["dp"] = df["p_tensao_media"].diff().fillna(0)

            # Remover padding
            mask = (df["deltaEa"].abs() < 1e-10) & (df["dq"].abs() < 1e-10)
            mask.iloc[0] = False
            df = df[~mask].reset_index(drop=True)

            colunas = ["po_kpa","teor_de_fibra","comp_de_fibra","e_inicial_vazio_inicial",
                       "p_tensao_media","q_tensao_desviadora","e_indice_de_vazios",
                       "ea_deformacao_axial","ev_deformacao_volumetrica",
                       "deltaEa","deltaev","dq","dp"]
            df = df[colunas]

            print(f"        ✓ {len(df)} pontos extraídos")
            todos_dados.append(df)
            log.append({**cenario, "n_pontos": len(df), "status": "ok"})

    finally:
        app.quit()

    # Consolidar
    if todos_dados:
        df_final = pd.concat(todos_dados, ignore_index=True)
        df_final.to_csv(CSV_SAIDA, index=False)
        pd.DataFrame(log).to_csv(CSV_SAIDA.replace(".csv","_log.csv"), index=False)
        print(f"\n✓ CSV salvo: {CSV_SAIDA} ({len(df_final)} pontos)")


if __name__ == "__main__":
    main()
