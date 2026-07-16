# Modelo constitutivo para areia reforçada com fibras via redes neurais

Código, dados e resultados da dissertação de mestrado *Desenvolvimento de um
Modelo Constitutivo para Solos Arenosos Reforçados com Fibras utilizando Redes
Neurais Artificiais* (PPGEC/FECIV/UFU).

A abordagem combina um pré-treinamento com dados sintéticos, gerados por um
modelo analítico do tipo Cam-Clay modificado, com um ajuste fino sobre 24
ensaios triaxiais experimentais, avaliado por validação cruzada *leave-one-out*
(LOOCV).

## Organização

```
.
├── dados/
│   ├── dados_experimentais_tier1.csv        Ensaios triaxiais (Conceição et al., 2022)
│   ├── dados_sinteticos_tier1_filtrado.csv  Trajetórias do modelo analítico (após limpeza)
│   ├── FRS_2002_mod_conceicao_Pf1.xlsx       Planilha-fonte do modelo analítico (Pf = 1,0 %)
│   └── FRS_2002_mod_conceicao_pf05.xlsx      Planilha-fonte do modelo analítico (Pf = 0,5 %)
│
├── codigo/
│   ├── modelo_constitutivo_MLP_v9.py         Script principal: pré-treinamento + ajuste fino + LOOCV
│   ├── gerar_dados_sinteticos_tier1_xlwings.py  Geração dos dados sintéticos a partir das planilhas
│   ├── extrair_dados_experimentais.py        Extração e preparação dos dados experimentais
│   ├── verificar_trajetorias_sinteticas.py   Verificação das trajetórias sintéticas
│   ├── linha_base_analitica.py               Desempenho do modelo analítico como linha de base (Seção 4.6)
│   └── ablacao_multiseed.py                  Estudo de ablação, com e sem pré-treinamento (Seção 4.7)
│
└── resultados/
    ├── loocv_resultados.csv                  Métricas por ensaio da validação cruzada
    ├── fig1_loocv_boxplot.png                Distribuição do R² por ensaio
    ├── fig2_loocv_curvas_repr.png            Curvas dos ensaios representativos
    ├── fig3_loocv_dispersao.png              Dispersão previsto x observado
    ├── fig4_loocv_historico.png              Evolução da função de perda
    ├── curvas_sinteticas_verificacao.png     Verificação das curvas sintéticas
    ├── loocv_predicoes.npz                   previsões ponto a ponto da validação cruzada
    └── ablacao_e_linha_base/
        ├── linha_base_analitica.csv          R² por ensaio do modelo analítico
        ├── ablacao_5sementes.csv             Resumo da ablação (5 sementes x 2 configurações)
        └── ablacao_5sementes_folds.csv       Detalhamento por fold da ablação (240 linhas)
```

## Como reproduzir

Requisitos: Python 3, PyTorch, NumPy, pandas e scikit-learn.

1. **Validação cruzada principal** (reproduz `loocv_resultados.csv`):
   ```
   python codigo/modelo_constitutivo_MLP_v9.py
   ```
   Com semente 42, os resultados por ensaio coincidem com os do arquivo.

2. **Linha de base analítica** (Seção 4.6):
   ```
   python codigo/linha_base_analitica.py
   ```
   Avalia o modelo analítico sobre os 1.207 pontos experimentais.

3. **Estudo de ablação** (Seção 4.7):
   ```
   python codigo/ablacao_multiseed.py
   ```
   Executa a validação cruzada com e sem o pré-treinamento, para cinco
   sementes independentes. Como aferição, a configuração com pré-treinamento
   e semente 42 reproduz `loocv_resultados.csv`.

Os scripts de `codigo/` esperam os arquivos de `dados/`. Ajuste os caminhos no
início de cada script conforme a organização local, se necessário.

## Dados de terceiros

Os dados experimentais derivam dos ensaios de Conceição et al. (2022) e o
modelo analítico que gera os dados sintéticos é de Machado et al. (2024). O
crédito por essas fontes é dos respectivos autores; consulte a dissertação para
as referências completas.

## Licença

Código sob licença MIT (ver `LICENSE`). Os dados são disponibilizados para fins
de reprodutibilidade acadêmica; para outros usos, consulte as fontes originais.
