# MVP 2 — Machine Learning & Analytics: Forecasting de Demanda no contexto de S&OP

Este repositório contém o notebook utilizado no projeto de Machine Learning & Analytics aplicado ao contexto de S&OP (Sales & Operations Planning), com foco em treinar, comparar e validar estatisticamente modelos de previsão de séries temporais para as famílias de modelagem identificadas no MVP 1.

## Repositório

├── `MVP2_Machine_Learning_e_Analytics_Thiago_Dias.ipynb` → notebook com todo o código, modelagem e análises

> O dataset é carregado diretamente por URL pública dentro do próprio notebook — não é necessário download manual. Os clusters do MVP 1 são reconstruídos de forma autocontida (mesmos parâmetros, mesma seed), sem dependência do notebook anterior para execução.

## Contexto

Este projeto é a continuação direta do [MVP 1 — Análise de Dados e Boas Práticas](https://github.com/thiago-dias-paes/MVPAnalisedeDadoseBoasPraticas), que identificou 2 famílias de modelagem (clusters) a partir de perfis sazonais similares entre 50 SKUs de uma loja. O MVP 1 respondeu *"quais itens podem ser modelados conjuntamente?"*. Este MVP responde à pergunta seguinte: *"qual modelo de previsão entrega a melhor acurácia para cada família, e esse resultado é confiável e estatisticamente robusto?"*

Em S&OP, erros de previsão se propagam por toda a cadeia — compras, produção, estoque e finanças. A abordagem de modelagem por cluster com desagregação via mix histórico só se justifica se for competitiva com a modelagem individual por SKU. Este MVP testa essa premissa empiricamente, incluindo validação de robustez temporal (backtesting) e significância estatística das comparações entre modelos.

## Objetivo

Treinar, comparar e avaliar modelos de previsão de séries temporais aplicados aos 2 clusters do MVP 1, verificando se a abordagem agregada (cluster + mix histórico) supera o baseline naive e a modelagem individual por SKU — e se essa conclusão se sustenta ao longo do tempo e é estatisticamente significativa.

Este projeto testa 4 hipóteses centrais:

- **H1:** Modelos com componente sazonal explícita (Holt-Winters) capturam melhor o padrão de pico em julho do que modelos simples (SES, Holt).
- **H2:** Modelar séries agregadas por cluster produz MAPE menor do que a modelagem individual por SKU via naive sazonal.
- **H3:** A otimização de hiperparâmetros melhora o desempenho em relação ao baseline naive em pelo menos 10 pontos percentuais.
- **H4:** A limitação de desempenho do XGBoost frente aos modelos estatísticos decorre da acumulação de erro na previsão recursiva multi-passo.

## Principais Resultados

**H1 — Modelos sazonais superam modelos simples?**
✅ Confirmada. O Holt-Winters superou SES e Holt por margens de 13 a 20 pontos percentuais de MAPE em ambos os clusters, confirmando que a sazonalidade anual (amplitude de ~60%) é a componente dominante do portfólio.

**H2 — Cluster + Mix supera a modelagem individual por SKU?**
✅ Confirmada. MAPE médio de 3,37% (cluster + mix) contra 4,30% (naive individual por item) — ganho de +0,93pp, com 43 dos 50 itens beneficiados pela abordagem agregada.

**H3 — Otimização de hiperparâmetros supera o baseline em ≥10pp?**
⚠ Não confirmada. O ganho observado foi de apenas +0,68pp e +0,88pp sobre o Naive Sazonal. A causa raiz está conectada à alta homogeneidade sazonal do portfólio identificada no MVP 1 (correlação média de 0,996 entre SKUs): como o padrão de vendas se repete de forma muito regular ano após ano, o próprio baseline naive já é extremamente competitivo (MAPE de 3,6% e 3,3%), deixando pouca margem de melhoria percentual.

**H4 — XGBoost melhora em horizontes de previsão mais curtos?**
❌ Não confirmada — resultado inconclusivo por confundimento no desenho do teste. O MAPE piorou (não melhorou) em horizontes curtos para XGBoost e também para o Holt-Winters, revelando que o efeito é causado pela alta volatilidade do mês de janeiro (vale sazonal) isolado no horizonte de 1 mês, e não pela acumulação de erro recursivo do XGBoost.

| Modelo vencedor | Cluster 0 | Cluster 1 | Robustez (CV backtesting) |
|---|---|---|---|
| Holt-Winters (add, add, período 12) | MAPE 2,91% | MAPE 2,46% | 32,3% / 28,9% |

O **teste de Wilcoxon** mostrou que a diferença de MAPE entre Holt-Winters e XGBoost não é estatisticamente significativa em nenhum dos clusters (p = 0,30 e p = 0,11) — a escolha do modelo final é justificada por simplicidade computacional e ausência de acumulação de erro, não por superioridade estatística comprovada. O **backtesting em 3 janelas temporais** indica que o MAPE real esperado em produção deve ser comunicado como uma faixa (2,5%–4,9%), não como valor fixo.

A abordagem mantém a redução de **96% no esforço de modelagem** (50 modelos individuais → 2 famílias), agora validada com testes estatísticos de robustez.

## Técnicas Utilizadas

**Modelagem Estatística (ETS/SARIMA)**
- Naive Sazonal (baseline)
- Suavização Exponencial Simples (SES) e Dupla (Holt/DES)
- Holt-Winters (aditivo, multiplicativo, com/sem damped trend)
- SARIMA
- Grids manuais de coeficientes (α/β/γ) com análise val vs. test oracle

**Machine Learning**
- Engenharia de features temporais (lags, médias móveis, indicadores calendáriais)
- XGBoost com previsão recursiva multi-passo
- RandomizedSearchCV com TimeSeriesSplit (expanding window)

**Validação e Robustez**
- Divisão temporal treino/validação/teste sem vazamento de dados
- Backtesting com janela deslizante (rolling-origin) em 3 períodos
- Teste de significância estatística de Wilcoxon (signed-rank)
- Teste de hipótese com controle experimental (isolamento de confundimento no teste de horizonte)

**Avaliação**
- MAE, RMSE, razão RMSE/MAE (diagnóstico de erro sazonal)
- MAPE (métrica principal)
- Análise de erro percentual mês a mês
- Desagregação via mix histórico e avaliação por SKU

## Dataset

Store Item Demand Forecasting Challenge — Kaggle

- 913.000 registros diários de vendas
- 10 lojas × 50 itens × 5 anos (2013–2017)
- Escopo utilizado: Loja 1, granularidade mensal, 50 itens
- 🔗 https://www.kaggle.com/c/demand-forecasting-kernels-only

## Notebook

🔗 [Abrir no Google Colab](https://github.com/thiago-dias-paes/MVPMachineLearning-Analytics/blob/main/MVP_Machine_Learning_%26_Analytics_Thiago_Dias.ipynb)
