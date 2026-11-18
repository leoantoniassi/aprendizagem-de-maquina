# Explicação da resolução da Avaliação P2

Este documento explica o que foi feito no notebook [`exp2.ipynb`](exp2.ipynb) e por que cada escolha foi tomada. O notebook foi executado por completo: **0 células com erro** e 10 gráficos gerados.

## 1. Material de referência usado

| Tema | Material consultado | O que foi aproveitado |
| --- | --- | --- |
| NumPy e matplotlib | `Aula 2/MAtplotlib e Numpy.ipynb` | leitura de dados e gráficos |
| Matriz de confusão e métricas | `Aula 6/regressao_logistica_matriz_confusao.ipynb` | cálculo manual de VP/VN/FP/FN |
| Perceptron | `Aula 9/perceptron.ipynb` | classe `Perceptron`, regra `w += eta*y*x`, `np.sign`, gráfico de erro/fronteira |
| MLP | `Aula 10/MLP.ipynb` | camadas ocultas, ReLU, entropia cruzada, Otimização, mini-batches, saída multiclasse |
| K-means | `Aula 11/kmean.ipynb` | distância euclidiana, associação/atualização de centróides e critério de parada |

### Bibliotecas

Usei apenas **NumPy + matplotlib**. O enunciado (instrução VI) fala em "bibliotecas permitidas" sem enumerá-las; no material da turma, `scikit-learn`, `pandas`, `Keras` e `TensorFlow` **nunca aparecem**, e o padrão é implementar os algoritmos do zero. O `torch` aparece só na Aula 10 e não está instalado no ambiente. Implementar o MLP em NumPy resolve a ambiguidade: o notebook roda em qualquer ambiente com as duas bibliotecas básicas da disciplina.

## 2. Questão 1 — Marketing: K-means

**Problema:** 300 clientes, duas variáveis numéricas, **sem rótulo**. O objetivo é descobrir perfis. Isso é aprendizado não supervisionado → **K-means**.

**Por que k = 3:** calculei a inércia para k = 2..7 com 10 reinicializações (sementes fixas). A maior queda acontece entre k=2 (6555) e k=3 (1275); a partir de k=4 os ganhos são pequenos (1116, 961, 813...). O "joelho" da curva está em 3, o que coincide com a sugestão do enunciado de perfis como "ganha muito/gasta muito" e "ganha pouco/gasta pouco". Com k=4, um dos grupos reais é dividido em dois.

**Resultado:** três clusters equilibrados (100 clientes cada), centróides aproximados:

- `(-6,89 / -7,04)` → renda baixa e gasto baixo: perfil conservador;
- `(4,80 / 2,03)` → renda alta e gasto acima da média: perfil premium;
- `(-2,70 / 9,06)` → renda abaixo da média, mas gasto muito alto: perfil de consumo elevado/aspiração.

Os números são padronizados, então "negativo" significa abaixo da média do conjunto. A campanha pode ser direcionada por perfil: baixo custo/poupança, produtos exclusivos e parcelamento/fidelidade.

## 3. Questão 2 — Crédito: Perceptron

**Problema:** separação "clara" por uma reta e pedido explícito do **modelo mais simples possível com eficiência computacional** → **Perceptron** (uma camada, atualização por amostra), não MLP.

**Adaptações em relação à Aula 9:**

1. Rótulos `0/1` convertidos para `-1/+1`, pois o Perceptron usa `np.sign`.
2. **Padronização z-score** com média/desvio calculados **só no treino** (80% treino, 20% teste, `seed=42`). As duas features têm desvios muito diferentes (≈1,43 e ≈0,38), o que desestabiliza a atualização.
3. **Pocket**: os dados não são perfeitamente separáveis (as duas classes formam faixas com correlações opostas e se sobrepõem). O Perceptron oscila; guardar os pesos de **maior acurácia** encontrados durante as épocas evita que o último passo destruza um bom modelo. O erro por época continua sendo plotado, como na Aula 9.

**Resultado:** 86,25% no treino e **86,67% no teste**; matriz de confusão: VN=36, VP=16, FN=8, FP=0. Uma busca exaustiva por retas mostra que o teto linear é ≈87,7%, então o modelo está praticamente no limite do que uma linha consegue fazer. Os 16 acertos entre bons pagadores e os 8 erros mostram o custo da sobreposição: o Perceptron é a escolha certa pelo critério "simplicidade", ainda que um MLP pudesse ganhar alguns pontos.

**Fronteira:** plotada com `meshgrid`/`contourf` sobre os dados originais (a grade é padronizada com a mesma média/desvio do treino antes de `predict`).

## 4. Questão 3 — Fraude: MLP

**Problema:** fraudes têm raio médio **0,502 ± 0,056** e transações normais **0,999 ± 0,048**: são círculos concêntricos. Nenhuma reta separa o interior do anel do exterior → **Perceptron está descartado**. Escolheu-se o **MLP** (Aula 10), que produz fronteiras curvas.

**Implementação do zero em NumPy:**

- arquitetura `2 → 16 → 16 → 2` com ReLU nas ocultas e inicialização He (`sqrt(2/fan_in)`);
- saída com 2 neurônios + **softmax** e **entropia cruzada** (equivalente ao `CrossEntropyLoss` da Aula 10; no caso binário da aula foi usado BCE, mas 2 saídas softmax generalizam melhor para a Questão 4);
- **Adam** implementado manualmente (momentos com correção de viés), mini-batches de 32, 200 épocas, `seed=42`, split 80/20;
- `plot_decision_boundary` com `meshgrid` + `contourf`, mesmo padrão das aulas.

**Resultado:** perda cai de ~0,60 para ~0,00 e a acurácia chega a **100% no teste**. A fronteira desenhada é um círculo, confirmando que as camadas ocultas aprenderam a estrutura radial sem feature engineering manual.(validação e teste usaram o mesmo conjunto de 60 amostras, separado do treino.)

## 5. Questão 4 — Investimentos: MLP multiclasse

**Problema:** 4 entradas e saída em **3 classes** (0=baixo, 1=médio, 2=alto). Exigência explícita de "múltiplas saídas" → o **mesmo MLP**, com `3 → 3` neurônios na saída + softmax: a classe prevista é o `argmax` das probabilidades.

**Dados:** 400 ativos, classes quase balanceadas (134/133/133) e médias muito distintas em cada classe (desvio ≈1), o que torna a tarefa linearmente separável em 4D; ainda assim o MLP é o modelo pedido e lida bem com o caso.

**Resultado:** **100% de acurácia no teste**, matriz de confusão perfeitamente diagonal (17, 28, 35 nas classes 0, 1, 2). Também foi plotada a projeção das regiões de decisão nas duas primeiras variáveis, mantendo Liquidez e Tempo de Maturação na média — apenas para visualização, já que o modelo usa as 4 features.

## 6. Resultados consolidados

| Questão | Algoritmo | Métrica principal |
| --- | --- | --- |
| 1 — Marketing | K-means, k=3 | 3 grupos de 100 clientes; joelho em k=3 |
| 2 — Crédito | Perceptron (z-score + pocket) | 86,67% acurácia no teste |
| 3 — Fraude | MLP 2→16→16→2 | 100% acurácia no teste |
| 4 — Investimentos | MLP 4→16→16→3 | 100% acurácia no teste |

## 7. Decisões transversais

- **Sementes fixas (42)** em splits, inicializações e K-means: resultados reproduzíveis.
- **Split 80/20 aleatório com semente fixa** (`np.random.default_rng(42).permutation`): resultado reproduzível e classes próximas do balanceamento original.
- **Padronização apenas quando necessária**: no Perceptron (escalas diferentes); no K-means os dados já vêm padronizados; no MLP as feature scales são adequadas e a normalização não foi necessária.
- **Nenhuma biblioteca "pronta" de ML**: todo o aprendizado (perda, backpropagation, Adam, métricas) está explícito no código, facilitando a avaliação dos critérios da prova.
- **Explicações dentro do notebook**: cada questão tem células markdown com conclusão baseada nos números impressos e nos gráficos, atendendo à instrução VI.

Para reproduzir: abra `ex p2/exp2.ipynb` com o diretório de trabalho em `ex p2` (onde estão os CSVs) e execute todas as células.
