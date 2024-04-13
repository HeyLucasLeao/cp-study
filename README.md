# Estudo Exploratório de Previsão Conforme

## Sumário

1. Introdução
2. Calibração OOB
3. Venn-Abers
4. Cobertura Marginal para Classificação Conformal
5. Perda de Hinge como Métrica de Não Conformidade
6. Diagrama de Modelo
7. Notion de Estudos

## Introdução

Este projeto segue apenas meu jupyter notebook relacionado a Previsão Conforme. Nele, correlaciono algumas técnicas vistas em alguns artigos cientificos aos quais utilizam a técnica que de calibração de modelo a partir da amostragem "out-of-bag" de um modelo "aprendiz", assim tirando a necessidade de criar uma amostra de dados somente para calibração.

## Calibração OOB

Este projeto faz uso do Random Forest como referência, um modelo de conjunto que emprega a técnica de bootstraping para a criação de árvores de decisão menores. Esta técnica envolve a reamostragem de um conjunto de dados com reposição, selecionando aleatoriamente subconjuntos do conjunto de dados original.

Os dados que não são utilizados no treinamento de uma árvore específica, também conhecidos como amostras OOB (Out-of-Bag), podem ser empregados para estimar a precisão desses modelos menores. No entanto, sugere-se a utilização dessas amostras não utilizadas para a calibração de uma camada superior, eliminando assim a necessidade de reservar um percentual específico de dados para a calibração.

A ideia de utilizar as amostras OOB (Out-of-Bag) para a calibração de uma camada superior é defendida em diversos estudos. Aparicio Vázquez em seu trabalho "Venn Prediction for Survival Analysis" discute a aplicação de Venn Predictors como camada superior em Random Forests para a área de Análise de Sobrevida, utilizando esta mesma técnica. Um artigo publicado na Springer, "Efficient Venn predictors using random forests", também explora a utilização de Venn Predictors para classificação com Random Forests, destacando a eficácia da calibração de mesmo método.

## Venn-Abers

O Venn-Abers é um método de calibração probabilística que pode ser aplicado em problemas de classificação binária. Ele faz parte da família de modelos conformais e é implementado como uma função de calibração sem distribuição que mapeia as pontuações produzidas por um classificador para probabilidades bem calibradas.

Ao contrário da regressão isotônica, o Venn-Abers não sofre de overfitting em conjuntos de dados de tamanho não grande, ele se destaca por é uma forma mais avançada e regularizada de regressão isotônica. Em sua essência, em vez de empregar a regressão uma vez, é aplicado duas vezes, postulando que cada objeto de teste poderia pertencer à rótulo 0 ou 1. Este objeto é então integrado ao conjunto de calibração duas vezes, sob ambas as etiquetas, levando a duas probabilidades resultantes: p0 e p1, transformando uma incerteza heurística a uma probabilidade empírica. Isso significa que o Venn-ABERS é capaz de calcular duas probabilidades distintas para cada objeto de teste, considerando a possibilidade de pertencer a ambas as classes.

## Cobertura Marginal para Classificação Conformal

A Cobertura Marginal representa uma das abordagens de garantia de cobertura na Previsão Conforme. Neste método, uma proporção de 1−α das regiões de previsão é projetada para incluir o rótulo correto para novas instâncias de dados, com base em um determinado nível de confiança. No contexto deste estudo, foi adotado um nível de confiança de 95%.

## Perda de Hinge como Métrica de Não Conformidade

Para calcular o escore de não conformidade, eu utilizo a função hinge, também conhecida como inversão probabilística, onde o cálculo é realizado como 1 - f(x), sendo f(x) a representação da previsão probabilistica da camada de Venn-Abers.

## Diagrama de Treinamento
![chrome_4nlHQR36OP](https://github.com/HeyLucasLeao/cp-study/assets/26440910/704b53ae-c2ce-4b66-9c08-afaae8bd8448)

## Notion de Estudos

Segue também o [link](https://cp-study.notion.site/Predi-o-Conformal-99bed789d7eb480f8032878a460321d0) para minhas referências de estudos.
