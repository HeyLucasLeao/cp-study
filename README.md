# 🎉 Update 2025-01-18
Thank you to everyone who has supported me and responded to my messages, even when I felt like I was being persistently annoying. To contribute to more reliable work in the industry, I've created a library called TinyCP. This library aims to enhance my studies and allows others to incorporate it into their own projects.

In TinyCP, I've introduced some changes to the initial concept. Whether you appreciate the primary objective of my previous work, have suggestions for tests, or simply want to support me, please check out the new library. Thank you!

[TinyCP](https://github.com/HeyLucasLeao/tinycp)

# Estudo Exploratório de Previsão Conforme

## Sumário

1. Introdução
2. Calibração OOB
3. Venn-Abers
4. Previsão Conforme Mondrian por Classe Condicional
5. Margem como Métrica de Não Conformidade
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

## Previsão Conforme Mondriana por Classe Condicional

Anteriormente foi feito utilizando Cobertura Marginal. Neste método, uma proporção de 1−α das regiões de previsão é projetada para incluir o rótulo correto para novas instâncias de dados, com base em um determinado nível de confiança.

A diferença para esta metodologia é devido as regiões de previsão serem calculadas separadamente por classe. Isto garante que o desbalanceamento da distribuição de dados não interfira com as classes com menores representatividades.

![chrome_v2mwQMSiOd](https://github.com/HeyLucasLeao/cp-study/assets/26440910/0f3c6877-f7b2-4bbe-8cf7-902ac221906d)

## Score de Não Conformidade

Para fins de aprendizado, desenvolvi dois templates de modelo, utilizando por padrão o score de probabilidade inversa, ou Hinge Loss. Neste método, calculamos 1 - f(x), onde f(x) representa a previsão probabilística da camada de Venn-Abers. O valor resultante varia de zero a um:

- Zero indica que o modelo está correto
- Um indica que o modelo está completamente errado

Além disso, implementei a score de margem, que se refere à diferença entre a probabilidade da predição mais incorreta e a probabilidade correta:

- Valores ≤ 0: indicam que o modelo tem certa confiança em relação à classe verdadeira
- Valores > 0: representam uma incerteza na predição

Este template foi criado e armazenado apenas como referência de ideia e estudo pessoal. Para o desenvolvimento e leitura do conteúdo principal, não há necessidade de considerá-lo.

A fim otimizar o tempo de processamento, também refatorei o cálculo de margem para modelos binários. Como referência, utilizei o código da biblioteca [crepes](https://github.com/henrikbostrom/crepes/blob/main/src/crepes/base.py), que oferece uma implementação mais robusta.

## Diagrama de Treinamento
![Dados de treinamento](https://github.com/HeyLucasLeao/cp-study/assets/26440910/79dc819d-c37f-49ec-98fc-82e26cf82911)



## Notion de Estudos

Segue também o [link](https://polyester-citrine-4e7.notion.site/Meu-Modelo-Conforme-134a0de3378e80728ad4f279c80fb065) para minhas referências de estudos.
