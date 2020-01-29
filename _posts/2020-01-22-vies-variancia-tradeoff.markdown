---
title: "[Parte 10] Trade-off entre Viés-Variância"
categories: [Machine Learning - Learning from Data]
---

Continuando com a discussão, introduzida na [parte 09](/dimensao-vc/) [dessa série de textos]({{ site.baseurl }}/categorias/#machine-learning-learning-from-data), de que deve existir um "meio termo" sobre o tamanho de $\mathcal{H}$ $-$ lembre-se: se $\mathcal{H}$ é muito grande, conseguimos diminuir o termo $E_{in}$; porém, somos penalizados na generalização do modelo para dados fora de $\mathcal{D}$. Em contrapartida, se $\mathcal{H}$ é pequeno, conseguimos, com probabilidade suficientemente alta, que $E_{out} \approx E_{in}$, mas perdemos a capacidade de fazer $E_{in}$ arbitrariamente pequeno $-$, vamos, ao longo dessa postagem, abordar esse problema de um ponto de vista diferente.

Ao invés de cotar (por cima) o termo $E_{out}$ por $E_{in} + \Omega$ (recorde que $\Omega$ pode ser interpretado como uma "penalidade" aplicada à cota de $E_{out}$ como resultado da complexidade do modelo escolhido), vamos decompor $E_{out}$ em dois termos diferentes: **viés** e **variância**. Para isso, iremos, como no modelo de regressão, trabalhar com uma medida de erro específica, o **erro quadrático**. Da [parte 05](/modelo-de-regressao-linear/), temos que $$E_{out}(h) = \mathbb{E}\left[(h(\mathbf{x}) - f(\mathbf{x}_n))^2\right]$$ e $$E_{in}(h) = \frac{1}{N} \sum_{n = 1}^{N}(h(\mathbf{x}_n) - f(\mathbf{x}_n))^2$$, para $$h \in \mathcal{H}$$.

Assim, podemos escrever que:

$$
\begin{align*}
E_{out}(g^{(\mathcal{D})}) = \mathbb{E}_{\mathbf{x}}\left[(g^{(\mathcal{D})}(\mathbf{x})-f(\mathbf{x}))^2\right],
\end{align*}
$$

onde $g \in \mathcal{H}$ é a hipótese escolhida por $\mathcal{A}$ e depende, obviamente, do conjunto de dados escolhido $\mathcal{D}$. Além disso, $\mathbb{E}_{\mathbf{x}}\left[\cdot\right]$ é o valor esperado com respeito a $\mathbf{x}$. Agora, para tirar a dependência que $g$ tem de $\mathcal{D}$, podemos tomar a esperança (dessa vez, com respeito a $\mathcal{D}$) dos dois lados da equação $-$ assim, temos que:

$$
\begin{align*}
\mathbb{E}_{\mathcal{D}}\left[E_{out}(g^{(\mathcal{D})})\right] & = \mathbb{E}_{\mathcal{D}}\left[\mathbb{E}_{\mathbf{x}}\left[(g^{(\mathcal{D})}(\mathbf{x})-f(\mathbf{x}))^2\right]\right] \\
& = \mathbb{E}_{\mathbf{x}}\left[\mathbb{E}_{\mathcal{D}}\left[(g^{(\mathcal{D})}(\mathbf{x})-f(\mathbf{x}))^2\right]\right] \text{, já que } (\cdot)^2 > 0 \\
& = \mathbb{E}_{\mathbf{x}}\left[ \mathbb{E}_{\mathcal{D}}\left[g^{(\mathcal{D})}(\mathbf{x})^2\right] - 2 \bar{g}(\mathbf{x}) f(\mathbf{x}) + f(\mathbf{x})^2 \right] \text{, onde } \bar{g}(\mathbf{x}) = \mathbb{E}_{\mathcal{D}}\left[g^{(\mathcal{D})}(\mathbf{x})\right] \\
& = \mathbb{E}_{\mathbf{x}}\left[ \mathbb{E}_{\mathcal{D}}\left[g^{(\mathcal{D})}(\mathbf{x})^2\right] - \bar{g}(\mathbf{x})^2 + \bar{g}(\mathbf{x})^2 - 2 \bar{g}(\mathbf{x}) f(\mathbf{x}) + f(\mathbf{x})^2 \right] \\
& = \mathbb{E}_{\mathbf{x}}\left[ \mathbb{E}_{\mathcal{D}}\left[(g^{(\mathcal{D})}(\mathbf{x}) - \bar{g}(\mathbf{x}))^2\right] + (\bar{g}(\mathbf{x}) - f(\mathbf{x}))^2 \right].
\end{align*}
$$

Nesse caso, vamos dizer que $\text{Viés}(\mathbf{x}) = (\bar{g}(\mathbf{x}) - f(\mathbf{x}))^2$ e que $\text{Var}(\mathbf{x}) = \mathbb{E}_{\mathcal{D}}\left[( g^{(\mathcal{D})}(\mathbf{x}) - \bar{g}(\mathbf{x}))^2\right]$. **Observação 1:** Aqui, a notação e a forma de definir as quantidades de interesse talvez lhe pareça estranho $-$ para mim, esse também foi o caso $-$, mas as utilizarei com a intenção de seguir o livro [Learning from Data](http://www.work.caltech.edu/textbook.html). Dessa forma, temos que:

$$
\begin{align*}
\mathbb{E}_{\mathcal{D}}\left[E_{out}(g^{(\mathcal{D})})\right] & = \mathbb{E}_{\mathcal{x}}\left[\text{Viés}(\mathbf{x}) + \text{Var}(\mathbf{x}) \right] \\
& = \text{Viés} + \text{Var},
\end{align*}
$$

onde $$\text{Viés} = \mathbb{E}_{\mathbf{x}}\left[\text{Viés}(\mathbf{x})\right]$$ e $$\text{Var} = \mathbb{E}_{\mathbf{x}}\left[\text{Var}(\mathbf{x})\right]$$. Nesse caso, perceba que a decomposição que fizemos assume que **não** existe ruído $-$ caso fôssemos incluí-lo; i.e., se optássemos por escrever $$y = f(\mathbf{x}) + \epsilon$$, teríamos um termo a mais na última soma.

Vamos ver agora que o problema associado ao tamanho (ou complexidade) de $\mathcal{H}$ é capturado pela decomposição de viés-variância de $E_{out}$ que fizemos. Para isso, considere dois cenários extremos:

1. $\mathcal{H} = \lbrace h \rbrace$, tal que $h \neq f$. Nesse caso, como o conjunto de hipóteses tem cardinalidade $1$, teremos $g^{(\mathcal{D})}(\mathbf{x}) = \bar{g}(\mathbf{x})$; logo, $\text{Var} = 0$. Porém, o termo $\text{Viés}$ dependerá apenas do quão bem $h$ aproxima $f$ $-$ assim, via de regra, teremos viés alto.

2. $\mathcal{H} = \lbrace h_1, \cdots h_k \rbrace$, para $k$ "muito grande". Aqui, se $f \in \mathcal{H}$, teremos viés muito próximo de zero. Entretanto, apesar de $\bar{g}(\mathcal{x}) \approx f$, a escolha da hipótese $g$ por $\mathcal{A}$ irá variar de acordo com $\mathcal{D}$, fazendo com que o termo $\text{Var}$ assuma valores (potencialmente) "altos".

**Exemplo:** Suponha que $f(x) = \sin(\pi x)$, e que $\mathcal{D}$ é tal que $N = 2$; com pontos $(x_1, y_1)$ e $(x_2, y_2)$ tal que $x$ é amostrado uniformemente de $[-1, 1]$. Para esse exemplo, considere dois conjuntos de hipóteses:

$$
\begin{align*}
\mathcal{H}_0 &: h(x) = b \\
\mathcal{H}_1 &: h(x) = ax + b.
\end{align*}
$$

Para $\mathcal{H}_0$, $b$ é escolhido de tal forma que $b = \frac{y_1 + y_2}{2}$. Similarmente, para $\mathcal{H}_1$, nós escolhemos $a$ e $b$ de tal forma que a reta definida por esses valores passa por $(x_1, y_1)$ e $(x_2, y_2)$. Assim, através de simulação, podemos estimar o viés e variância para os modelos que acabamos de definir. Mas antes, veja a figura abaixo $-$ que mostra como as curvas $h(x)$ foram ajustadas para $\mathcal{H}_0$ e $\mathcal{H}_1$ nas várias iterações do processo de simulação que sorteia os pontos de $\mathcal{D}$:

![Simulação-dois-modelos]({{ site.baseurl }}/assets/images/vies-variancia-tradeoff_files/figura-1.png)
*Figura 1 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Curva $f(x) = \sin(\pi x)$ para os modelos $\mathcal{H}_0$ e $\mathcal{H}_1$.*

Perceba, a partir da Fig. 1, que as retas ajustadas para $\mathcal{H}_1$ variam muito mais se comparadas às retas de $\mathcal{H}_0$. A imagem a seguir formaliza essa diferença e sumariza as informações de viés e variância para os dois conjuntos de hipóteses: 

![Sumário-de-viés-variância]({{ site.baseurl }}/assets/images/vies-variancia-tradeoff_files/figura-2.png)
*Figura 2 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Análise de viés e variância para $\mathcal{H}_0$ e $\mathcal{H}_1$.*

Como é possível perceber a partir da Fig. 2, apesar de o modelo mais simples $$\mathcal{H}_0$$ ter maior viés (se comparado a $$\mathcal{H}_1$$), ele tem valor esperado para $$E_{out}$$ menor (soma de viés e variância) e, portanto, é o conjunto de hipóteses preferível nesse cenário. **Observação 2:** como o termo $$\text{Var}$$ diminui com o aumento de $$N$$, caso tivéssemos um amostra maior, o modelo $$\mathcal{H}_1$$ poderia começar a se tornar a melhor opção.

Aqui, o objetivo prático é diminuir a variância sem aumentar consideravelmente o viés; ou, de forma análoga, diminuir o viés sem aumentar muito a variância. Para alcançar resultados nesse sentido, diversas técnicas podem ser empregadas $-$ *regularization* é uma delas.

## Conclusão

A partir da ideia de que a complexidade de um modelo, associada ao tamanho do conjunto de hipóteses $\mathcal{H}$, é um fator complicado de se tratar $-$ $\mathcal{H}$ grande facilita a minimização de $E_{in}$, mas compromete a generalização de $f$ por $g$; ao passo que $\mathcal{H}$ pequeno compromete o erro amostral, mas permite $E_{out} \approx E_{in}$ $-$, conseguimos, através das quantidades **viés** e **variância**, introduzir um novo tipo de análise. Modelos mais complexos são mais permissíveis em relação ao ajuste da curva aos dados, mas sofrem mais variação com as possíveis amostras para $\mathcal{D}$; por outro lado, modelos mais simples podem ter dificuldade de captar o comportamento de $f$, mas são menos suscetíveis às variações do conjunto de dados. Esse tipo de análise será feita mais vezes à medida que formos implementando outros exemplos. Por fim, o próximo post voltará a falar de modelos lineares.


Qualquer dúvida, sugestão ou *feedback*, por favor, deixe um comentário abaixo.

> Essa postagem faz parte de uma [série]({{ site.baseurl }}/categorias/#machine-learning-learning-from-data) de textos que tem o objetivo de estudar, principalmente, o curso "[Learning from Data](http://www.work.caltech.edu/telecourse.html)", ministrado pelo Professor Yaser Abu-Mostafa. Outros materiais utilizados serão sempre referenciados.
