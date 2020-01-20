---
title: "[Parte 08] Teoria da Generalização"
categories: [Machine Learning - Learning from Data]
---

Ao longo dessa e da próxima postagem da nossa [série de textos]({{ site.baseurl }}/categorias/#machine-learning-learning-from-data), vamos estabelecer e estudar a distinção que tem que existir entre o conjunto de dados utilizados para treinar o nosso modelo, e o conjunto de dados utilizado para testá-lo.

Nesse sentido, e considerando medidas de erro, o termo $$E_{out}$$ nos diz o quão bem o nosso modelo ajustado, baseado em $$\mathcal{D}$$, generalizou o comportamento da função alvo (ou distribuição alvo) fora desse conjunto de dados $$(\mathbf{x}_1, y_1), \cdots, (\mathbf{x}_n, y_n)$$ $$-$$ ou seja, a medida $$E_{out}$$ é definida por pontos $$\mathbf{x} \not\in \mathcal{D}$$. Em contrapartida, o erro $$E_{in}$$ baseia-se nos dados de treinamento (conjunto $$\mathcal{D}$$). Além disso, como começamos a discutir na [parte 03](/memorizar-nao-e-aprender/), a relação entre $$E_{in}$$ e $$E_{out}$$ $$-$$ dado um conjunto de hipóteses $$\mathcal{H}$$ de tamanho $$M$$ $$-$$, é definida por:

$$
\begin{align*}
\mathbb{P}(\lvert E_{in}(g) - E_{out}(g) \rvert > \epsilon) \leq 2 M e^{-2 \epsilon^2 N}.
\end{align*}
$$

Veja que a inequação acima torna-se pouco útil quando consideramos um conjunto $\mathcal{H}$ de tamanho infinito (que é o que acontece na maior parte dos casos). Lembre-se também de que essa cota foi obtida pela propriedade de *union bound* para a união dos eventos $\left[ \lvert E_{in}(h_m) - E_{out}(h_m) \rvert > \epsilon \right]$, com $m = 1, \cdots, M$. Entretanto, se existir grande interseção entre esses eventos (que é o que acontece), essa cota não é boa $-$ vamos, dessa forma, tentar melhorá-la.


Para alcançar um objetivo como esse, vamos, primeiro, definir uma nova quantidade: a **função de crescimento**. Essa função é o que substituirá $M$ na nossa cota. A sequência de observações e definições a seguir nos levará onde queremos chegar.

Agora, ao invés de tomarmos $h: \mathcal{X} \longrightarrow \lbrace -1, +1 \rbrace$ (para simplificar os argumentos, iremos considerar funções avaliadas no conjunto $\lbrace -1, +1 \rbrace$, ao invés de $\mathbb{R}$), vamos restringir o domínio da função aos pontos de $\mathcal{D}$; i.e., $h: \lbrace \mathbf{x}_1, \cdots, \mathbf{x}_N \rbrace \longrightarrow \lbrace -1, +1 \rbrace$. Assim, se $h$ for avaliada em uma amostra finita $\lbrace \mathbf{x}_1, \cdots, \mathbf{x}_N \rbrace \subset \mathcal{X}$, iremos obter um $N$-upla $(h(\mathbf{x}_1), \cdots, h(\mathbf{x}_N))$ de $\pm 1 $'s.Essa "nova" função $h$ será chamada de **dicotomia**.

**Definição 1:** Tome $\lbrace \mathbf{x}_1, \cdots, \mathbf{x}_N \rbrace \subset \mathcal{X}$. As dicotomias geradas por $\mathcal{H}$ sobre esse conjunto de pontos serão definida por:

$$
\begin{align*}
\mathcal{H}(\mathbf{x}_1, \cdots, \mathbf{x}_N) = \lbrace (h(\mathbf{x}_1), \cdots, h(\mathbf{x}_N)) \mid h \in \mathcal{H} \rbrace
\end{align*}
$$

**Definição 2**: A **função de crescimento** é definida, para um conjunto $\mathcal{H}$, como:

$$
\begin{align*}
m_{\mathcal{H}}(N) = \max_{\lbrace \mathbf{x}_1, \cdots, \mathbf{x}_N \rbrace \subset \mathcal{X}} \lvert \mathcal{H}(\mathbf{x}_1, \cdots, \mathbf{x}_N) \rvert
\end{align*},
$$

onde $\lvert \cdot \rvert$ é a cardinalidade do conjunto.

Em palavras, $$m_{\mathcal{H}}(N)$$ é número máximo de dicotomias que podem ser geradas por $$\mathcal{H}$$ a partir de um conjunto de $$N$$ pontos. Aqui, note que, como os elementos de $$\mathcal{H}(\mathbf{x}_1, \cdots, \mathbf{x}_N)$$ são subconjuntos de $$\lbrace -1, +1 \rbrace^N$$, então $$m_{\mathcal{H}}(N) \leq 2^N$$. Além disso, se $$\mathcal{H}$$ for capaz de gerar todas as possíveis dicotomias sobre $$\lbrace \mathbf{x}_1, \cdots, \mathbf{x}_N \rbrace$$, então $$\mathcal{H}(\mathbf{x}_1, \cdots, \mathbf{x}_N) = \lbrace -1, +1 \rbrace^N$$ e nós dizemos que $$\mathcal{H}$$ pode *quebrar* $$\lbrace \mathbf{x}_1, \cdots, \mathbf{x}_N \rbrace$$.

Para entendermos melhor os conceitos que acabamos de definir, considere $\mathcal{H}$ como sendo o conjunto de hipóteses associado ao algoritmo Perceptron avaliado em duas dimensões. A partir da figura abaixo, podemos dizer que:

![Learning from Data]({{ site.baseurl }}/assets/images/teoria-da-generalizacao_files/perceptron-possibilidades.png)
*Figura 1 \[adaptado de: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Análise da função de crescimento para o Perceptron.*

Olhando para $${}^{(a)}$$, onde os pontos $$\mathbf{x}_1, \mathbf{x}_2$$ e $$\mathbf{x}_3$$ foram dispostos de maneira colinear, observe que não conseguimos, utilizando o Perceptron, obter todas as oito $$3$$-uplas; entretanto, considerando $${}^{(b)}$$, isso é possível, logo $$m_{\mathcal{H}}(3) = 8$$. Por outro lado, para $$N = 4$$, somente catorze das possíveis dezesseis $$4$$-uplas podem ser geradas pelo Perceptron $$-$$ assim, $$m_{\mathcal{H}}(4) = 14$$.

Veja, nesse caso, que não é fácil calcular, como função de $N$, a quantidade $m_{\mathcal{H}}(N)$. Observe ainda que, se considerarmos diferentes conjuntos $\mathcal{H}$, associados a diferentes algorimos, essa diculdade pode ser ainda maior. Felizmente, para conseguirmos o que estamos querendo, basta que sejamos capazes de determinar uma cota superior "boa" (veremos a seguir, o que quero dizer por "boa") para $m_{\mathcal{H}}(N)$. Mas antes disso, mais uma definição:

**Definição 3:** se nenhum conjunto de pontos de tamanho $k$ puder ser *quebrado* (de acordo com a definição que demos ao termo "quebrar") por $\mathcal{H}$, então $k$ é chamado de ***break point*** para $\mathcal{H}$.

A partir da Def. 3, é fácil notar que se $\mathcal{H}$ é a classe de funções associada ao algoritmo Perceptron em duas dimensões, então $k = 4$ é *break point* para $\mathcal{H}$.

A Def.3 também é fundamental para conseguirmos a cota "boa" que gostaríamos de ter para $m_{\mathcal{H}}(N)$. Veja o teorema a seguir:

**Teorema 1:** Se $m_{\mathcal{H}}(N) < 2^k$ para algum valor de $k$; isto é, se existe *break point*  $k$ para $\mathcal{H}$, então

$$
\begin{align*}
m_{\mathcal{H}}(N) \leq \sum_{i = 0}^{k - 1} {N \choose i}
\end{align*}
$$

para todo $N$. **Observação:** aqui, o lado direito da inequação é um polinômio em $N$ de grau $k - 1$.

*A demonstração do Teorema 1 pode ser encontrada no Capítulo 2 de [Learning from Data](http://www.work.caltech.edu/textbook.html).*

Observe que um cota como a estabelecida pelo Teor. 1 é **boa**, pois, se pudermos substituir $M$ por $m_{\mathcal{H}}(N)$ em $\mathbb{P}(\lvert E_{in}(g) - E_{out}(g) \rvert > \epsilon) \leq 2 M e^{-2 \epsilon^2 N}$, então: 

1. $m_{\mathcal{H}}(N)$ será dominada por alguma coisa que cresce polinomiamente rápido com $N$.
2. $e^{-2 \epsilon^2 N}$ decresce exponecialmente rápido com $N$.
3. Logo $\mathbb{P}(\lvert E_{in}(g) - E_{out}(g) \rvert > \epsilon)$  será menor ou igual que alguma quantidade arbitrariamente próxima de zero, como gostaríamos que fosse.

### Desigualdade de Vapnik-Chervonenkis 

Finalmente, o último passo que queremos conseguir justificar é o da troca de $M$ por $m_{\mathcal{H}}(N)$. Se isso for possível, resolvemos nosso problema. Esse resultado (a menos de troca de constantes) é conhecido como Desigualdade de Vapnik-Chervonenkis e é enunciado a seguir.

**Teorema 2 (Desigualdade de Vapnik-Chervonenkis):** Para qualquer $N \in \mathbb{N}$, vale: 

$$
\begin{align*}
\mathbb{P}\left[ \lvert E_{in}(g) - E_{out}(g) \rvert > \epsilon \right] \leq 4 \cdot m_{\mathcal{H}}(2N) \cdot e^{-\frac{1}{8} \epsilon^2 N} \text{, } \forall \epsilon > 0.
\end{align*}
$$

*A demonstração do Teorema 2 pode ser encontrada no Apêndice A de [Learning from Data](http://www.work.caltech.edu/textbook.html).*

No Teor. 2, perceba que as constantes que foram alteradas, em comparação à Desigualdade de Hoeffding, **não** "jogam ao nosso favor". Porém, para $N$ suficientemente grande, o lado direito da desigualdade ainda é arbitrariamente pequeno.


## Conclusão

Como já foi estabelecido desde a [parte 03](/memorizar-nao-e-aprender/), nosso interesse é argumentar que o erro $E_{out}$ (construído a partir de $\mathcal{X} - \mathcal{D}$) é bem aproximado por $E_{in}$ (construído a partir de $\mathcal{D}$). Havíamos resolvido essa questão quando o tamanho de $\mathcal{H}$ é finito; porém, do ponto de vista prático, esse quase nunca é o caso. Assim, se pudermos construir uma cota (como na Desigualdade de Hoeffding) que depende de $m_{\mathcal{H}}(N)$ ao invés de $M$, conseguimos generalizar adequamente a ideia de que é possível que o nosso modelo, a partir de um conjunto de hipóteses $\mathcal{H}$ com cardinalidade (potencialmente) infinita, aprenda. Felizmente, a desigualdade de Vapnik-Chervonenkis estabelece essa relação e resolve esse problema.

Qualquer dúvida, sugestão ou *feedback*, por favor, deixe um comentário abaixo.

> Essa postagem faz parte de uma [série]({{ site.baseurl }}/categorias/#machine-learning-learning-from-data) de textos que tem o objetivo de estudar, principalmente, o curso "[Learning from Data](http://www.work.caltech.edu/telecourse.html)", ministrado pelo Professor Yaser Abu-Mostafa. Outros materiais utilizados serão sempre referenciados.