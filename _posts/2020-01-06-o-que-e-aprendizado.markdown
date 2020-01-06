---
title: "[Parte 01] O que é Aprendizado (de Máquina) & PLA"
categories: [Machine Learning - Learning from Data]
---



Quando se fala de **Aprendizado de Máquina**, há pelo menos duas interpretações para o que isso significa: a primeira delas diz respeito a um conjunto de técnicas e modelos estatísticos que são usados, primariamente, para se fazer inferência (estimação pontual dos parâmetros de interesse, contrução de intervalo de confiança, teste de hipótese, etc.); enquanto que a segunda abordagem, que será a utilizada daqui para frente, se preocupa em, principalmente, fazer predição sobre novas observações $-$ nesse caso, o(s) modelo(s) utilizado(s) pode(m), inclusive, não ter forma explícita (Izbicki, R.; dos Santos, T. M. [Machine Learning sob a ótica estatística](http://www.work.caltech.edu/telecourse.html)). Dito isso, podemos começar.

## Componentes do aprendizado

A fim de dar contexto ao que vem a seguir, tome o seguinte exemplo: imagine que um banco contratou uma empresa de consultoria para ajudá-lo a determinar se, para cada cliente que pede um empréstimo, o banco deve ou não conceder esse crédito (aqui, um cliente é bom se, de alguma forma, faz com que o banco ganhe dinheiro; e é ruim, caso contrário). Para construir um modelo que resolve esse tipo de problema, o banco tem um conjunto de dados de clientes antigos, com várias de suas características (salário, estado civil, bens materiais, etc.), além de, se no passado, esses clientes fizeram a instituição ganhar ou perder dinheiro. De posse dessas informações, a empresa de consultoria pode escrever um modelo que ajuda o banco a predizer se clientes futuros darão (ou não) algum tipo de lucro.

Considerando esse exemplo, podemos dar nomes às componentes (do apredizado) de interesse. Seja $\mathbf{x}$ o vetor de entrada (informação que o banco tem de um cliente), então $f: \mathcal{X} \longrightarrow \mathcal{Y}$ é a função alvo (que é desconhecida) $-$ onde $\mathcal{X}$ é o espaço de entrada (para o caso onde existem $d$ características sobre um cliente, $\mathcal{X}$ é o Espaço Euclidiano $d$-dimensional) e $\mathcal{Y}$ é o espaço de saída (no exemplo, $\mathcal{Y} = \lbrace +1, -1 \rbrace$; ou seja, o banco concede ou não o empréstimo). Além disso, temos o conjunto de dados com entradas e saídas, definido por $\mathcal{D} = (\mathbf{x}_1, y_1), \cdots, (\mathbf{x}_N, y_N)$ $-$ onde $y_n = f(\mathbf{x}_n)$, tal que $n = 1, \cdots, N$ $-$, e um algoritmo de aprendizagem $\mathcal{A}$ que usa $\mathcal{D}$ para determinar uma função $g: \mathcal{X} \longrightarrow \mathcal{Y}$ que aproxima $f$. Nesse caso, o algoritmo "escolhe" uma função $g$ a partir de uma classe de funções que são relevantes para o problema (esse conjunto que contempla $g$ será denotado por $\mathcal{H}$, e receberá o nome de *Hypothesis Set*).

Voltando ao exemplo, o banco irá, baseado em $g$, decidir para quais clientes realizará o empréstimo (lembre-se que $f$ é desconhecida). Nesse caso, o algoritmo $\mathcal{A}$ selecionou $g \in \mathcal{H}$ a partir da análise do conjunto de dados $\mathcal{D}$; na esperança que o comportamento de clientes futuros ainda possa ser modelado por essa função escolhida. A figura a seguir ilustra a relação entre todas essas componentes.

![Learning from Data]({{ site.baseurl }}/assets/images/o-que-e-aprendizado/img-componentes-aprendizado.png)
*Figura 1 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Componentes do aprendizado.*

A Fig. 1 será utilizada como *framework* para tratarmos o "problema do aprendizado". Mais tarde, esse esquema vai passar por alguns refinamentos, mas a base será a mesma: 

+ Existe uma $f$ para ser aprendida (que é, e continuará sendo desconhecida para nós).

+ Temos um conjunto de dados $\mathcal{D}$ gerados por essa função alvo.

+ O algoritmo de aprendizagem $\mathcal{A}$ utiliza $\mathcal{D}$ para encontrar uma função $g \in \mathcal{H}$ que aproxima bem $f$. 

## Um modelo de aprendizado simples

Sobre as componentes do aprendizado que acabamos de discutir, a função alvo $f$ e o conjunto de dados $\mathcal{D}$ vêm do problema com o qual estamos lidando. Entretanto, o conjunto de possíveis soluções $\mathcal{H}$ e o algoritmo de aprendizagem $\mathcal{A}$ são ferramentas nós temos que escolher para determinar $g$; nesse sentido, eles ($\mathcal{H}$ e $\mathcal{A}$) são chamados de *modelo de aprendizado*.

Vejamos, então, um modelo de aprendizado simples: seja $\mathcal{X} = \mathbb{R}^d$ e $\mathcal{Y} = \lbrace +1, -1 \rbrace$; onde $+1$ e $-1$, denotam "sim" e "não", respectivamente. No exemplo do banco, diferentes coordenadas de $\mathbf{x} \in \mathcal{X}$ representam cada uma das características do cliente (salário, estado civil, bens materiais, etc.); enquanto que o espaço de saída $\mathcal{Y}$ faz referência ao fato de o banco conceder ou não o empréstimo. Em adição, vamos dizer que $\mathcal{H}$ é composto por todas as funções $h \in \mathcal{H}$ que têm forma ditada por:

$$
\begin{align}
h(\mathbf{x}) & = \text{sign}\left(\left(\sum_{i = 1}^{d} w_i x_i\right) - \text{threshold}\right) \\
              & = \text{sign}\left(\left(\sum_{i = 1}^{d} w_i x_i\right) + w_0\right), \text{ onde } w_0 = (-1) \cdot\text{threshold} \\
              & = \text{sign}\left(\sum_{i = 0}^{d} w_i x_i \right), \text{ com } x_0 = 1 \text{ fixo} \\
              & = \text{sign}\left(\mathbf{w}^{\text{T}} \mathbf{x} \right),
\end{align}
$$

onde $\mathbf{w}$ é um vetor de "pesos" e $\mathbf{x} \in \mathcal{X}$, com $\mathcal{X} = \lbrace 1 \rbrace \times \mathbb{R}^d$. O que está sendo feito aqui é simples: a família de funções $h(\cdot)$ $-$ perceba que $h$ não está completamente definida, já $\textbf{w}$ não é parâmetro da função $-$, atribui pesos $w_i$ para cada uma das $d$ características dos indivíduos. Dessa forma, podemos determinar a regra de que, se $\sum_{i = 1}^{d} w_i x_i > \text{threshold}$, então o banco aprova o empréstimo; caso contrário, não. $h(\mathbf{x})$ traduz essa ideia; além de assumir os valores $+1$ ou $-1$, como gostaríamos que fosse.

O modelo que acabamos de descrever é chamado de *perceptron*. O algoritmo de aprendizagem $\mathcal{A}$ vai procurar por valores de $\mathbf{w}$ ($w_0$ incluído) que se adaptam bem as dados. A escolha ótima será a nossa função $g$.

**Observação:** se o conjunto de dados for linearmente separável, então existirá um $\mathbf{w}$ que classifica todas as observações corretamente.

Por fim, vamos ver então qual é esse algoritmo $\mathcal{A}$. O algoritmo de aprendizagem nesse caso é chamado de *perceptron learning algorithm* (PLA), e funciona como descrito abaixo.

Para encontrar $\mathbf{w}$ tal que todos os pontos estão corretamente classificados, vamos considerar um processo de iteração em $t$ $-$ tal que $t = 0, 1, 2, \cdots$. Nesse caso, o vetor de "pesos" no $t$-ésimo instante será denotado por $\mathbf{w}(t)$; aqui, $\mathbf{w}(0)$ é escolhido arbitrariamente. Para cada etapa do processo, o algoritmo seleciona uma das obervações que **não** está classificada corretamente $-$ vamos chamá-la de $(\mathbf{x}(t), y(t))$ $-$, e aplica a seguinte regra:

$$
\begin{align}
\mathbf{w}(t+1) = \mathbf{w}(t) + y(t) \, \mathbf{x}(t).
\end{align}
$$

O que a regra que acabamos de definir faz é "mover" a reta $w_0 + w_1 x_1 + w_2 x_2 = 0$ (para o caso com $2$ dimensões) que divide os pontos do conjunto de dados; a fim de classificar corretamente a observação $(\mathbf{x}(t), y(t))$. 

Como dito anteriormente, se os dados são linearmente separáveis, o PLA converge, classificando corretamente todas as observações; o que implica em duas coisas interessantes:

1. Repare que, para cada etapa do processo de iteração, apesar de corrigir a observação que está sendo considerada, o algoritmo pode "bagunçar" a classificação (em um primeiro momento, correta) dos outros pontos; mas mesmo assim, sob a hipótese de que os dados são linearmente separáveis, o algoritmo converge (a demonstração pode ser vista [aqui](http://www.cs.columbia.edu/~mcollins/courses/6998-2012/notes/perc.converge.pdf)).

2. Para esse tipo de classificador que acabamos de detalhar, $\mathcal{H}$ é uma classe infinita de funções; assim, dizer que, nesse caso, o algorimo converge, é o mesmo que dizer que, mesmo em um conjunto de cardinalidade infinita, foi necessário uma quantidade finita de iterações para encontrarmos uma solução ótima (no sentido de classificar corretamente todos os pontos) para o problema $-$ o que é, no mínimo, interessante.

## Conclusão

Na primeira parte dessa série de textos, vimos o *framework* básico com o qual vamos trabalhar quando queremos modelar o processo de aprendizado (de máquina). Essas ideias serão revisitadas à exaustão e, por isso, são tão importantes. Por fim, vimos também como essas componentes do apredizado podem nos guiar na construção de um modelo de classificação simples. Tal modelo, o "perceptron", é de fato limitado; mas é um excelente primeiro passo que podemos dar. A próxima postagem será dedicada à implementação desse classificador em Python, bem como à discussão de um exemplo.

Qualquer dúvida, sugestão ou *feedback*, por favor, deixe um comentário abaixo.

> Essa postagem faz parte de uma [série]({{ site.baseurl }}/categorias/#machine-learning-learning-from-data) de textos que tem o objetivo de estudar, principalmente, o curso "[Learning from Data](http://www.work.caltech.edu/telecourse.html)", ministrado pelo Professor Yaser Abu-Mostafa. Outros materiais utilizados serão sempre referenciados.