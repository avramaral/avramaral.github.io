---
title: "[Parte 03] Memorizar não é Aprender"
categories: [Machine Learning - Learning from Data]
---


O que nós fizemos até agora foi, dado um conjunto $\mathcal{D}$, treinar um modelo que "explicasse" (no caso do Perceptron, "classificasse") os dados $(\mathbf{x}_1, y_1), \cdots, (\mathbf{x}_N, y_N)$. Porém, isso significa, de fato, aprender? Isto é, a função $g \in \mathcal{H}$ escolhida pelo algoritmo aproxima a função alvo $f$ no sentido de ter bom desempenho em explicar $\mathbf{x} \not\in \mathcal{D}$?

O que nós vamos fazer agora é introduzir uma componente aleatória no *framework* de aprendizado que começamos a discutir na [parte 01](/o-que-e-aprendizado/) [dessa série de textos]({{ site.baseurl }}/categorias/#machine-learning-learning-from-data) $-$ que vai nos permitir responder "sim" às perguntas do parágrafo anterior.

Para $$h \in \mathcal{H}$$, seja $$\mu = \mathbb{P}\left[h(\mathbf{x}_n) \neq f(\mathbf{x}_n)\right]$$, com $$n = 1, \cdots, N$$; i.e., a probabilidade de que o valor de uma função fixa $$h$$  avaliada em $$\mathbf{x}_n$$ $$-$$ com $$h$$ escolhida antes de $$\mathcal{D}$$ ser gerado $$-$$ seja diferente da função alvo $f$ avaliada no mesmo ponto. Por consequência, $$1 - \mu = \mathbb{P}\left[h(\mathbf{x}_n) = f(\mathbf{x}_n)\right]$$. Além disso, defina $$\nu = \frac{1}{N} \sum_{n= 1}^{N} \mathbb{I}_{\lbrace h(\mathbf{x}_n) \neq f(\mathbf{x}_n)\rbrace}$$; ou seja, $$\nu$$ é a proporção de vezes que $$h(\mathbf{x}_n)$$ é diferente de $$f(\mathbf{x}_n)$$ para uma amostra $$\mathcal{D}$$. A ideia é que, desde que a $$\mathcal{D}$$ seja gerada aleatoriamente seguindo uma distribuição $$P$$ (não necessariamente conhecida), então $$\nu$$ aproxima bem $$\mu$$. A relação a seguir, conhecida como *Desigualdade de Hoeffding*, quantifica essa aproximação:

$$
\begin{align*}
    \mathbb{P}\left[\lvert \nu - \mu\rvert > \epsilon\right] \leq 2e^{-2 \epsilon^2 N} \text{, para todo N e }\forall \epsilon > 0.
\end{align*}
$$

O que a inequação acima diz é que a probabilidade de $\nu$ estar arbitrariamente próximo de $\mu$ é algo como "$1$ menos alguma coisa que decai exponencialmente com $N$". O que é o mesmo que dizer que, para $N$ "grande", $\nu$ aproxima bem o comportamento de $\mu$. Assim, a quantidade de vezes que $h$ erra na amostra $\mathcal{D}$ é proporcional à quantidade de erros que $h$ cometeria fora de $\mathcal{D}$. Veja abaixo um esquema atualizado das nossas componentes do aprendizado.

![Framework de aprendizado com componente estocástica]({{ site.baseurl }}/assets/images/memorizar-nao-e-aprender_files/comp-aprendiz-estocastico.png)
*Figura 1 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Framework de aprendizado atualizado com componente estocástica.*

Dito tudo isso, se houvesse apenas uma função em $\mathcal{H}$, seria fácil de verificar se $h$ tem bom desempenho em avaliar pontos fora de $\mathcal{D}$; só que esse não é o caso $-$ na maior parte das vezes, $\mathcal{H}$ tem cardinalidade infinita, inclusive. Isso nos motiva a introduzir uma nova notação:

Defina $$E_{in}(h) = \frac{1}{N} \sum_{n= 1}^{N}\mathbb{I}_{\lbrace h(\mathbf{x}_n) \neq f(\mathbf{x}_n)\rbrace}$$ e $$E_{out}(h) = \mathbb{P}\left[h(\mathbf{x}_n) \neq f(\mathbf{x}_n)\right]$$; ou seja, $$E_{in}(h)$$ e $$E_{out}(h)$$ são, respectivamente, as quantidades $$\nu$$ e $$\mu$$ **como função de $$h$$**. Então é óbvio que, para todo $$N$$, $$\mathbb{P}\left[\lvert E_{in}(h) - E_{out}(h)\rvert > \epsilon\right] \leq 2e^{-2 \epsilon^2 N}$$, $$\forall \epsilon > 0$$. O ganho em definir essa nova notação aparece no próximo parágrafo. **Observação:** o subscrito "$$in$$" faz referência ao termo *in-sample*; da mesma forma, "$$out$$" quer dizer *out-of-sample*.

A cota que temos até agora diz respeito a uma única função $h \in \mathcal{H}$ ; porém, se $\mathcal{H}$ tem cardinalidade maior que $1$ (o que, na prática, é sempre verdade), podemos escrever uma relação parecida para uma função $g \in \mathcal{H}$ escolhida por $\mathcal{A}$. Seja $\mathcal{H}$ conjunto finito de tamanho $M$, então vale:

$$
\begin{align*}
\mathbb{P}\left[\lvert E_{in}(g) - E_{out}(g)\rvert > \epsilon\right] & \leq \mathbb{P}\left[\bigcup_{m = 1}^{M} \left[ \lvert E_{in}(h_m) - E_{out}(h_m)\rvert > \epsilon \right] \right] \\
& \leq \sum_{m = 1}^{M} \mathbb{P}\left[\lvert E_{in}(h_m) - E_{out}(h_m)\rvert > \epsilon \right] \\
& \leq \sum_{m = 1}^{M} 2e^{-2 \epsilon^2 N} = 2 M e^{-2 \epsilon^2 N}.
\end{align*}
$$

Nas equivalências acima, a primeira desigualdade é justificada por inclusão de eventos, a segunda por *union bound*, e a terceira, como já vimos, vem da Desigualdade de Hoeffding.

A princípio, essa cota que conseguimos para $g$ só faz sentido se $M$ for finito; já que o lado direito da desigualdade cresce com $M$. Entretanto, esse resultado pode ser generalizado para $\mathcal{H}$ conjunto infinito.

Em resumo, é possível interpretar o resultado de que $\mathbb{P}\left[\lvert E_{in}(g) - E_{out}(g)\rvert > \epsilon\right] \leq 2 M e^{-2 \epsilon^2 N}$ da seguinte forma: a depender de $M$ e $\epsilon$, o nosso modelo consegue, de fato, **aprender**, pois a função $g \in \mathcal{H}$ escolhida por $\mathcal{A}$ se aproxima de $f$ quando $N$ cresce $-$ no sentido de, se $E_{in}(g) \approx 0$, ter probabilidade de erro arbitrariamente pequena para avaliar $\mathbf{x} \not\in \mathcal{D}$.

## Conclusão

Vimos que, com a introdução de uma componente estocástica no nosso *framework* de aprendizado, é possível que nosso modelo aprenda (e não apenas memorize). Nesse caso, quando $N$ cresce, $E_{in}(g) \approx E_{out}(g)$. Assim, se conseguirmos fazer com que $E_{in}(g) \approx 0$, então teremos que $E_{out}(g) \approx 0$; o que é o mesmo que dizer que se o algoritmo $\mathcal{A}$ conseguir escolher uma função $g \in \mathcal{H}$ que tem bom desempenho em avaliar $\mathbf{x} \in \mathcal{D}$, então $g$ também terá bons resultados para observações fora de $\mathcal{D}$.

Qualquer dúvida, sugestão ou *feedback*, por favor, deixe um comentário abaixo.

> Essa postagem faz parte de uma [série]({{ site.baseurl }}/categorias/#machine-learning-learning-from-data) de textos que tem o objetivo de estudar, principalmente, o curso "[Learning from Data](http://www.work.caltech.edu/telecourse.html)", ministrado pelo Professor Yaser Abu-Mostafa. Outros materiais utilizados serão sempre referenciados.