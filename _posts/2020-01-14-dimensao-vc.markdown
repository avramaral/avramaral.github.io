---
title: "[Parte 09] Dimensão de Vapnik-Chervonenkis"
categories: [Aprendizado de Máquina - Learning from Data]
tags: [aprendizado-de-maquina]
---

Continuando a discussão que começamos na [última parte](/teoria-da-generalizacao/) da [nossa série de textos]({{ site.baseurl }}/categories/aprendizado-de-máquina-learning-from-data/), vamos estudar mais propriedades associadas ao que demos o nome de "Teoria da Generalização"; ou, em outras palavras, ao estudo de como os nossos modelos podem ser generalizados para conjuntos de dados fora de $\mathcal{D}$.

Nesse sentido, a primeira definição que vamos introduzir $-$ e a mais importante do ponto de vista **prático**, é a de Dimensão Vapnik-Chervonenkis (VC).

**Definição 1:** a **dimensão VC** de um conjunto $\mathcal{H}$, denotada por $d_{\text{VC}}(\mathcal{H})$ (ou somente $d_{\text{VC}}$), é o maior valor de $N$ para o qual $m_{\mathcal{H}}(N) = 2^N$.

De forma simples, a dimensão VC corresponde ao máximo de pontos $N$ (para **algum** conjunto de pontos) que $\mathcal{H}$ pode *quebrar* (lembre-se do significado que demos à essa expressão). Além disso, como é fácil perceber, se $d_{\text{VC}}$ é dimensão VC para $\mathcal{H}$, então $k = d_{\text{VC}} + 1$ é *break point*. Assim, o **Teor. 1 da [parte 08](/teoria-da-generalizacao/)** pode ser reescrito da seguinte forma:

$$
\begin{align*}
m_{\mathcal{H}(N)} \leq \sum_{i = 0}^{d_{\text{VC}}}{N \choose i},
\end{align*}
$$

onde o termo do lado direito da inequação é um polinômio de grau $d_{\text{VC}}$; i.e., a dimensão VC também determina o grau do polinômio que limita a função de crescimento.

Uma outra consequência direta dessa nova definição é que conseguimos, sem surpresa, caracterizar o fenômeno $f \approx g$; ou seja, se $d_{\text{VC}}$ é finito, então $g \in \mathcal{H}$ irá generalizar o comportamento de $f$. Em adição, perceba que a dimensão VC é independente de: 

- $\mathcal{A}$, o algoritmo de aprendizagem $-$ note que, sob a hipótese de que $d_{\text{VC}}$ é finito, $f$ é generalizada (bem ou mal) por qualquer $h \in \mathcal{H}$.
- $P$, a distribuição de entrada $-$ nesse caso, perceba que, para a função de crescimento, escolhemos um conjunto de pontos que maximiza o números de dicotomias geradas por $\mathcal{H}$; assim, a escolha da distribuição que determina os pontos de $\mathcal{D}$ não exerce papel fundamental na ideia de "generalização".
- $f$, a função alvo $-$ veja que, satisfeita determinadas condições, a Desigualdade de Vapnik-Chervonenkis é válida independente de que função $f$ estamos tentando generalizar.

Agora, utilizando o conceito que acabamos de definir, podemos tentar determinar a dimensão VC para algum conjunto conhecido de hipóteses $\mathcal{H}$. Por exemplo, para o Perceptron $d$-dimensional, temos que $d_{\text{VC}} = d + 1$. A demonstração desse fato não é complicada, e o argumento principal recai sobre o fato de que é possível mostrar que: ${}^{(a)}$ $d_{\text{VC}} \leq d + 1$ e ${}^{(b)}$ $d_{\text{VC}} \geq d + 1$. Entretanto, mais importante do que o resultado em si, é sua interpretação: qual o significado da quantidade $d + 1$ em um modelo Perceptron de $d$ dimensões? 

Para o algoritmo Perceptron $d$-dimensional, a quantidade $d + 1$ diz respeito à quantidade de parâmetros necessários para se determinar a hipótese $h \in \mathcal{H}$. Nesse sentido, podemos dizer que, de alguma forma, a dimensão VC é tão maior (ou menor) quanto for o número de **parâmetros efetivos** (aqui, talvez o termo "**graus de liberdade**" seja mais adequado) do modelo com o qual estamos trabalhando.

Além disso, uma outra interpretação da quantidade "dimensão VC" pode ser introduzida $-$ nesse caso, uma interpretação **prática**. Mas antes, vamos relembrar o que a Desigualdade Vapnik-Chervonenkis diz:

$$
\begin{align*}
\mathbb{P}\left[ \lvert E_{in}(g) - E_{out}(g) \rvert > \epsilon \right] \leq \delta,
\end{align*}
$$

onde $\delta = 4 \cdot m_{\mathcal{H}}(2N) \cdot e^{-\frac{1}{8} \epsilon^2 N}$.

Da equação anterior, podemos nos perguntar: fixados $\delta$ e $\epsilon$, como se relacionam $N$ e $d_{\text{VC}}$? Para tentar responder à essa pergunta, ao invés de olharmos para $\delta$ como definido acima, vamos nos concentrar em analisar $\delta = N^{d_{\text{VC}}} \cdot e^{-N}$ (que é uma simplificação "honesta" do problema que estamos tentando estudar). O gráfico de $\delta(N)$ para diferentes valores de $d_{\text{VC}}$ é apresentado a seguir.


```python
# plot of (N^d * exp{-N}) for different values of "d"

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

def my_func(N, d):
    return ((N ** d) * np.exp(-1 * N))

d_values = [5, 10, 15, 20, 25, 30]

x = np.linspace(0, 200, num = 100)
for d in d_values:
    y = my_func(x, d)
    plt.plot(x, y, label = "d = " + str(d))

plt.axhline(1, color = "black")
plt.xlabel("N")
plt.ylabel("Upper bound for the desired probab. (log scale)")
plt.xlim(0, 200)
plt.ylim(10e-3, 10e9)
plt.yscale("log")
plt.legend()
plt.show()
```


![png]({{ site.baseurl }}/assets/images/dimensao-vc_files/dimensao-vc_9_0.png)


Perceba que a cota superior para a probabilidade deseja também cresce à medida que $d_{\text{VC}}$ cresce. **Observação 1:** apesar de estarmos olhando para uma cota superior de $\mathbb{P}\left[ \lvert E_{in}(g) - E_{out}(g) \rvert > \epsilon \right]$, o comportamento apresentado no gráfico traduz, empiricamente, o comportamento da probabilidade desejada. Assim, se estamos trabalhando com modelos de alta dimensão VC, precisaremos de uma amostra de tamanho proporcionalmente maior $-$ como regra **prática**, utilizaremos $N = 10 \cdot d_{\text{VC}}$.

### Cota para Teoria da Generalização

Por fim, antes de terminar esse texto, vamos trabalhar um pouco com a Desigualdade de Vapnik-Chervonenkis, reescrevendo-a em um formato que nos será mais útil. Dessa forma, se $\delta = 4 \cdot m_{\mathcal{H}}(2N) \cdot e^{-\frac{1}{8} \epsilon^2 N}$, então:

$$
\begin{align*}
\epsilon = \sqrt{\frac{8}{N} \, \ln \left[{\frac{4 \cdot m_{\mathcal{H}}(2N)}{\delta}}\right]} = \Omega(N,\mathcal{H},\delta).
\end{align*}
$$

Assim, com probabilidade $\geq 1 - \delta$,

$$
\begin{align*}
\lvert E_{in}(g) - E_{out}(g) \rvert \leq \Omega(N,\mathcal{H},\delta) = \Omega.
\end{align*}
$$

Porém, como o termo $E_{in}$ é (quase sempre) forçadamente menor que $E_{out}$ (lembre-se que, quando ajustamos um modelo, estamos tentando minimiar $E_{in}$), podemos simplificar um pouco mais a nossa notação e escrever que, com probabilidade $\geq 1 - \delta$, 

$$
\begin{align*}
E_{out} - E_{in} \leq \Omega.
\end{align*}
$$

A inequeção que acabamos de escrever é conhecida como uma cota para o "erro generalizado".

Finalmente, com probabilidade $\geq 1 - \delta$, 

$$
\begin{align*}
E_{out} \leq E_{in} + \Omega.
\end{align*}
$$

Essa forma de reescrever a Desigualdade de Vapnik-Chervonenkis é interessante porque, apesar do termo da esquerda permanecer desconhecido, nós temos algum controle sobre $E_{in}$ e $\Omega$ (o primeiro é o que tentamos minimizar, enquanto que o segundo relaciona-se à escolha de $\mathcal{H}$). **Observação 2:** aqui, note que um conjunto de hipóteses $\mathcal{H}$ "grande" ajuda a minimar $E_{in}$, mas faz com que $\Omega$ cresça (já que, se $\mathcal{H}$ cresce, então $d_{\text{VC}}$ também é potencialmente maior), o que é ruim. Dessa forma, deve existir um balanço sobre o tamanho de $\mathcal{H}$ para minimizar o termo $E_{out}$.

## Conclusão

Dando continuidade à [parte 08](/teoria-da-generalizacao/), definimos e estudamos uma nova quantidade: **dimensão VC** ou $d_{\text{VC}}(\mathcal{H})$. Vimos, nesse caso, como o valor de $d_{\text{VC}}$ afeta, do ponto de vista prático, o tamanho de amostra $N$ que temos que ter para minimar a diferença entre $E_{out}$ e $E_{in}$. Por último, rearranjando os termos da Desigualdade de Vapnik-Chervonenkis, fomos capazes de escrever uma cota, como função de $E_{in}$ e de $\Omega(N, \mathcal{H}, \delta)$, para o termo $E_{out}$. No próximo post vamos discutir, principalmente, a questão do *tradeoff* entre viés e variância.

Qualquer dúvida, sugestão ou *feedback*, por favor, deixe um comentário abaixo.

> Essa postagem faz parte de uma [série]({{ site.baseurl }}/categories/aprendizado-de-máquina-learning-from-data/) de textos que tem o objetivo de estudar, principalmente, o curso "[Learning from Data](http://www.work.caltech.edu/telecourse.html)", ministrado pelo Professor Yaser Abu-Mostafa. Outros materiais utilizados serão sempre referenciados.