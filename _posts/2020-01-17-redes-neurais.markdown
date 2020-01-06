---
title: "[Parte 12] Redes Neurais"
categories: [Aprendizado de Máquina - Learning from Data]
tags: [aprendizado-de-maquina]
---

Ao longo desse post vamos, como temos feito para todos os modelos que discutimos até agora [nessa série de textos]({{ site.baseurl }}/categories/aprendizado-de-máquina-learning-from-data/), estudar o que são, do ponto de vista mais teórico, as redes neurais. A ideia é que, na próxima postagem, a gente consiga implementar em Python o que vamos estudar a partir desse momento. 

Para começar, podemos tentar definir o que são as **redes neurais**. Bem, há várias analogias sobre como o modelo matemático que recebe o nome de "rede neural" se compara à forma através da qual o ser humano aprende, etc., etc. Mas, grosso modo, redes neurais são modelos, compostos por pequenas "peças", que têm o objetivo de aprender (no sentido que temos estudado) funções mais complexas. Essas "peças" podem, em essência, ser construídas a partir de qualquer transformação não-linear. Porém, o caso mais comum é trabalharmos com blocos que utilizam funções do tipo "*s-shaped curves*". A imagem abaixo ilustra uma combinação desse tipo.

![Rede Neural]({{ site.baseurl }}/assets/images/redes-neurais/rede-neural.png)
*Figura 1 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Rede Neural.*

A partir da Fig 1., podemos identificar algumas componentes. Em primeiro lugar, cada coluna de "nós" (ou "*nodes*") é chamada de "*layer*" (ou "camada"); à primeira camada, vamos dar o nome de "*input*", e à última, "*output*" $-$ as camadas intermediárias vão ser chamadas de "*hidden layers*". A função de ativação $\theta$ será definida a partir da "tangente hiperbólica"; i.e., $\theta(s) = \tanh(s) = \frac{e^s - e^{-s}}{e^s + e^{-s}}$. A imagem abaixo ilustra o comportamento de $\theta(s)$.


```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

def theta_func(s):
    return (np.exp(s) - np.exp(-1 * s)) / (np.exp(s) + np.exp(-1 * s))

x = np.linspace(-6, 6, num = 120)
y = theta_func(x)
plt.plot(x, y)
plt.axvline( 0, linestyle = "--", color = "black", linewidth = 1)
plt.axhline( 0, linestyle = "--", color = "black", linewidth = 1)
plt.axhline(-1, linestyle = "--", color = "black", linewidth = 1)
plt.axhline( 1, linestyle = "--", color = "black", linewidth = 1)
plt.xlim(-6, 6)
plt.xlabel("s")
plt.ylabel("$\Theta$(s)")
plt.show()
```


![png]({{ site.baseurl }}/assets/images/redes-neurais/redes-neurais_4_0.png)


Como pode ser visto a partir do gráfico acima, a maneira através da qual definimos a função $\theta$ é muito parecida com o que fizemos na [parte 11](/modelo-de-regressao-logistica/), quando estudamos o modelo de regressão logística. A razão de termos escolhido uma função com contradomínio ligeiramente diferente $-$ nesse caso, $\tanh$ $-$ é a facilidade que temos em lidar com sua derivada.

Agora, vamos estabelecer algumas notações que nos serão úteis ao longo do texto. As camadas (ou *labels*) serão denotadas por $l = 0, 1, 2, \cdots, L$; por exemplo, a camada $l = L$ é aquela à qual demos o nome de *output*. Para nos referirmos a uma determiada camada $l$, utilizaremos o sobrescrito ${}^{(l)}$. Cada *layer* tem dimensão $d^{(l)}$, o que significa que em $l$ existem $d^{(l)} + 1$ nós (lembre-se de $x_0$, chamado também de *bias*).

O conjunto de hipóteses para o modelo de rede neural será denotado por $\mathcal{H}_{nn}$, e é completamente especificado uma vez que é determinada a *arquitetura* da rede; ou seja, a dimensão $$\mathbf{d} = [d^{(0)}, \cdots, d^{(L)}]$$ para todas as *layers*. Similarmente, uma hipótese $$h \in \mathcal{H}_{nn}$$ é caracterizada pelo peso $$w^{(l)}_{i j}$$ atribuído a cada "flecha" da rede. Vamos olhar de mais perto um par de nós da nossa rede.

![Componentes entre um par de nós]({{ site.baseurl }}/assets/images/redes-neurais/uma-relacao.png)
*Figura 2 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Componentes entre um par de nós.*

Da Fig. 2, um nó tem um sinal de entrada $s$ e um de saída $x$. O peso que conecta o nó $j$ (da camada $l$) ao nó $i$ (da camada **anterior**) é denotado por, como já havíamos visto, $w^{(l)}_{i j}$. Dessa forma, teremos $1 \leq l \leq L$, $0 \leq i \leq d^{(l - 1)}$ e $1 \leq j \leq d^{(l)}$.

Além disso, quando tivermos o interesse *macro* de analisar os termos da nossa rede; ou seja, olhar para as camadas como componentes únicas, podemos utilizar a notação vetorial. Para a coleção de sinais de entrada $1, \cdots, d^{(l)}$ na camada $l$, teremos $\mathbf{s}^{(l)}$. De forma parecida, para o conjunto de saídas em $l$, iremos utilizar o vetor $\mathbf{x}^{(l)}$ $-$ composto por elementos de índices $0, 1, \cdots, d^{(l)}$. Para as "flechas" que conectam $(l-1)$ à camada $l$, existirá uma matriz $(d^{(l-1)} + 1) \times d^{(l)}$ de pesos $W^{(l)}$, tal que a $(i, j)$-ésima entrada de $W^{(l)}$ é o termo $w^{(l)}_{ij}$. Por fim, todas as matrizes do tipo $W^{(l)}$ serão guardadas em um vetor $\mathbf{w} = [W^{(1)}, \cdots, W^{(L)}]$.

### Forward Propagation

Para calcularmos o valor de $h(\mathbf{x})$ (e, por consequência, o erro $e(h(\mathbf{x}), f(\mathbf{x})$), vamos utilizar o algoritmo "*Forward Propagation*" (ou, em português, algo como "Propagação para Frente"). Mas antes de qualquer outra coisa, observe que as entradas e saídas em uma camada $l$ podem ser relacionadas por:

$$
\begin{align*}
\mathbf{x}^{(l)} = \left[ 1, \theta(\mathbf{s}^{(l)})\right]^{\text{T}},
\end{align*}
$$

onde $$\theta(\mathbf{s}^{(l)})$$ é o vetor cujas componentes são $$\theta(s^{(l)}_j)$$. Dessa forma, temos que $$s^{(l)}_j = \sum_{i = 0}^{d^{(l-1)}} w^{(l)}_{ij} x^{(l-1)}_i$$. Em notação vetorial:

$$
\begin{align*}
\mathbf{s}^{(l)} = \left( W^{(l)} \right)^{\text{T}} \mathbf{x}^{(l-1)}.
\end{align*}
$$

Note que, agora, só falta atribuírmos $\mathbf{x}$ a $\mathbf{x}_0$. Sendo assim, podemos utilizar o algoritmo abaixo para calcular $h(\mathbf{x})$.

![Algoritmo Forward Propagation]({{ site.baseurl }}/assets/images/redes-neurais/algoritmo-forward.png)
*Figura 3 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Algoritmo Forward Propagation.*

Calculado $x^{(L)} = h(\mathbf{x})$ podemos, finalmente, determinar $E_{in}$. Utilizando a medida de erro quadrático; isto é, $e(h(\mathbf{x}), f(\mathbf{x})) = (h(\mathbf{x}) - f(\mathbf{x}))^2$, temos:

$$
E_{in} = \frac{1}{N} \sum_{n = 1}^{N}(\mathbf{x}^{(L)}_n - f(\mathbf{x}_n))^2.
$$

Agora, o que temos que fazer é, como de costume, minimizar $E_{in}$. Discutiremos isso na próxima seção.

### Backpropagation Algorithm

Na [parte 11](/modelo-de-regressao-logistica/), utilizamos o algoritmo *gradient descent* para encontrar $\mathbf{w}$ tal que $E_{in}(\mathbf{w})$ é mínimo (local). Bastou inicializar $\mathbf{w}(0)$ e, para $t = 1, 2, \cdots$, atualizar o vetor de pesos da seguinte maneira:

$$
\begin{align*}
\mathbf{w}(t+1) = \mathbf{w}(t) - \eta \nabla E_{in}(\mathbf{w}(t)).
\end{align*}
$$

Porém, para implementar esse procedimento, precisamos encontrar o vetor gradiente. Poderíamos fazer isso "na mão"; mas é nesse ponto em que o *Backpropagation Algorithm* (ou "Algoritmo de Propagação para Trás") entra em ação.

Esse algoritmo se baseia em aplicações sucessivas da "regra da cadeia" a fim de escrever as derivadas parciais na *layer* $l$ utilizando, para isso, as derivadas parciais de $l + 1$.

Comece definindo o "vetor sensitivo" ($\mathbf{\delta}^{(l)}$) para a camada $l$ da seguinte forma:

$$
\begin{align*}
\mathbf{\delta}^{(l)} = \frac{\partial e(\mathbf{w})}{\partial \mathbf{s}^{(l)}};
\end{align*}
$$

ou seja, $\mathbf{\delta}^{(l)}$ é a derivada parcial da medida de erro $e(\mathbf{w})$ com respeito ao sinal de entrada $\mathbf{s}^{(l)}$. Dessa forma, podemos escrever a derivada de interesse como:

$$
\begin{align*}
\frac{\partial e(\mathbf{w})}{\partial W^{(l)}} = \mathbf{x}^{(l-1)}(\mathbf{\delta}^{(l)})^{\text{T}}
\end{align*}.
$$

Aqui, note que $\frac{\partial \mathbf{s}^{(l)}}{\partial W^{(l)}} = \mathbf{x}^{(l-1)}$. Assim, como o termo $\mathbf{x}^{(l)}$, para $l \geq 0$, pode ser obtido por *forward propagation*, é suficiente que nos preocupemos apenas com $\mathbf{\delta}^{(l)}$. E é aqui que as coisas ficam interessantes. Com uma pequena modificação na rede e "rodando-a" *ao contrário* (por isso o nome "*backpropagation*"), conseguimos $\mathbf{\delta}^{(l)}$.

Ao invés de cada *layer* "cuspir" o vetor $\mathbf{x}^{(l)}$ (como acontece quando estamos no algoritmo *forward propagation*), fazendo o caminho inverso, cada camada irá devolver o vetor $\mathbf{\delta}^{(l)}$. 

Em relação à "pequena modificação" que temos que fazer, agora cada nó faz uma transformação do tipo "multiplicação por $\theta^{\prime}(\mathbf{s}^{(l)})$". Sendo assim, se $\theta(\cdot) = \tanh(\cdot)$, então $\tanh^{\prime}(\mathbf{s}^{(l)}) = 1-\tanh^2(\mathbf{s}^{(l)}) = 1 - \mathbf{x}^{(l)} \otimes\mathbf{x}^{(l)}$; onde $\otimes$ significa produto termo-a-termo. A imagem abaixo ilustra esse procedimento.

![Backpropagation]({{ site.baseurl }}/assets/images/redes-neurais/backpropagation.png)
*Figura 4 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Esquema de "Backpropagation".*

Além disso, a partir da Fig. 4, vemos que a camada $(l+1)$ "cospe" (para trás) o vetor $\mathbf{\delta}^{(l+1)}$, que é multiplicado por $W^{(l+1)}$, somado, e entregue para os nós da *layer* $l$. Essa operação pode ser escrita como:

$$
\begin{align*}
\mathbf{\delta}^{(l)} = \theta^{\prime}(\mathbf{s}^{(l)}) \otimes \left[ W^{(l + 1)} \mathbf{\delta}^{(l+1)} \right]^{d^{(l)}}_1,
\end{align*}
$$

onde o vetor $\left[ W^{(l + 1)} \mathbf{\delta}^{(l+1)} \right]^{d^{(l)}}_1$ contém as componentes de $W^{(l + 1)} \mathbf{\delta}^{(l+1)} $, excluíndo o valor de índice zero (excluindo o termo "*bias*"). **Observação 1:** a igualdade acima não é trivial; para ver o passo-a-passo de como obtê-la, consulte o capítulo e-7 de
[Learning from Data]().

Sendo assim, se temos $\mathbf{\delta}^{(l+1)}$, podemos encontrar $\mathbf{\delta}^{(l)}$. Para iniciar essa "cadeia", basta determinar $\mathbf{\delta}^{(L)}$:

$$
\begin{align*}
\mathbf{\delta}^{(L)} & = \frac{\partial e(\mathbf{w})}{\partial \mathbf{s}^{(L)}} \\
& = \frac{\partial}{\partial \mathbf{s}^{(L)}} (\mathbf{x}^{L} - f(\mathbf{x}))^2 \\
& 2(\mathbf{x}^{(L)} - f(\mathbf{x})) \frac{\partial \mathbf{x}^{(L)}}{\partial \mathbf{s}^{(L)}} \\
& 2(\mathbf{x}^{(L)} - f(\mathbf{x})) \theta^{\prime}(\mathbf{s}^{(L)}).
\end{align*}
$$

**Observação 2:** quando a transformação de saída é $\theta(\cdot) = \tanh(\cdot)$, temos que $\theta^{\prime}(\mathbf{s}^{(L)}) = 1 - (\mathbf{x}^{(L)})^2$; porém, quando a transformação de saída é a função identidade, $\theta^{\prime}(\mathbf{s}^{(L)}) = 1$. **Observação 3:** se existir somente um nó de saída, $\mathbf{s}^{(L)}$ é escalar (e, por consequência, $\mathbf{\delta}^{(L)}$ também).

Por fim, utilizando a fórmula de $\mathbf{\delta}^{(l)}$, podemos calcular todos os "vetores sensitivos". O algoritmo para esse tipo de dedução é apresentado a seguir (assumindo transformação nas *hidden layers* igual a $\tanh(\cdot)$ $-$ se esse não for o caso, adaptar o passo $3$). 

![Algoritmo Backpropagation]({{ site.baseurl }}/assets/images/redes-neurais/calculate-sensitivity.png)
*Figura 5 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Algoritmo "Backpropagation".*

Agora, para obter a derivada da medida de erro com respeito ao vetor de pesos, basta lembrar que $\frac{\partial e(\mathbf{w})}{\partial W^{(l)}} = \mathbf{x}^{(l-1)}(\mathbf{\delta}^{(l)})^{\text{T}}$.

Estamos quase lá! O que vamos fazer a seguir é calcular a derivada de $E_{in}(h)$, como gostaríamos que fosse feito (mas veremos na última parte desse texto que existe, nesse sentido, uma abordagem melhor: o *stochastic gradient descent*):

$$
\begin{align*}
\frac{\partial E_{in}}{\partial W^{(l)}} = \frac{1}{N} \sum^{N}_{n = 1} \frac{\partial e(\mathbf{x}_n)}{\partial W^{(l)}}.
\end{align*}
$$

O algorimo a seguir, onde $G^{(l)}(\mathbf{x}_n)$ é o vetor gradiente no ponto $\mathbf{x}_n$, sumariza tudo que fizemos até agora.

![Algoritmo Gradiente]({{ site.baseurl }}/assets/images/redes-neurais/algoritmo-gradiente.png)
*Figura 6 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Algoritmo para cálculo do vetor gradiente.*

### Stochastic Gradient Descent

Existe uma versão do algoritmo de gradiente descendente, conhecida como *Stochastic Gradient Descent*, que reduz consideravelmente o tempo computacional gasto para minimizar $E_{in}$.

No algoritmo de *gradient descent* que havíamos visto até então, o vetor gradiente era calculado para todo o conjunto de dados antes de sermos capazes de atualizar $\mathbf{w}$. Nessa nova versão, ao invés de considerar os $N$ pontos, escolha um $(\mathbf{x}_n, y_n)$ uniformemente em $\mathcal{D}$ e leve em conta apenas esse 'par ordenado' para calcular a derivada ($e^{\prime}(\mathbf{w})$); então, esse vetor gradiente obtido é utilizado para atualizar o vetor de pesos **da mesma forma** que fazíamos antes. 

A ideia de o porquê isso funciona vem do fato de que:

$$
\begin{align*}
\mathbb{E}_{\mathbf{x}_n}\left[\nabla e(h(\mathbf{x}_n), f(\mathbf{x}_n))\right] & = \frac{1}{N}\sum_{n = 1}^{N} \nabla e(h(\mathbf{x}_n), f(\mathbf{x}_n)) \\
& = \nabla E_{in}(h);
\end{align*}
$$

ou seja, "em média", o vetor gradiente leva o processo de minimização para a direção correta (exceto por pequenos desvios). 

Além de ser computacionalmente mais barato, o *Stochastic Gradient Descent* também nos ajuda a contornar problemas de "ficar preso em mínimos locais" $-$ já que a minimização, como dito, não é "direta".

Se utilizarmos essa técnica para minimizar o termo $E_{in}$ na nossa rede neural, o algoritmo **geral** pode ser estabelecido como:

![Algoritmo Final]({{ site.baseurl }}/assets/images/redes-neurais/algoritmo-final.png)
*Figura 7 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Algoritmo geral para rede neural.*

## Conclusão

Ao longo do texto, vimos como resolver, pedaço por pedaço, os problemas que enfrentamos quando tentamos implementar um modelo do tipo "rede neural". Vimos que, depois de construída a arquitetura da rede, calcular as medidas de erro não foi um problema $-$ bastou utilizar o algoritmo *Forward Propagation*. Porém, quando tentammos encontrar $\mathbf{w}$ que minimiza $E_{in}$, percebemos que definir o vetor gradiente $\nabla E_{in}(\mathbf{w})$ não é tarefa fácil. Aqui, o *Backpropagation Algorithm* nos foi muito útil e resolveu esse problema. Por fim, vimos a vantagem de implementar o algoritmo *Stochastic Grandient Descent* ao invés do gradiente descendente "normal" que havíamos visto no post passado. Na próxima postagem, vamos fazer a implementação em Python do que estudamos hoje.








Qualquer dúvida, sugestão ou *feedback*, por favor, deixe um comentário abaixo.

> Essa postagem faz parte de uma [série]({{ site.baseurl }}/categories/aprendizado-de-máquina-learning-from-data/) de textos que tem o objetivo de estudar, principalmente, o curso "[Learning from Data](http://www.work.caltech.edu/telecourse.html)", ministrado pelo Professor Yaser Abu-Mostafa. Outros materiais utilizados serão sempre referenciados.