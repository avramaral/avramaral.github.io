---
title: "[Parte 11] Modelo de Regressão Logística & Implementação em Python"
categories: [Machine Learning - Learning from Data]
---

Voltando a falar de modelos lineares, como os de classificação $-$ explorados nas partes [02](/implementacao-perceptron/) e [04](/modelos-lineares-de-classificacao-e-pocket/) $-$ ou de regressão linear, a exemplo do que vimos na [parte 05](/modelo-de-regressao-linear/), iremos discutir nesse post o que é e como funciona a regressão logística. Veremos que esse novo modelo herda características das duas classes de algoritmos que já estudamos, já que é a função alvo é real-avaliada (como na regressão linear), mas é limitada (como no Perceptron ou Pocket, por exemplo). 

Na regressão logística, temos que $f(x) \in [0,1]$; e, como veremos a seguir, esse valor será interpretado como uma probabilidade. Para termos contexto, considere o seguinte exemplo: suponha que você quer modelar, com base em caracteríticas como idade, peso, nível de colesterol, etc., a probabilidade de um indivíduo sofrer um ataque cardíaco nos próximos 12 meses. De posse desse cenário, vamos começar a estabelecer as quantidades de interesse.

Primeiro, vamos começar definindo nossa classe de funções $\mathcal{H}$ para um problema de regressão logística. Nesse caso, teremos que

$$
\begin{align*}
h(\mathbf{x}) = \theta(\mathbf{w}^{\text{T}} \mathbf{x}),
\end{align*}
$$

onde $\theta(s) = \frac{e^s}{1 + e^s}$. Para enxergar como se comporta a função $\theta$, veja o gráfico a seguir.


```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

def theta_func(s):
    return np.exp(s) / (1 + np.exp(s))

x = np.linspace(-6, 6, num = 120)
y = theta_func(x)
plt.plot(x, y)
plt.axvline(0, linestyle = "--",  color = "black", linewidth = 1)
plt.axhline(0, linestyle = "--", color = "black", linewidth = 1)
plt.axhline(1, linestyle = "--", color = "black", linewidth = 1)
plt.xlim(-6, 6)
plt.xlabel("s")
plt.ylabel("$\Theta$(s)")
plt.show()
```


![png]({{ site.baseurl }}/assets/images/modelo-de-regressao-logistica_files/modelo-de-regressao-logistica_3_0.png)


Agora, podemos nos concentrar em analisar a função alvo que a regressão logística está tentando aprender. Logo,

$$
\begin{align*}
f(\mathbf{x}) = \mathbb{P}[y = +1 \mid \mathbf{x}].
\end{align*}
$$

Porém, perceba que, utilizando $\mathcal{D}$, não temos valores de $f(\mathbf{x})$, mas sim de amostras geradas a partir dessa distribuição de probabilidade (no nosso exemplo, não sabemos dizer a probabilidade com que um paciente sofreu um ataque cardíaco; nesse caso, só sabemos que esse incidente aconteceu). Por isso, estaremos interessados em trabalhar com, ao invés da função alvo, a **distribuição alvo**:

$$ 
P(y \mid \mathbf{x}) = 
\begin{cases}  
    f(\mathbf{x}) & \text{ para } y = +1 \\
    1 - f(\mathbf{x}) & \text{ para } y = -1.
\end{cases}
$$

Finalmente, só nos resta definir uma medida de erro $\text{e}(h(\mathbf{x}), y)$ para minimizar. Aqui, a ideia de verossimilhança irá exercer papel fundamental $-$ ou seja, o processo será o de encontrar a $h \in \mathcal{H}$ que, baseado em $\mathbf{x}$, melhor explica o resultado $y$. Assim, a verossimilhança será dada por:

$$
P(y \mid \mathbf{x}; \mathbf{w}) = 
\begin{cases}  
    h_{\mathbf{w}}(\mathbf{x}) & \text{ para } y = +1 \\
    1 - h_{\mathbf{w}}(\mathbf{x}) & \text{ para } y = -1.
\end{cases}
$$

Então, substituindo $h_{\mathbf{w}}(\mathbf{x})$ por $\theta(\mathbf{x}^{\text{T}} \mathbf{x})$ e utilizando o fato de que $\theta(-s) = 1 - \theta(s)$, temos que:

$$
\begin{align*}
P(y \mid \mathbf{x}; \mathbf{w}) = \theta(y \; \mathbf{w}^{\text{T}}\mathbf{x}).
\end{align*}
$$

Já que, por hipótese, os pontos de $\mathcal{D}$ são gerados de maneira independente, então:

$$
\begin{align*}
\prod_{n = 1}^{N} P(y_n \mid \mathbf{x}_n; \mathbf{w}) = \prod_{n = 1}^{N} \theta(y_n \; \mathbf{w}^{\text{T}}\mathbf{x}_n).
\end{align*}
$$

Como maximizar a função de verossimilhança é o mesmo que minizar o erro, podemos definir $E_{in}$ como:

$$
\begin{align*}
E_{in}(\mathbf{w}) & = -\frac{1}{N} \; \ln \left[ \prod_{n = 1}^{N} \theta(y_n \; \mathbf{w}^{\text{T}}\mathbf{x}_n) \right] \\
& = \frac{1}{N} \sum_{n = 1}^{N} \ln \left[ \frac{1}{\theta(y_n \; \mathbf{w}^{\text{T}}\mathbf{x}_n)} \right] \\
& = \frac{1}{N} \sum_{n = 1}^{N} \ln \left[ 1 + e^{- y_n \mathbf{w}^{\text{T}}\mathbf{x}_n} \right] \text{, tal que } \theta(s) = \frac{1}{1 + e^{-s}}
\end{align*}
$$

De forma explicita, acabamos de dizer que $\text{e}(h(\mathbf{x}_n), y_n)= \left[ 1 + e^{- y_n \mathbf{w}^{\text{T}}\mathbf{x}_n} \right]$. A medida de erro definida dessa maneira é chamada de *'cross entropy' error*.

Agora, a nossa missão é encontrar $\mathbf{w}$ que minimiza $E_{in}$; porém, da maneira como definimos o erro, não é possível determinar uma solução analítica para esse problema. Assim, utilizaremos uma técnica (de iteração) conhecida como **Gradiente Descendente**.

### Gradiente Descendente para Regressão Logística

Gradiente descendente é um algoritmo utilizado para minimizar funções que são, pelo menos, duas vezes diferenciáveis, tal como $E_{in}(\mathbf{w})$. A ideia é que, através de um processo de iteração, o algoritmo, que começa com $\mathbf{w}(0)$ definido arbitrariamanente, encontra $\mathbf{w}(m)$ $-$ para $m$ suficientemente grande $-$, que é *mínimo local* de $E_{in}(\mathbf{w})$. Entretanto, o fato de não existir garantia de que o *mínimo local* é, também, *mínimo global*, é um problema que precisa ser tratato (esse cenário será discutido em outra oportunidade). Felizmente, para a regressão logística (bem como para o modelo de regressão linear), $E_{in}$ é função convexa, o que implica que, nesse caso, o ponto de mínimo é único.

Dito isso, vamos descrever como encontrar esse ponto de mínimo. Aqui, suponha que vamos dar pequenos passos de tamanho $\eta$ na direção do vetor unitário $\mathbf{\hat{v}}$; assim, teremos $\mathbf{w}(1) = \mathbf{w}(0) + \eta \; \mathbf{\hat{v}}$. Dessa forma, com a intenção de fazer $\Delta E_{in}$ o menor possível, obtemos, utilizando a expansão de Taylor ([ref.](https://en.wikipedia.org/wiki/Taylor_series#Taylor_series_in_several_variables)):

$$
\begin{align*}
\Delta E_{in} & = E_{in}(\mathbf{w}(0) + \eta \mathbf{\hat{v}}) - E_{in}(\mathbf{w}(0)) \\
& = E_{in}(\mathbf{w}(0)) + (\mathbf{w}(1) - \mathbf{w}(0))^{\text{T}} \; \nabla E_{in}(\mathbf{w}(0)) + \mathcal{O}(\eta^2) - E_{in}(\mathbf{w}(0)) \\
& \geq (\eta \; \mathbf{\hat{v}})^{\text{T}} \; \nabla E_{in}(\mathbf{w}(0)) \\
& = \eta \; \nabla E_{in}(\mathbf{w}(0))^{\text{T}} \; \mathbf{\hat{v}} \\
& \geq - \eta \; \lvert \lvert \nabla E_{in}(\mathbf{w}(0)) \rvert \rvert \text{, já que } \max_{u: \lvert \lvert u \rvert \rvert = 1} \langle u, v \rangle = \lvert \lvert v \rvert \rvert.
\end{align*}
$$

Assim, em relação a última desigualdade, vale o "$=$" se e somente se:

$$
\begin{align*}
\mathbf{\hat{v}} = -\frac{\nabla E_{in}(\mathbf{w}(0))}{\lvert \lvert \nabla E_{in}(\mathbf{w}(0)) \rvert \rvert}.
\end{align*}
$$

Ou seja, o vetor unitário $\mathbf{\hat{v}}$ com direção como definido acima, é o que me dá a maior variação negativa de $\Delta E_{in}$ para um passo de tamanho $\eta$. A ideia, a partir desse ponto, é iterar sobre o processo que acabamos de descrever.

Porém, um problema surge: como podemos definir adequadamente o tamanho do passo $\eta$? Se ele for muito pequeno, como na imagem da esquerda (Fig. 1), o algoritmo é ineficiente quando não estamos perto do mínimo local; se ele for muito grande, como na imagem do centro, podemos, inclusive, aumentar $E_{in}$. Nesse caso, o ideal é tomar passos de tamanho proporcional à distância que estamos do ponto de mínimo, como na imagem da direita.

![Tamanho de eta]({{ site.baseurl }}/assets/images/modelo-de-regressao-logistica_files/tamanho-de-eta.png)
*Figura 1 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Opções para tamanho de $\eta$.*

Para alcançar uma solução como a apresentada na imagem da direita (Fig. 1) basta fazer $\eta = \eta \; \lvert\lvert \nabla E_{in} \rvert\rvert$. Essa estratégia funciona pois, longe do mínimo local, a norma vetor gradiente é tipicamente maior; ao passo que, perto do ponto de mínimo, essa quantidade diminui.

Por fim, como 

$$
\nabla E_{in} = - \frac{1}{N} \sum_{n = 1}^{N} \frac{y_n \mathbf{x}_n}{1 + e^{y_n \mathbf{w}^{\text{T}}\mathbf{x}_n}},
$$

somos capazes de implementar um algoritmo, como explicitado na Fig. 2, para resolver o problema de regressão logística.

![Algoritmo Regressão Logística]({{ site.baseurl }}/assets/images/modelo-de-regressao-logistica_files/algoritmo-reg-log.png)
*Figura 2 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Algoritmo para Regressão Logística.*

### Implementação em Python

Vamos começar por, seguindo as ideias que desenvolvemos ao longo do texto e o algoritmo apresentado na Fig. 2, implementar a classe `LogisticRegression`.


```python
# Raw implementation
class LogisticRegression:
    """
    Logist Regression Algorithm
    """
    def __init__(self, eta = 0.1, n_iter = 100, prob_threshold = 0.5, prob_out = False):
        self.eta = eta
        self.n_iter = n_iter
        self.prob_out = prob_out
        self.prob_threshold = prob_threshold
    
    def sigmoid_(self, s):
        return np.exp(s) / (1 + np.exp(s))
    
    def gradient_(self, X, y):
        partial_sum = 0
        X = np.c_[np.ones(X.shape[0]), X]
        for X_i, y_i in zip(X, y):
            num = y_i * X_i 
            den = 1 + (np.exp(y_i * (np.dot(X_i, self.w_))))
            partial_sum += (num / den)
        return (-1 / X.shape[0]) * partial_sum 
            
    def fit(self, X, y):
        self.w_ = np.zeros(X.shape[1] + 1)
        for _ in np.arange(self.n_iter):
            grad = self.gradient_(X, y)
            self.w_ -= (self.eta * grad) 
        return self
    
    def predict(self, X_i):
        probability = self.sigmoid_(np.dot(X_i, self.w_[1:]) + self.w_[0])
        if self.prob_out:
            return probability
        else:
            return np.where(probability >= self.prob_threshold, 1, -1)
```

Adotando o mesmo padrão de nomes para outras classes que já escrevemos, o método `fit()` ajusta o modelo $-$ utilizando, para isso, a função `gradient_()`. O método `predict()`, através da função `sigmoid_()`, calcula a probabilidade de uma nova observação assumir valor $+1$ ou, como usaremos a partir desse daqui, classifica um novo ponto de acordo com o limite `prob_threshold`.

Agora, vamos, com um conjunto de dados simulado, ajustar o modelo. Mas primeiro, veja como os pontos são distribuídos.


```python
N = 250

rn = np.random.RandomState(999)
c1 = rn.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], N)
c2 = rn.multivariate_normal([0, 3], [[1, 0.5], [0.5, 1]], N)

X = np.r_[c1, c2]
y = np.r_[np.asarray([-1 for _ in np.arange(N)]), np.asarray([1 for _ in np.arange(N)])]

plt.scatter(X[:N, 0], X[:N, 1], color = "red",   marker = "o", alpha = 0.5, label = "Class A")
plt.scatter(X[N:, 0], X[N:, 1], color = "green", marker = "s", alpha = 0.5, label = "Class B")
plt.legend()
plt.show()
```


![png]({{ site.baseurl }}/assets/images/modelo-de-regressao-logistica_files/modelo-de-regressao-logistica_26_0.png)


Perceba que as duas classes `Class A` e `Class B` **não** são linearmente separáveis; mais do que isso, na verdade: a região de interseção entre as duas nuvens de pontos é razoavelmente grande. Abaixo, o ajuste do modelo:


```python
my_logReg = LogisticRegression()
my_logReg.fit(X, y)
print(my_logReg.w_)
```

    [-1.1072962  -0.52486136  1.16919589]


A seguir, vamos plotar as regiões de decisão para cada uma das classes com `prob_threshold = 0.5`. Para isso, utilizaremos a nossa função, já várias vezes mencionada ao longo dos posts, `plot_decision_regions()`.


```python
def plot_decision_regions(X, y, classifier, feature_names, resolution = 0.01, plot_lim = 0.25):
    # general settings
    markers = ["o", "s", "*", "x", "v"]
    colors  = ("red", "green", "blue", "gray", "cyan")
    x1_min, x1_max = X[:, 0].min() - plot_lim, X[:, 0].max() + plot_lim
    x2_min, x2_max = X[:, 1].min() - plot_lim, X[:, 1].max() + plot_lim
    # define a grid
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # classify each grid point
    result = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    result = result.reshape(xx1.shape)
    # make a plot
    plt.contourf(xx1, xx2, result, colors = colors[0:len(np.unique(y))], alpha = 0.5)
    for index, value in enumerate(np.unique(y)): 
        plt.scatter(x = X[y == value, 0], y = X[y == value, 1], 
                    color = colors[index],
                    marker = markers[index],
                    label = feature_names[index],
                    edgecolor = 'black')
```


```python
feature_names = ['Classe A', 'Classe B']
plot_decision_regions(X, y, my_logReg, feature_names, plot_lim = 0.15)
plt.title("Fitted Model with Raw Implementation of Logistic Regression")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend(loc = 2)
plt.show()
```


![png]({{ site.baseurl }}/assets/images/modelo-de-regressao-logistica_files/modelo-de-regressao-logistica_31_0.png)


No gráfico acima, veja que conseguimos, à medida do que é possível com o conjunto de dados de treinamento $\mathcal{D}$, criar regiões que classificam corretamente a maior parte dos pontos.

Entretanto, abordando o problema dessa maneira $-$ como um problema de classificação simples $-$, perdemos uma característica importante do modelo de regressão logística: a ideia de que cada ponto vale $+1$ com **probabilidade** $h_{\mathbf{w}}(\mathbf{x})$ (na verdade, $g_{\mathbf{w}}(\mathbf{x})$ $-$ se $g \in \mathcal{H}$ foi escholida por $\mathcal{A}$). Veja o trecho de código abaixo:


```python
# Test: predict the probability, instead of the class
my_logReg.prob_out = True
chosen_point  = np.array([0, 1]) # a very 'difficult' point to classify
print("With probability {:.2f}, the point {} is evaluated as +1.".format(my_logReg.predict(chosen_point), chosen_point))
```

    With probability 0.52, the point [0 1] is evaluated as +1.


Observe que o ponto $(0, 1)$, analisando as regiões de decisão que foram plotadas, está no limiar da fronteira de classificação. O que faz sentido com o fato de que obtivemos $\mathbb{P}[y = +1 \mid \mathbf{x}] = 0.52$; ou seja, apesar de $(0, 1)$ ter sido classificado como $+1$ (para o threshold de $0.5$), isso aconteceu por bem pouco. 

Para finalizar, assim como já fizemos outras vezes, vamos utilizar a implementação do algoritmo pela biblioteca Sklearn.


```python
from sklearn.linear_model import LogisticRegression
sk_logReg = LogisticRegression()
sk_logReg.fit(X, y)
print(sk_logReg.intercept_)
print(sk_logReg.coef_)
```

    [-4.5329416]
    [[-1.70651758  3.08848517]]



```python
feature_names = ['Class A', 'Class B']
plot_decision_regions(X, y, sk_logReg, feature_names, plot_lim = 0.15)
plt.title("Fitted model with Sklearn version of Logistic Regression")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend(loc = 2)
plt.show()
```


![png]({{ site.baseurl }}/assets/images/modelo-de-regressao-logistica_files/modelo-de-regressao-logistica_38_0.png)


Como era de se espera, a utilização da classe `LogisticRegression` é simples e apresenta resultados semelhantes aos que obtivemos acima. A performance nesse segundo caso é, obviamente, muito mais otimizada; porém, como forma de aprendizado, a nossa classe cumpriu bem a tarefa.

## Conclusão

Complementando os modelos lineares (e transformações não-lineares) que já havíamos começado a estudar nas partes [05](/modelo-de-regressao-linear/) e [06](/transformacoes-nao-lineares/), vimos como o modelo de Regressão Logística pode ser utilizado para, por exemplo, problemas de classificação. Entretando, a sua capacidade de fazer esse tipo de análise do ponto de vista probabilístico é o que é o mais interessante. Implementamos o algoritmo em Python e, em adição às classes que escrevemos antes, estamos começando a construir um bom repositório de algoritmos de *machine learning*. O próximo post discutirá o assunto "*neural networks*" (ou "redes neurais").




Qualquer dúvida, sugestão ou *feedback*, por favor, deixe um comentário abaixo.

> Essa postagem faz parte de uma [série]({{ site.baseurl }}/categorias/#machine-learning-learning-from-data) de textos que tem o objetivo de estudar, principalmente, o curso "[Learning from Data](http://www.work.caltech.edu/telecourse.html)", ministrado pelo Professor Yaser Abu-Mostafa. Outros materiais utilizados serão sempre referenciados.