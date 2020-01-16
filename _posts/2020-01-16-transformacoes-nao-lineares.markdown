---
title: "[Parte 06] Transformações NÃO-lineares & Implementação em Python"
categories: [Machine Learning - Learning from Data]
---

Como vimos até agora [nessa série de textos]({{ site.baseurl }}/categorias/#machine-learning-learning-from-data), modelos lineares (tanto de classificação quanto de regressão) utilizam a quantidade $\sum_{i = 0}^{d} w_i x_i$ para calcular a função $h \in \mathcal{H}$. Note que essa expressão é linear para os $x_i$'s e $w_i$'s; porém, como discutimos na [parte 05](/modelo-de-regressao-linear/), os $x_i$'s podem, do ponto de vista do algoritmo, ser encaradados como constantes (pense que o conjunto de dados $\mathcal{D}$, no momento que vamos ajustar o modelo, já está definido). Dessa forma, basta que a expressão que estamos considerando seja linear para o vetor de pesos $-$ o que, em outras palavras, significa dizer que podemos aplicar tranformações não-lineares em $\mathbf{x} \in \mathcal{D}$.

Nessa postagem vamos discutir (e implementar) dois exemplos: um de classificação $-$ utilizando o algoritmo Perceptron $-$, e outro de regressão; ambos com transformações não lineares aplicadas no conjunto de dados.

### Classificação

Para o primeiro caso, vamos gerar um conjunto de dados que será classificado de acordo com a posição de um ponto dentro (ou fora) de um circunferência com centro na origem.

Veja como o conjunto de dados foi gerado, bem como uma representação gráfica dos pontos.


```python
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rn = np.random.RandomState(999)
x1 = rn.normal(0, 1.00, 100)
x2 = rn.normal(0, 1.00, 100)

mask_in  = ((x1 ** 2) + (x2 ** 2) - 1.00 < 0)
mask_out = ((x1 ** 2) + (x2 ** 2) - 1.44 > 0) & ((x1 ** 2) + (x2 ** 2) - 4.00 < 0) 

x1_in  = x1[mask_in]
x2_in  = x2[mask_in]
x1_out = x1[mask_out]
x2_out = x2[mask_out]

X = np.array(np.c_[x1_in.tolist() + x1_out.tolist(), x2_in.tolist() + x2_out.tolist()])
y = np.asarray([np.where(y < x1_in.shape[0], -1, 1) for y in np.arange(x1_in.shape[0] + x1_out.shape[0])])

plt.scatter(X[:x1_in.shape[0], 0], X[:x1_in.shape[0], 1], color = "red", marker = "o", label = "In")
plt.scatter(X[x1_in.shape[0]:, 0], X[x1_in.shape[0]:, 1], color = "green", marker = "s", label = "Out")
plt.axis("scaled")
plt.xlim(-2.25, 2.25)
plt.ylim(-2.25, 2.25)
plt.xlabel("original $x_1$")
plt.ylabel("original $x_2$")
plt.legend(loc = 1)
plt.show()
```


![png]({{ site.baseurl }}/assets/images/transformacoes-nao-lineares_files/transformacoes-nao-lineares_4_0.png)


Nesse exemplo, pontos sorteados segundo uma distribuição $N(0, 1)$ para cada uma das coordenadas foram definidos como `In` $-$ caso o ponto esteja dentro da circunferência de raio $1$ $-$ ou `Out` $-$ caso o ponto esteja fora da circunferência de raio $1.2$ e dentro da circunferência de raio $2$. Obviamente, essas duas classes **não** são linearmente separáveis.

Assim, vamos aplicar uma transformação nos pontos de $\mathbf{x}$ tal que $\phi: (x_1, x_2) \rightarrow ({x_1}^2, {x_2}^2)$, para todo $\mathbf{x} \in \mathcal{D}$. Veja, agora, como os dados transformados podem ser representados:


```python
X_transformed = np.power(X, 2)

plt.scatter(X_transformed[:x1_in.shape[0], 0], X_transformed[:x1_in.shape[0], 1], color = "red", marker = "o", label = "In")
plt.scatter(X_transformed[x1_in.shape[0]:, 0], X_transformed[x1_in.shape[0]:, 1], color = "green", marker = "s", label = "Out")
plt.axis("scaled")
plt.xlim(-0.25, 4.25)
plt.ylim(-0.25, 4.25)
plt.xlabel("transformed $x_1$")
plt.ylabel("transformed $x_2$")
plt.legend(loc = 1)
plt.show()
```


![png]({{ site.baseurl }}/assets/images/transformacoes-nao-lineares_files/transformacoes-nao-lineares_7_0.png)


Feita as transformação nos dados de acordo com a nossa função $\phi$, para todo $\mathcal{x} \in \mathcal{D}$ (nesse caso, por construção), o meu conjunto de pontos é linearmente separável. Sendo assim, posso aplicar, por exemplo, o Perceptron (utilizando o Sklearn $-$ caso queira ver a implementação completa do algoritmo, consulte a [parte 02](/implementacao-perceptron/)):


```python
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(X_transformed, y)

print("Intercept weight: {}.".format(perceptron.intercept_))
print("Weights vector: {}.".format(perceptron.coef_))
```

    Intercept weight: [-2.].
    Weights vector: [[2.36966542 1.48327095]].


De posse do modelo ajustado; i.e., do vetor $\mathbf{w}$ completamente definido, podemos plotar as regiões de decisão, utilizando a função `plot_decision_regions()` introduzida, pela primeira vez, na [parte 02](/implementacao-perceptron/).


```python
# modified function to admit data transformation
def plot_decision_regions(X, y, classifier, feature_names, modified = False, transf1 = lambda x: x, transf2 = None, resolution = 0.01, axis_lim = 1):
    # general settings
    markers = ["o", "s", "*", "x", "v"]
    colors  = ("red", "green", "blue", "gray", "cyan")
    x1_min, x1_max = X[:, 0].min() - axis_lim, X[:, 0].max() + axis_lim
    x2_min, x2_max = X[:, 1].min() - axis_lim, X[:, 1].max() + axis_lim
    # define a grid
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # tranform the point that will be predicted, but NOT the one which will be plotted
    if modified:
        if transf2 == None:
            transf2 = transf1
        xx1_mod = transf1(xx1)
        xx2_mod = transf2(xx2)
    else:
        xx1_mod = xx1
        xx2_mod = xx2      
    # classify each grid point
    result = classifier.predict(np.array([xx1_mod.ravel(), xx2_mod.ravel()]).T)
    result = result.reshape(xx1_mod.shape)
    # make a plot
    plt.contourf(xx1, xx2, result, colors = colors[0:len(np.unique(y))], alpha = 0.5)
    for index, value in enumerate(np.unique(y)): # plot each point & 'enumerate()' returns index and value of the given array
        plt.scatter(x = X[y == value, 0], y = X[y == value, 1], # select each X and y vectors by creating a mask
                    color = colors[index],
                    marker = markers[index],
                    label = feature_names[index],
                    edgecolor = 'black')
```

Perceba que a função foi levemente modificada para comportar dois novos parâmetros: `transf1` e `transf2` (além do `modified`, que é, apenas, uma tecnicalidade). Esses variáveis recebem uma função responsável pela transformação de cada uma das *features* do nosso conjunto de dados (por exemplo, uma transformação como a definida por $\phi(\mathbf{x})$). Mas antes de utilizarmos esse novo artifício, vamos ver como ficam as regiões de decisão para os dados transformados:


```python
feature_names = ["In", "Out"]
plot_decision_regions(X_transformed, y, perceptron, feature_names)
plt.title("Fitted model using transformed $X$")
plt.xlabel("Transformed $x_1$")
plt.ylabel("Transformed $x_2$")
plt.axis("scaled")
plt.xlim(-0.25, 4.25)
plt.ylim(-0.25, 4.25)
plt.legend(loc = 1)
plt.show()
```


![png]({{ site.baseurl }}/assets/images/transformacoes-nao-lineares_files/transformacoes-nao-lineares_13_0.png)


Por enquanto, nada de novo; como era de se esperar, os dados foram corretamente classificados e separados.

Agora o que podemos fazer (que é, de fato, a parte interessante) é contruir as regiões de decisão para os dados **originais**; nesse caso, utilizaremos os parâmetros `transf1` e `transf2` passando, como argumento, a função `lambda: x: x ** 2` $-$ ou seja, do mesmo modo que definimos $\phi$. Veja:


```python
feature_names = ["In", "Out"]
plot_decision_regions(X, y, perceptron, feature_names, True, lambda x: np.power(x, 2))
plt.title("Fitted model using original $X$")
plt.xlabel("Original $x_1$")
plt.ylabel("Original $x_2$")
plt.axis("scaled")
plt.xlim(-2.25, 2.25)
plt.ylim(-2.25, 2.25)
plt.legend(loc = 1)
plt.show()
```


![png]({{ site.baseurl }}/assets/images/transformacoes-nao-lineares_files/transformacoes-nao-lineares_15_0.png)


Tomando esse tipo de estratégia, nós conseguimos, como pode ser visto acima, classificar dois grupos de pontos que, à princípio, não eram linearmente separáveis.

### Regressão

Agora, nesse segundo exemplo, vamos estudar um problema de regressão. Para isso, considere o seguinte cenário: suponha que as informações "salário" e "felicidade" (para alguma medida arbitrária que captura esse sentimento) foram coletadas a partir de um grupo de $100$ indivíduos; suponha, ainda, que o salário dessas pessoas segue distribuição $\text{Gamma}(2, 5000)$ (para uma distribuição desse tipo, temos média $10\times 10^3$ e assimetria à direita) $-$ veja o histograma a seguir.


```python
rn = np.random.RandomState(1)
X  = rn.gamma(2, 5000, 100)
X  = X.reshape(100, 1)
plt.hist(X, color = "red", alpha = 0.5, bins = 10)
plt.xlabel("Salary (\$)")
plt.ylabel("Frequency")
plt.xticks(np.arange(0, 45e3, 5e3))
plt.show()
```


![png]({{ site.baseurl }}/assets/images/transformacoes-nao-lineares_files/transformacoes-nao-lineares_19_0.png)


Assim, temos vários indivíduos que ganham salários em torno dos dez mil reais, e alguns outros poucos trabalhadores que ganham quantias muito maiores do que essa.

Agora, suponha que a "felicidade" depende do "salário", mas essa relação não é linear; ou seja, depois de uma determinada quantidade de dinheiro, receber mais não se traduz em ser proporcionalmente mais feliz. Podemos representar essa dependência através da função $\log(\cdot)$.


```python
y = np.log(X) + rn.normal(0, 0.2, 100).reshape(100, 1)
plt.scatter(X, y, marker = "o", color = "red")
plt.title("Salary (\$) vs. Happiness")
plt.xlabel("Salary (\$)")
plt.ylabel("Happiness measure")
plt.xticks(np.arange(0, 45e3, 5e3))
plt.show()
```


![png]({{ site.baseurl }}/assets/images/transformacoes-nao-lineares_files/transformacoes-nao-lineares_22_0.png)


No código acima, perceba perceba que a "felicidade", como função do "salário", foi somada a valores que vêm de uma distruição $N(0, 0.04)$ $-$ esse termo será denominado por "ruído" ($\epsilon$), e refere-se à diferença entre os valores observado e **real** (não observável) da variável dependente. Lembre-se de que, quando falamos de regressão linear, estamos interessados em estudar um modelo do tipo: $y = w_0 + w_1 x_1 + \cdots + w_d x_d + \epsilon$.

Visualmente (e por construção, nesse caso), a relação descrita acima **não** é linear; nesse caso, faz sentido aplicarmos alguma transformação na variável independente.


```python
X_transformed = np.log(X)
plt.scatter(X_transformed, y, marker = "s", color = "green", alpha = 0.75)
plt.title("Transformed Salary ($\log$ of \$) vs. Happiness")
plt.xlabel("Transformed Salary ($\log$ of \$)")
plt.ylabel("Happiness measure")
plt.show()
```


![png]({{ site.baseurl }}/assets/images/transformacoes-nao-lineares_files/transformacoes-nao-lineares_25_0.png)


Note que, aplicando $\log(\cdot)$ em $x$, obtemos algo que se aproxima mais de uma relação linear. Nesse caso, podemos ajustar o modelo de regressão (utilizando o Sklearn $-$ caso queira ver a implementação completa do algoritmo, consulte a [parte 05](/modelo-de-regressao-linear/)). Primeiro, ajustaremos o modelo considerando $x$ e, depois, considerando $\log(x)$.


```python
from sklearn.linear_model import LinearRegression

regression_original = LinearRegression()
regression_original.fit(X, y)

x_min = X[:, 0].min()
x_max = X[:, 0].max()
x_values = np.arange(x_min, x_max, 100)
y_values = regression_original.predict(np.asarray(x_values).reshape(len(x_values), 1))
plt.plot(x_values, y_values, color = "darkred")
plt.scatter(X, y, marker = "o", color = "red")
plt.title("Fitted Regression for Salary")
plt.xlabel("Salary (\$)")
plt.ylabel("Happiness measure")
plt.xticks(np.arange(0, 45e3, 5e3))
plt.show()
```


![png]({{ site.baseurl }}/assets/images/transformacoes-nao-lineares_files/transformacoes-nao-lineares_27_0.png)



```python
regression_transformed = LinearRegression()
regression_transformed.fit(X_transformed, y)

x_min = X_transformed[:, 0].min()
x_max = X_transformed[:, 0].max()
x_values = np.arange(x_min, x_max, 0.01)
y_values = regression_transformed.predict(np.asarray(x_values).reshape(len(x_values), 1))
plt.plot(x_values, y_values, color = "darkgreen")
plt.scatter(X_transformed, y, marker = "s", color = "green", alpha = 0.75)
plt.title("Fitted regression for $\log$ of Salary")
plt.xlabel("Transformed Salary ($\log$ of \$)")
plt.ylabel("Happiness measure")
plt.show()
```


![png]({{ site.baseurl }}/assets/images/transformacoes-nao-lineares_files/transformacoes-nao-lineares_28_0.png)


Perceba que o primeiro modelo ajustado, com a variável $x$ original (indicado pelo gráfico de título `Fitted Regression for Salary`) não representa bem o conjunto de dados (no próximo parágrafo, vamos analisar isso quantitativamente). Ao passo que, quando transformamos $x$, aplicando, nesse caso, a função $\log(\cdot)$, obtemos uma reta (visualmente) melhor ajustada.

Uma medida que podemos utilizar para quantificar a intuição de "qual modelo se ajusta melhor aos dados" é a quantidade $R^2$ (R-squared); que, a grosso modo, diz o quanto da variação de $y$ é explicada pela regressão. Vamos aproveitar a função `score()` implementada na classe `LinearRegression` para obter essa medida.


```python
r2_original = regression_original.score(X, y)
print("R-squared for the original model: {}.".format(r2_original))
print("Intercept: {} & Coefficients: {}.".format(regression_original.intercept_[0], regression_original.coef_[0, 0]))
```

    R-squared for the original model: 0.7538771643539155.
    Intercept: 7.975309624979791 & Coefficients: 9.828555226466561e-05.



```python
r2_transformed = regression_transformed.score(X_transformed, y)
print("R-squared for the 'tranformed' model: {}.".format(r2_transformed))
print("Intercept: {} & Coefficients: {}.".format(regression_transformed.intercept_[0], regression_transformed.coef_[0, 0]))
```

    R-squared for the 'tranformed' model: 0.9431020022916391.
    Intercept: 0.12929691342341698 & Coefficients: 0.9867610189477893.


Mais uma vez, o primeiro resultado diz respeito ao modelo que considera $x$, enquanto que o segundo, ao modelo que considera $\log(x)$ como variável independente. Perceba que, quando aplicamos a tranformação no nosso preditor "salário", obtemos um $R^2$ maior $-$ que é, **quase** sempre, melhor.

Um ponto importante é como devemos interpretar, quando olhamos para o segundo modelo, o coeficiente que obtemos; aqui, $w_1 \approx 0.987$. Nesse caso, podemos dizer que $1\%$ de aumento do salário, resulta em aumento de, aproximadamente, $\frac{0.987}{100} = 0.00987$ "*unidades* de felicidade".

## Conclusão

No começo do texto vimos que, para aplicar os modelos lineares que estudamos até agora (seja de classificação, seja de regressão), basta que, olhando para a expressão $\sum_{i = 0}^{d} w_i x_i$, os $w_i$'s sejam lineares; ou seja, podemos aplicar transformações não-lineares no conjunto de dados $X$. Nesse sentido, vimos dois exemplos (implementados em Python), nos quais transformações do tipo $(\cdot)^2$ e $\log(\cdot)$ foram aplicadas. No próximo post vamos falar um pouco mais sobre medidas de erro. 

Qualquer dúvida, sugestão ou *feedback*, por favor, deixe um comentário abaixo.

> Essa postagem faz parte de uma [série]({{ site.baseurl }}/categorias/#machine-learning-learning-from-data) de textos que tem o objetivo de estudar, principalmente, o curso "[Learning from Data](http://www.work.caltech.edu/telecourse.html)", ministrado pelo Professor Yaser Abu-Mostafa. Outros materiais utilizados serão sempre referenciados.