---
title: "[Parte 02] Implementação em Python: Perceptron"
categories: [Aprendizado de Máquina - Learning from Data]
tags: [python, aprendizado-de-maquina]
---

Como mencionado na [parte 01](/o-que-e-aprendizado/) [dessa série de textos](/categories/aprendizado-de-máquina-learning-from-data/), essa postagem será dedicada à implementação do *Perceptron Learning Algorithm* (PLA) em Python, utilizando, para isso, a biblioteca Numpy. Discutiremos um exemplo e, ao final, vamos ver como utilizar a implementação desse mesmo algoritmo pela biblioteca Sklearn.

O código a seguir é baseado (feitas algumas modificações) nos capítulos iniciais do que é apresentado no livro [Python Machine Learning](https://sebastianraschka.com/books.html).

Primeiro vamos à implementação do algoritmo, e depois discutiremos os trechos do que foi escrito. Considere a classe `Perceptron` a seguir:


```python
import numpy as np

# Raw implementation
class Perceptron():
    """
    Perceptron learning algorithm implementation
    """
    
    def __init__(self, eta = 1, random_seed = 1):
        self.eta = eta
        self.random_seed = random_seed
        
    def fit(self, X, y):
        rn = np.random.RandomState(self.random_seed)
        self.w_ = rn.normal(loc = 0, scale = 0.01, size = X.shape[1] + 1)
        errors = True
        
        while errors:
            errors = 0
            for X_i, y_i in zip(X, y):
                if(y_i != self.predict(X_i)):
                    # update weights vector for misclassified points
                    update = self.eta * y_i
                    self.w_[1:] += update * X_i
                    self.w_[0] += update 
                    errors += 1
                    
        return self
    
    def predict(self, X_i):
        eval_func = np.dot(X_i, self.w_[1:]) + self.w_[0] # ATTENTION: this implemented arguments order is more intuitive
        return np.where(eval_func >= 0, 1, -1)
```

Note que, para criar uma instância da classe, são necessários dois parâmetros: o primeiro deles, `eta`, diz respeito a uma pequena generalização feita na regra de atualização do vetor $\mathbf{w}$ (discutida no próximo parágrafo), enquanto que o parâmetro `random_seed` define uma semente para geração aleatória do vetor inicial de pesos.

Como acabei de dizer, podemos generalizar a regra de atualização de $\mathbf{w}$ apresentada na parte 01 dessa série introduzindo o parâmetro $\eta$. A regra, então, ficaria assim:

$$
\begin{align*}
    \mathbf{w}(t+1) = \mathbf{w}(t) + \eta \, y(t) \, \mathbf{x}(t)
\end{align*}
$$

Perceba que para $\eta = 1$, a regra é exatamente a mesma que vimos antes. Dessa forma, a única coisa que $\eta$ faz é mexer no quanto a reta definida por $\mathbf{w}$ "se move" para classificar corretamente o ponto considerado. Nesse sentido, perceba que, para $\eta$ grande, a chance de eu "bagunçar" a classificação dos demais pontos também é grande; portanto, $\eta$ maior não é necessariamente melhor (normalmente, $0 < \eta \leq 1$). 

Continuando, o método `__init__()` apenas inicializa os atributos `eta` e `random_seed`. O método `fit()` é o que, de fato, ajusta o modelo; ou seja, ajusta os valores do vetor $\mathbf{w}$. Nesse caso, para os pontos que não estão classificados corretamente (`y_i != self.predict(X_i)`), o atributo `w_` é atualizado de acordo com a regra que acabamos de discutir $-$ esse processo é realizado quantas vezes forem necessárias. Uma tecnicalidade importante é a de que o vetor de pesos não deve ser inicializado com todas as entradas nulas; caso isso aconteça, o termo `update * X_i` afetará somente a escala de $\mathbf{w}$. Por fim, o método `predict()`, baseado na função $h(\mathbf{x})$ definida no post anterior, faz a classificação de cada ponto (ou conjunto de pontos) `X_i`. 

Para verificar o funcionamento do algoritmo, vamos utilizar o conjunto de dados [Iris](https://archive.ics.uci.edu/ml/datasets/iris), que classifica três espécies de flores de acordo com o comprimento e largura de suas pétalas e sépalas. Veja as primeiras linhas do conjunto de dados:


```python
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt

# Iris Data Set
df = pd.read_csv("https://bit.ly/2Mg0qkZ", header = None, encoding = "UTF")

df.columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'category']
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal-length</th>
      <th>sepal-width</th>
      <th>petal-length</th>
      <th>petal-width</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



Entretanto, para que consigamos visualizar facilmente o resultado do algoritmo, vamos nos concentrar em duas *features*: "Sepal length" e "Petal length". Além disso, vamos denotar a espécie "Iris Setosa" (50 primeiras linhas) por $-1$ e a espécie "Iris Versicolor" (linhas 50 a 100) por $+1$. Nesse caso, não utilizaremos as informações da terceira espécie; já que o Perceptron é um classificador **binário**. Veja como ficaram os dados:


```python
df = df.loc[0:100, ['sepal-length', 'petal-length', 'category']]

X = df.loc[0:100, ['sepal-length', 'petal-length']].values
y = df.category.values
y = np.where(y == 'Iris-setosa', -1, 1)

plt.scatter(X[0:50, 0], X[0:50, 1], color = "red", marker = "o", label = "Iris-ventosa")
plt.scatter(X[50:100, 0], X[50:100, 1], color = "green", marker = "s", label = "Iris-versicolor")
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc = 2)
plt.show()
```


![png]({{ site.baseurl }}/assets/images/implementacao_perceptron_files/perceptron_10_0.png)


Agora podemos ajustar o modelo:


```python
my_perceptron = Perceptron()
my_perceptron.fit(X, y)

print("Weights vector: {}.".format(my_perceptron.w_))
```

    Weights vector: [-1.98375655 -3.50611756  9.19471828].


Nesse caso, instanciamos um objeto da classe `Perceptron` e depois ajustamos o modelo baseado no conjunto de dados que acabamos de filtrar. Note, então, que o vetor `w_` foi completamente determinado. Assim, o que podemos fazer é visualizar os resultados de forma gráfica. Para isso, utiizaremos a função a seguir, `plot_decision_regions()`:


```python
def plot_decision_regions(X, y, classifier, feature_names, resolution = 0.01):
    # general settings
    markers = ["o", "s", "*", "x", "v"]
    colors  = ("red", "green", "blue", "gray", "cyan")
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # define a grid
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # classify each grid point
    result = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    result = result.reshape(xx1.shape)
    # make a plot
    plt.contourf(xx1, xx2, result, colors = colors[0:len(np.unique(y))], alpha = 0.5)
    for index, value in enumerate(np.unique(y)): # plot each point & 'enumerate()' returns index and value of the given array
        plt.scatter(x = X[y == value, 0], y = X[y == value, 1], # select each X and y vectors by creating a mask
                    color = colors[index],
                    marker = markers[index],
                    label = feature_names[index],
                    edgecolor = 'black')
```

Agora podemos plotar o gráfico de interesse:


```python
feature_names = ['Iris ventosa', 'Iris versicolor']
plot_decision_regions(X, y, my_perceptron, feature_names)
plt.title('Fitted model with raw implementation')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc = 2)
plt.show()
```


![png]({{ site.baseurl }}/assets/images/implementacao_perceptron_files/perceptron_16_0.png)


Como podemos ver, o algoritmo classificou corretamente todas as flores. Entretanto, um ponto muito importante sobre o qual ainda não disctutimos, é a eficiência (ou acurácia) desse modelo para classificar novas observações. Esse tipo de medida será discutida (e implementada) ao longo dos próximos textos.

Antes de finalizar, vamos ver como ajustar esse mesmo modelo utilizando a implementação do Sklearn:


```python
# Sklearn usage
from sklearn.linear_model import Perceptron

sklearn_perceptron = Perceptron()
sklearn_perceptron.fit(X, y)

print("Weight vector (without w_0): {}.".format(sklearn_perceptron.coef_))
```

    Weight vector (without w_0): [[-2.4  5. ]].


Como a classe Perceptron já está implementada na biblioteca, ajustar o modelo é super simples; basta, mais uma vez, utilizar o método `fit()`. O vetor de pesos é ligeiramente diferente, mas também é uma solução para o nosso problema $-$ veja o gráfico abaixo:


```python
feature_names = ['Iris ventosa', 'Iris versicolor']
plot_decision_regions(X, y, sklearn_perceptron, feature_names)
plt.title('Fiited model with Sklearn usage')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc = 2)
plt.show()
```


![png]({{ site.baseurl }}/assets/images/implementacao_perceptron_files/perceptron_21_0.png)


## Conclusão

Nessa postagem vimos a implementação em Python do *Perceptron Learning Algorithm* (PLA) que havíamos discutido antes. Entretanto, a maior parte desses resultados em *machine learning* já estão disponíveis através da biblioteca Sklearn (veja a [documentação](https://scikit-learn.org/stable/)) $-$ o que não nos impede, como forma de estudo, de escrever as nossas próprias versões dos algoritmos. Por fim, como foi brevemente mencionado, é importante que consigamos avaliar a efeciência dos nossos modelos; esse tópico começará a ser abordado no próximo post.

Qualquer dúvida, sugestão ou *feedback*, por favor, deixe um comentário abaixo.

> Essa postagem faz parte de uma [série]({{ site.baseurl }}/categories/aprendizado-de-máquina-learning-from-data/) de textos que tem o objetivo de estudar, principalmente, o curso "[Learning from Data](http://www.work.caltech.edu/telecourse.html)", ministrado pelo Professor Yaser Abu-Mostafa. Outros materiais utilizados serão sempre referenciados.