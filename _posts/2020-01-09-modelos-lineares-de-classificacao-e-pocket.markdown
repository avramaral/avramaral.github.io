---
title: "[Parte 04] Modelos Lineares de Classificação & Implementação em Python do Pocket"
categories: [Aprendizado de Máquina - Learning from Data]
tags: [python, aprendizado-de-maquina]
---

Na [parte 03](/memorizar-nao-e-aprender/) [dessa série de textos]({{ site.baseurl }}/categories/aprendizado-de-máquina-learning-from-data/), discutimos se é possível que nossos algoritmos, de fato, aprendam; ou seja, se conseguimos determinar uma função $g \in \mathcal{H}$ que aproxima bem $f$ para pontos $\mathbf{x} \not\in \mathcal{D}$. Esse assunto ainda não foi esgotado, mas por hora vamos nos concentrar em estudar mais alguns algoritmos $-$ em especial, alguns modelos lineares.

Esse post vai ser um pouco diferente, no sentido de mesclar a apresentação teórica com a implementação prática (em Python) de um dos algoritmos de interesse: o *Pocket Learning Algorithm* (ou só Pocket) $-$ uma variação do Perceptron.

Fazendo referência ao PLA (*Perceptron Learning Algorithm*), que estudamos nas partes [01](/o-que-e-aprendizado/) e [02](/implementacao-perceptron/), uma de suas limitações era a de que os dados precisavam ser linearmente separáveis; caso contrário, o algoritmo não era capaz de classificar corretamente todos os pontos $\mathbf{x} \in \mathcal{D}$, e, por consequência, não conseguir convergir. Perceba que essa é uma suposição bem forte, já que, na prática, os dados quase nunca tem essa característica. Uma solução para contornar esse problema seria a de limitar o número de iterações que o algoritmo poderia realizar antes de ser interrompido. Porém, dessa solução, surge um problema.

Lembre-se de que o Perceptron, na tentativa de ajustar o hiperplano definido por $$\mathbf{w}$$ que classifica corretamente um ponto $$\mathbf{x}_i$$, poderia "bagunçar" a classificação associada aos demais pontos. Em outras palavras, mais iterações **não** se traduzem em uma reta (para o caso de $$2$$ dimensões) "melhor" (no sentido de ter $$E_{in}(h)$$ menor). Dito isso, uma ideia para tratar esse problema seria a de, a cada etapa do processo, verificar o erro *in-sample* e, nas situações nas quais ele for o menor, tomar o vetor $$\mathbf{w}$$ associado como o "escolhido". Isso é exatamente o que o *Pocket Learning Algorithm* faz, veja o código a seguir: 


```python
# Raw implementation
class Pocket:
    """
    Pocket learning algorithm implementation
    """
    
    def __init__(self, eta = 1, random_seed = 1, n_iterations = 100):
        self.eta = eta
        self.random_seed = random_seed
        self.n_iterations = n_iterations 
    
    def fit(self, X, y):
        rn = np.random.RandomState(self.random_seed)
        self.w_ = rn.normal(loc = 0, scale = 0.01, size = X.shape[1] + 1)
        counter = 0
        errors = True
        
        partial_w = self.w_.copy()
        partial_error_in_sample = 0
        self.error_in_sample_ = 0 
        
        while errors and (counter < self.n_iterations):
            errors = 0
            error_freq = 0
            for X_i, y_i in zip(X, y):
                if y_i != self.predict(X_i):
                    # update weights for misclassified points
                    update = self.eta * y_i
                    self.w_[1:] += update * X_i
                    self.w_[0] += update
                    errors += 1    
            for X_i, y_i in zip(X, y):
                if y_i != self.predict(X_i):
                    # count misclassified points AFTER analyze all of them 
                    error_freq += 1
                partial_error_in_sample = (1 / X.shape[0]) * error_freq
            if (counter == 0) or (partial_error_in_sample < self.error_in_sample_):
                # update smallest error and best weights vector
                self.error_in_sample_ = partial_error_in_sample  
                partial_w = self.w_.copy()
            counter += 1
            
        self.w_ = partial_w.copy()
        return self
    
    def predict(self, X_i):
        eval_func = np.dot(X_i, self.w_[1:]) + self.w_[0]
        return np.where(eval_func >= 0, 1, -1)
```

A classe `Pocket` é muito similar à classe `Perceptron` que havíamos criado anteriormente; o que faz sentido, já que o Pocket é uma modificação do Perceptron. A principal diferença está no método `fit()`. Perceba que agora o número de iterações não é definido apenas pela quantidade de vezes que o vetor $\mathbf{w}$ tem que ser atualizado para que todos os pontos sejam corretamente classificados $-$ na verdade isso pode nem acontecer, já que, como dito, a suposição de que os dados são linearmente separáveis não é mais necessária. O atributo `n_iterations` cuida desse limite máximo.

Note também que, ao final de cada ciclo em que o algoritmo percorre todos os pontos $\mathbf{x} \in \mathcal{D}$, o erro $E_{in}(h)$ (erro *in-sample*) é calculado e armazenado na variável `partial_error_in_sample`. Depois disso, no caso de ele ser o menor erro encontrado até o momento, essa quantidade é salva no atributo `error_in_sample` $-$ bem como o vetor $\mathbf{w}$, que é armazenado em `partial_w`. Ao final, o vetor de pesos escolhido é aquele que teve o menor erro associado; ou seja, o vetor salvo em `partial_w`, que é então tranferido para o atributo `w_`.

Vamos ver agora como isso funciona em um conjunto de dados que **não** é linearmente separável. Considere o seguinte cenário: suponha que você quer classificar corretamente digitos númericos escritos a mão, como os mostrados na figura abaixo. Cada uma das imagens é composta por uma grade de $16 \times 16$ pixels que assume valores entre $0$ e $255$; ou seja, teríamos um espaço Euclidiano $256$-dimensional de funções (possivelmente) real-avaliadas.

![Digitos escritos a mão]({{ site.baseurl }}/assets/images/modelos-lineares-de-classificacao-e-pocket_files/digitos-escritos-a-mao.png)
*Figura 1 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Dígitos numéricos escritos a mão.*

A fim de diminuir essa quantidade de *features* (ou características, ou variáveis independentes, etc.), podemos considerar, apenas, alguma medida de intensidade e alguma medida de simetria dos pixels. Obviamente não estamos utilizando todas as informações, mas isso já deve ser o suficiente para termos bons resultados. No código a seguir, ajustaremos um modelo para classificar os dígitos $1$ e $5$ (lembre-se que o Pocket ainda é um classificador binário); veja:


```python
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("http://www.amlbook.com/data/zip/features.train", header = None, delimiter = r"\s+")
df.columns = ['number', 'intensity', 'symmetry']
df['number'] = df['number'].astype(int) 
df.head(3)
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
      <th>number</th>
      <th>intensity</th>
      <th>symmetry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>0.341092</td>
      <td>-4.528937</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>0.444131</td>
      <td>-5.496812</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>0.231002</td>
      <td>-2.886750</td>
    </tr>
  </tbody>
</table>
</div>




```python
mask = ((df.number == 1) | (df.number == 5)) # select only numbers '1' and '5'
df = df[mask]
df = df.reset_index(drop = True)
df.head(3)
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
      <th>number</th>
      <th>intensity</th>
      <th>symmetry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>0.444131</td>
      <td>-5.496812</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.123043</td>
      <td>-0.707875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.113859</td>
      <td>-0.931375</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1561 entries, 0 to 1560
    Data columns (total 3 columns):
    number       1561 non-null int64
    intensity    1561 non-null float64
    symmetry     1561 non-null float64
    dtypes: float64(2), int64(1)
    memory usage: 36.7 KB


Perceba que, depois de filtrar adequadamente o conjunto de dados, temos uma base com 1561 entradas que apresentam características (de intensidade e simetria dos pixels) dos números $1$ e $5$. Vamos, agora, plotar esses dados.


```python
X = df.loc[:, ['intensity', 'symmetry']].values
y = df.loc[:, 'number'].values
y = np.where(y == 1, -1, 1) # maps 1 to -1, and 5 to 1

plt.scatter(X[y == -1, 0], X[y == -1, 1], color = "red", marker = "o", label = "Number 1")
plt.scatter(X[y ==  1, 0], X[y ==  1, 1], color = "green", marker = "s", label = "Number 5")
plt.xlabel("Intensity measure")
plt.ylabel("Symmetry measure")
plt.legend(loc = 1)
plt.show()
```


![png]({{ site.baseurl }}/assets/images/modelos-lineares-de-classificacao-e-pocket_files/modelos-lineares-de-classificacao-e-pocket_14_0.png)


A primeira coisa a se notar é que os dados **não** são linearmente separáveis. Dito isso, vamos, finalmente, ajustar o modelo:


```python
my_pocket = Pocket(n_iterations = 100)
my_pocket.fit(X, y)

print("Weights vector: {}.".format(my_pocket.w_))
print("Smallest error in-sample: {}.".format(my_pocket.error_in_sample_))
```

    Weights vector: [-9.98375655 -1.494057   -4.21659422].
    Smallest error in-sample: 0.0038436899423446506.


Nesse caso, estamos considerando o vetor $\mathbf{w}$ associado ao menor erro $E_{in}(h)$ encontrado: `0.0038436899`. Podemos, então, plotar o gráfico com as regiões de decisão; utilizando, para isso, a função `plot_decision_regions` já discutida na [parte 02](/implementacao-perceptron/).


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
feature_names = ['Number 1', 'Number 5']
plot_decision_regions(X, y, my_pocket, feature_names, plot_lim = 0.15)
plt.title("Fitted model with Pocket Learning Algorithm")
plt.xlabel("Intensity measure")
plt.ylabel("Symmetry measure")
plt.legend(loc = 1)
plt.show()
```


![png]({{ site.baseurl }}/assets/images/modelos-lineares-de-classificacao-e-pocket_files/modelos-lineares-de-classificacao-e-pocket_19_0.png)


Como podemos observar, o Pocket encontrou uma reta que minimiza (considerando as cem primeiras iterações) o erro amostral $-$ o que, como discutido na [parte 03](/memorizar-nao-e-aprender/), é um dos passos necessários para dizermos que o nosso algoritmo aprendeu.

## Conclusão

Vimos ao longo do texto um generalização para o Perceptron, chamada Pocket. Esses dois algoritmos fazem parte de uma classe maior de modelos lineares, e são, portanto, classificadores lineares. No próximo post vamos estudar o modelo de regressão $-$ nesse caso, a função alvo $f: \mathcal{X} \rightarrow \mathcal{Y}$ terá contradominio nos números reais.

Qualquer dúvida, sugestão ou *feedback*, por favor, deixe um comentário abaixo.

> Essa postagem faz parte de uma [série]({{ site.baseurl }}/categories/aprendizado-de-máquina-learning-from-data/) de textos que tem o objetivo de estudar, principalmente, o curso "[Learning from Data](http://www.work.caltech.edu/telecourse.html)", ministrado pelo Professor Yaser Abu-Mostafa. Outros materiais utilizados serão sempre referenciados.