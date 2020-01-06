---
title: "[Parte 13] Implementação em Python: Redes Neurais"
categories: [Aprendizado de Máquina - Learning from Data]
tags: [python, aprendizado-de-maquina]
---


Como continuação direta da [parte 12](/redes-neurais/) [dessa série de textos]({{ site.baseurl }}/categories/aprendizado-de-máquina-learning-from-data/), vamos, ao longo desse post, implementar uma rede neural utilizando Python. E, para testá-la, vamos resolver um problema de classificação binária de dados que **não** são linearmente separáveis $-$ aqui, não faremos como na [parte 06](/transformacoes-nao-lineares/), onde utilizamos transformações explícitas sobre o conjunto de dados. Hoje, as transformações serão resultado do processo de aprendizado da rede.

Diferente dos outros textos nos quais implementamos algoritmos de *machine learning* $-$ e.g., Perceptron, Regressão Linear, Regressão Logística, etc., nessa postagem vamos adotar uma ordem diferente para as coisas. Vamos começar com o conjunto de dados com o qual vamos trabalhar.

Nesse caso, ao invés de utilizarmos algum banco de dados conhecido, ou mesmo de simularmos algo que nos daria a característica desejada, vamos utilizar um *data set* que vem da biblioteca Sklearn. Perceba que **não** iremos utilizar qualquer implementação de algoritmo pela biblioteca; aqui, **apenas** o banco de dados será considerado. Importando as bibliotecas que iremos utilizar e plotando o gráfico dos dados que obtivemos, temos o seguinte:


```python
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons

np.random.seed(1)

X, y = make_moons(n_samples = 100, noise = 0.15) # generate non-linear separable data
y = np.where(y == 0, -1, 1)

plt.scatter(X[y == -1, 0], X[y == -1, 1], color = "red", marker = "o", label = "Class A")
plt.scatter(X[y ==  1, 0], X[y ==  1, 1], color = "green", marker = "s", label = "Class B")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend(loc = 1)
plt.show()
```


![png]({{ site.baseurl }}/assets/images/implementacao-redes-neurais_files/implementacao-redes-neurais_3_0.png)


Aqui, o banco `make_moons` gera dados, como o nome sugere, que tem comportamento gráfico parecido com duas meias-luas. O parâmetro `noise` especifica o ruído sobre esse padrão. A única observação é que, ao invés de usarmos a classificação de `0` ou `1`, a `Class A` ficou rotulada como `-1` e a `Class B`, como `+1`. 

Agora vamos à nossa classe que implementa o algoritmo: `NeuralNetwork`. O código é grande, então, para conseguirmos estudá-lo adequadamente, vou inserí-lo por completo agora, e, na sequência, comentamos parte por parte.


```python
class NeuralNetwork:
    """
    Neural Network Model
    """
    def __init__(self, dim, eta = 1e-1, epoch = 1000):
        self.dim = dim # vector with layers dimensions - including input (0) and output (L)
        self.L = len(dim) - 1 # 'L' equals to the number of layers, which does not count input (0)
        self.eta = eta
        self.epoch = epoch
    
    
    def initialize_(self):
        rn = np.random.RandomState(1)
        self.w_ = {} # it will include 'bias'
        
        for l in np.arange(1, self.L + 1):
            self.w_['W' + str(l)] = rn.normal(0, 1e-1, (self.dim[l - 1] + 1) * self.dim[l]).reshape(self.dim[l - 1] + 1, self.dim[l])
    
    
    def theta_(self, s):
        return (np.exp(s) - np.exp(-1 * s)) / (np.exp(s) + np.exp(-1 * s))
    
    
    def forward_propagation_(self, X_n):
        X_n = np.r_[1, X_n]
        self.s_ = {}
        self.x_ = {}
        self.x_['x0'] = X_n
        
        for l in np.arange(1, self.L + 1):
            self.s_['s' + str(l)] = np.dot(self.w_['W' + str(l)].T, self.x_['x' + str(l - 1)])
            self.x_['x' + str(l)] = np.r_[1, self.theta_(self.s_['s' + str(l)])]
            
        self.x_['x' + str(self.L)] = self.x_['x' + str(self.L)][1:] # do not use extra '1' at the last layer
        return self.x_['x' + str(self.L)]
    
    
    def back_propagation_(self, x_n, y_n):
        self.d_ = {}
        self.d_['d' + str(self.L)] = 2 * (self.x_['x' + str(self.L)] - y_n) * (1 - (self.x_['x' + str(self.L)]) ** 2)
        
        for l in np.arange(self.L - 1, 0, -1): # reverse order iteration, from 'L - 1' to 1 (inclusive)
            t_prime = 1 - np.multiply(self.x_['x' + str(l)], self.x_['x' + str(l)])[1:]
            self.d_['d' + str(l)] = np.multiply(t_prime, (np.dot(self.w_['W' + str(l + 1)], self.d_['d' + str(l + 1)]))[1:])          
           
        
    def fit(self, X, y):
        self.initialize_()
        
        for n in np.arange(self.epoch):

            error_in = 0
            self.g_ = {}
            for item in np.arange(1, len(self.w_) + 1): # initialize 'G' matrix with all entries being zero 
                self.g_['G' + str(item)] = 0 * self.w_['W' + str(item)]

            for X_n, y_n in zip(X, y):
                self.forward_propagation_(X_n)
                self.back_propagation_(X_n, y_n)
                error_in += (1 / X.shape[0]) * (self.x_['x' + str(self.L)] - y_n) ** 2

                for l in np.arange(1, self.L + 1):
                    g_X_n = np.dot(self.x_['x' + str(l - 1)].reshape(self.x_['x' + str(l - 1)].shape[0], 1), self.d_['d' + str(l)].reshape(1, self.d_['d' + str(l)].shape[0])) # it uses a trick to reshape lists to perform a dot product
                    self.g_['G' + str(l)] = self.g_['G' + str(l)] + (1 / X.shape[0]) * g_X_n

            for item in np.arange(1, len(self.w_) + 1): # update weights
                self.w_['W' + str(item)] -= self.eta * self.g_['G' + str(item)]
            
            if n % 1000 == 0: print("Epoch n. {:4d}: {:.8f}.".format(n, error_in[0]))
            
        return error_in # return final 'error_in'
            
        
    def predict(self, X):
        prediction = []
        
        for item in X:
            prediction.append(self.forward_propagation_(item))
            
        return np.where(np.asarray(prediction) >= 0, 1, -1)
```

Vamos começar pelo método `__init__()`. Nesse caso, o parâmetro `dim` é uma lista com as dimensões em cada uma das *layers* da nossa rede neural $-$ de $0$ a $L$. `L` é, claramente, o número de camadas (lembre-se de que a contagem, consiredando a camada de entrada, começa do $0$). `eta` (ou $\eta$) é a taxa de aprendizagem; e, por fim, `epoch` é o número de iterações que vamos considerar para o nosso processo de otimização do termo $E_{in}$.

O método `initialize_()` inicializa o vetor de pesos adequadamente $-$ aqui, é interessante a estratégia que escolhemos tomar para indexar cada uma das *layes* (a inspiração veio [desse](https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/) artigo). A função `theta()` é o nosso nó de transformação não-linear, definido por $\tanh(\cdot)$.

O método `forward_propagation_()`, bem como `back_propagation_()` implementam **diretamente** o algoritmo apresentado na [parte 12](/redes-neurais/) $-$ a primeira função calcula o resultado $$h(\mathbf{x})$$ para cada par $$(\mathbf{x}_n, y_n)$$, enquanto que a segunda determina o vetor $$\mathbf{\delta}^{(l)}$$ (utilizado para, através do cálculo de $$\nabla E_{in}$$, modificação do vetor de pesos).

Agora, perceba que o método `fit()` foi implementada **sem** utilizar a técnica de *Stochastic Gradient Descent* (SGD). Ou seja, o vetor gradiente $\nabla E_{in}$ considera **todos** os pontos em $\mathcal{D}$ antes de cada atualização de $\mathbf{w}$. Isso foi feito intencialmente já que, logo abaixo, faremos a implementação desse mesmo método utilizando o SGD para conseguirmos estabelecer a diferença de tempo computacional requerida por cada uma dessas alternativas.

Por fim, o método `predict()`, como já temos feito há algum tempo, determina $h(\mathbf{x})$ para cada nova observação. Dito isso, vamos utilizar a classe que acabamos de criar:


```python
%%time

my_nn = NeuralNetwork(dim = [X.shape[1], 16, 16, 16, 1], eta = 1e-2, epoch = 10000)
error_in = my_nn.fit(X, y)
print("Epoch n. {:4d}: {:.8f}.".format(my_nn.epoch, error_in[0]))
```

    Epoch n.    0: 1.01090254.
    Epoch n. 1000: 0.34815072.
    Epoch n. 2000: 0.34731439.
    Epoch n. 3000: 0.34721020.
    Epoch n. 4000: 0.34710094.
    Epoch n. 5000: 0.34697082.
    Epoch n. 6000: 0.34678815.
    Epoch n. 7000: 0.34647321.
    Epoch n. 8000: 0.34577426.
    Epoch n. 9000: 0.34355829.
    Epoch n. 10000: 0.32932046.
    CPU times: user 5min 29s, sys: 218 ms, total: 5min 30s
    Wall time: 5min 30s


Antes de qualquer outra coisa, percebe que utilizamos o comando `%%time`, que, no Jupyter Notebook, serve para medir o tempo que uma célula demora para ser executada. Nesse caso, note que, para ajustar um modelo de $3$ hidden layers com $16$ nós em cada uma delas, para um total de dez mil iterações, gastamos um tempo de 5 minutos e 30 segundos. Lembre-se de que ainda **não** estamos utilizando o *Stochastic Gradient Descent*. Uma outra observação importante vem do fato de que, por quase nove mil iterações, o termo $E_{in}$ praticamente não foi diminuído, nos levando a ter que estender a simulação por muito mais tempo (o problema dos mínimos locais, discutido na [parte 12](/redes-neurais/) pode ter sido uma das razões desse fenômeno).

Ajustado o modelo, podemos, utilizando mais uma vez a função `plot_decision_regions`, determinar as regiões de classificação para o nosso problema.


```python
def plot_decision_regions(X, y, classifier, feature_names, resolution = 1e-2, plot_lim = 0.25):
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
plot_decision_regions(X, y, my_nn, feature_names, plot_lim = 0.15)
plt.title("Fitted Model with Implementation of Neural Network")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend(loc = 2)
plt.show()
```


![png]({{ site.baseurl }}/assets/images/implementacao-redes-neurais_files/implementacao-redes-neurais_16_0.png)


Do gráfico acima, perceba que as transformações não-lineares aprendidas pela rede **não** foram suficientes para captar corretamente o comportamento dos dados. Mais uma vez, isso se deve ao fato de que o algoritmo **não** foi capaz de minimar, a níveis suficientemente bons, o termo $E_{in}$ $-$ o que é, obviamente, um grande problema.

Para contornar esse obstáculo, vamos sobrescrever o método `fit()` utilizando, como novidade, a técnica SGD. Para isso, basta utilizar o conceito de "herança" e criar uma classe `NeuralNetworkSGD` que herda os atributos e métodos de `NeuralNetwork`.


```python
class NeuralNetworkSGD(NeuralNetwork):
    """
    Neural Network Model using Stochastic Gradient Descent (S.G.D.)
    """        
    def fit(self, X, y):
        self.initialize_()
        
        for n in np.arange(1, self.epoch + 1):
        
        
            rand_number = np.random.choice(np.arange(X.shape[0]))
        
            X_n = X[rand_number]
            y_n = y[rand_number] 

            self.forward_propagation_(X_n)
            self.back_propagation_(X_n, y_n)
            
            for item in np.arange(1, len(self.w_) + 1): # update weights
                self.w_['W' + str(item)] -= self.eta * np.dot(self.x_['x' + str(item - 1)].reshape(self.x_['x' + str(item - 1)].shape[0], 1), self.d_['d' + str(item)].reshape(1, self.d_['d' + str(item)].shape[0]))
              
        error_in = 0
        
        for X_n, y_n in zip (X, y):            
            error_in += (1 / X.shape[0]) * ((self.forward_propagation_(X_n) - y_n) ** 2)

        return error_in
```

Agora, como dito antes, somos capazes de ajustar o vetor de pesos $\mathbf{w}$ para cada rodada de iteração que considera **somente** um par $(\mathbf{x}_n, y_n)$, uniformemente escolhido em $\mathcal{D}$. Ajustando o modelo:


```python
%%time

my_nn_mod = NeuralNetworkSGD(dim = [X.shape[1], 256, 256, 256, 1], eta = 1e-2, epoch = 10000)
error_in  = my_nn_mod.fit(X, y)
print("Epoch n. {:4d}: {:.8f}.".format(my_nn_mod.epoch, error_in[0]))
```

    Epoch n. 10000: 0.01763535.
    CPU times: user 50.7 s, sys: 30.5 s, total: 1min 21s
    Wall time: 21.8 s


Note que, primeiro, mesmo considerando uma rede razoavelmente maior $-$ $3$ hidden layers, só que dessa vez com $256$ nós em cada uma delas (para as mesmas dez mil iterações) $-$ o tempo de execução foi bem menor: 1 minuto e 21 segundos. Além disso o termo $E_{in}$ foi minimizado até $\approx 0.0178$. Agora, vamos plotar o gráfico das regiões de decisão para essa segunda simulação.


```python
feature_names = ['Classe A', 'Classe B']
plot_decision_regions(X, y, my_nn_mod, feature_names, plot_lim = 0.15)
plt.title("Fitted Model with Implementation of N.N. using S.G.D.")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend(loc = 2)
plt.show()
```


![png]({{ site.baseurl }}/assets/images/implementacao-redes-neurais_files/implementacao-redes-neurais_23_0.png)


Note que que as regiões de decisão desenhadas representam muito melhor os dados do que o primeiro caso. O que nos leva a pensar que, de fato, a técnica de *Stochastic Gradient Descent* gera bons resultados. Há um problema, entretanto; como conversamos na [parte 10](/vies-variancia-tradeoff/), modelos mais complexos (ou seja, modelos que se adaptam melhor ao conjunto de dados fornecido) podem sofrer grande variação à medida que variamos $\mathcal{D}$ $-$ nesse caso, podemos sofrer de *overfitting*, situação na qual o modelo perde seu poder de generalização para dados fora de $\mathcal{D}$.

## Conclusão

Seguindo à risca a teoria que desenvolvemos na [parte 12](/redes-neurais/), foi relativamente fácil implementarmos um rede neural que aceita parâmetros bastante razoáveis sobre sua arquitetura. Vimos que, a depender do conjunto de dados com o qual estamos trabalhando, as transformações não-lineares aprendidas pela rede podem não ser boas o suficiente para descrevermos $\mathcal{D}$ $-$ isso acontece, dentre outros motivos, pelo fato de que minimizar $E_{in}$ pode ser complicado à medida que encontramos mínimios locais que não são globais. Porém, como também pudemos testar, esse problema pode ser mitigado empregando a técnica de *Stochastic Gradient Descent* $-$ que, no nosso caso, gerou bons resultados. 





Qualquer dúvida, sugestão ou *feedback*, por favor, deixe um comentário abaixo.

> Essa postagem faz parte de uma [série]({{ site.baseurl }}/categories/aprendizado-de-máquina-learning-from-data/) de textos que tem o objetivo de estudar, principalmente, o curso "[Learning from Data](http://www.work.caltech.edu/telecourse.html)", ministrado pelo Professor Yaser Abu-Mostafa. Outros materiais utilizados serão sempre referenciados.