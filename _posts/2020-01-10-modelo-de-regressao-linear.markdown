---
title: "[Parte 05] Modelo de Regressão Linear & Implementação em Python"
categories: [Aprendizado de Máquina - Learning from Data]
tags: [python, aprendizado-de-maquina]
---

Continuando a discussão sobre modelos lineares, assunto que começamos a estudar na [parte 04](/modelos-lineares-de-classificacao-e-pocket/) [dessa série de textos]({{ site.baseurl }}/categories/aprendizado-de-máquina-learning-from-data/), vamos, nesse post, falar do modelo de regressão. A principal diferença entre esse tipo de modelo e os classificadores que estudamos anteriormente é que agora a função alvo $f$ é real-avaliada $-$ que é o mesmo que dizer que $\mathcal{Y} \subset \mathbb{R}$.

Para dar contexto ao problema, considere o seguinte exemplo: você é professor de uma turma de Ensino Médio e, ao longo do ano, aplica três provas de 20 pontos cada. Depois de os alunos realizarem os dois primeiros testes, você deseja construir um modelo que lhe permita predizer a nota dos alunos na terceira prova. Perceba que, nesse caso, não estamos tentando "classificar" a terceira nota $-$ aqui, $y \in [0, 20]$. Para resolver esse tipo de tarefa, faremos uso de um modelo (linear) de regressão. **Observação 1:** inúmeras outras características dos alunos poderiam ser utilizadas como preditores (exemplo: horas de estudo, número de faltas ao longo do ano, etc.), mas para simplicidade do modelo, vamos nos atentar, somente, às notas nas duas primeiras avaliações.

Nesse caso, a classe de funções $h \in \mathcal{H}$ será defina por:

$$
\begin{align*}
h(\mathbf{x}) = \sum_{i = 0}^{d} w_i \, x_i = \mathbf{w}^{\text{T}}\mathbf{x}.
\end{align*}
$$

Perceba que $h$, como acabamos de definir, é muito parececido com a classe funções que utilizamos quando dicutimos o Perceptron; porém, ao invés do sinal $-$ $\text{sign}(\cdot)$ $-$, estamos interessado no valor de $\mathbf{w}^{\text{T}}\mathbf{x}$.

Continuando, da mesma forma que fizemos quando estudamos o modelo Pocket, o que queremos nesse caso é minimizar o erro amostral $-$ $E_{in}(h)$ $-$, associado ao modelo. Nesse sentido, utilizaremos uma medida de erro clássica para análise de regressão: o **erro quadrático**. Defina $E_{out}(h) = \mathbb{E}\left[(h(\mathbf{x}) - y)^2\right]$ $-$ nesse caso, o valor esperado é calculado com respeito à distruição $P(\mathbf{x}, y)$. Porém, como não temos acesso à medida de erro $E_{out}(h)$, como discutimos na [parte 03](/memorizar-nao-e-aprender/), uma alternativa é minimar o erro *in-sample*.

Assim, podemos escrever que $$E_{in}(h) = \frac{1}{N} \sum_{n = 1}^{N}(h(\mathbf{x}_n) - y_n)^2$$. Nesse sentido, nossa missão é encontrar $$\mathbf{w}$$ tal que $$E_{in}(h)$$ é mínimo. Veja:

$$
\begin{align*}
E_{in}(\mathbf{w}) & = \frac{1}{N} \sum_{n = 1}^{N} (\mathbf{w}^{\text{T}}\mathbf{x} - y_n)^2 \\
          & = \frac{1}{N} \mid\mid X \mathbf{w} - \mathbf{y} \mid\mid^2 \text{, onde } \mid\mid \cdot \mid\mid \text{ é a norma Euclidiana} \\
          & = \frac{1}{N} \left(\mathbf{w}^{\text{T}} X^{\text{T}} X \mathbf{w} - 2 \mathbf{w}^{\text{T}} X^{\text{T}}\mathbf{y} + \mathbf{y}^{\text{T}} \mathbf{y} \right).
\end{align*}
$$

Derivando $E_{in}(\mathbf{w})$; ou seja, calculando o vetor gradiente $\nabla E_{in}(h)$, obtemos:

$$
\begin{align*}
\nabla E_{in}(\mathbf{w}) = \frac{2}{N}\left(X^{\text{T}} X \mathbf{w} - X^{\text{T}} y \right).
\end{align*}
$$

Igualando $\nabla E_{in}(\mathbf{w})$ ao vetor $\mathbf{0}$, temos que:

$$
\begin{align*}
X^{\text{T}} X \mathbf{w} = X^{\text{T}} \mathbf{y}.
\end{align*}
$$

Assim, se $X^{\text{T}}X$ for invertível, então $\mathbf{w} = (X^{\text{T}} X)^{-1} X^{\text{T}} \mathbf{y}$; onde $X^{\dagger} = (X^{\text{T}} X)^{-1} X^{\text{T}}$ é conhecida como *pseudo-inversa* de $X$. Aqui, $(X^{\text{T}} X)^{-1}$ existirá sempre que nenhuma coluna de $X$ for combinação linear das outras; na prática, se $N$ for muito maior que $d + 1$, então isso quase sempre será satisfeito.

Aqui, dois pontos são importantes. **Obervação 2:** note que obtemos uma solução analítica para minimizar $E_{in}(\mathbf{w})$ $-$ para o caso do Pocket, por exemplo, esse não era o caso. **Observação 3:** perceba que $E_{in}(\mathbf{w})$ é função de $\mathbf{w}$, e não de $X$ (que, nessa situação, é constante).

Agora que temos um solução para o problema de minimazação do erro, podemos implementar o algoritmo. A classe `LinearRegression` cuida disso.


```python
# Raw implementation
class LinearRegression:
    """
    Linear Regression Algorithm
    """
    def __ini__(self):
        pass
    
    def fit(self, X, y):
        Z = np.array([1 for x in np.arange(X.shape[0])])
        X = np.c_[Z, X] # create an array by including each component as a column
        X_dagger = np.linalg.pinv(X) # find the pseudo-inverse matrix
        self.w_  = np.dot(X_dagger, y) 
        return self
    
    def predict(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
```

Perceba que no método `fit()` calculamos a *pseudo-inversa* de $X$ através da função `pinv()` (implementada pelo o módulo de álgebra linear do Numpy). Depois disso, bastou calcularmos o vetor de pesos `w_` definido por $\mathbf{w} =  X^{\dagger} \mathbf{y}$. O método `predict()` apenas implementa a função $h(\mathbf{x})$.

Agora, para ajustarmos o modelo, considere o exemplo das notas de três provas feitas por alunos do Ensino Médio discutido no começo do texto. Veja os dados:


```python
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("student-mat.csv", sep = ";")
df = df.loc[:, ['G1', 'G2', 'G3']]
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
      <th>G1</th>
      <th>G2</th>
      <th>G3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>8</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 395 entries, 0 to 394
    Data columns (total 3 columns):
    G1    395 non-null int64
    G2    395 non-null int64
    G3    395 non-null int64
    dtypes: int64(3)
    memory usage: 9.4 KB


Perceba que temos $395$ entradas não nulas para cada uma das três provas. Nesse caso, `G1` e `G2` serão nossos preditores, e `G3` será fará o papel da variável dependente. Vamos, então, visualizar como os dados se comportam:


```python
X = df.loc[:, ['G1', 'G2']].values
y = df.loc[:, 'G3'].values

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize = (6, 6))
ax  = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:, 0], X[:, 1], y, color = "red", marker = "o")

ax.set_xlim3d(0, 20)
ax.set_ylim3d(0, 20)
ax.set_zlim3d(0, 20)

ticks = np.arange(0, 24, 4)
ax.xaxis.set(ticks = ticks)
ax.yaxis.set(ticks = ticks)
ax.zaxis.set(ticks = ticks)

ax.set_xlabel("G1")
ax.set_ylabel("G2")
ax.set_zlabel("G3")

ax.view_init(30, 150)

plt.tight_layout()
plt.show()
```


![png]({{ site.baseurl }}/assets/images/modelo-de-regressao-linear_files/modelo-de-regressao-linear_17_0.png)


Note que, ao menos a primeira vista, os dados da amostra aparentam ter comportamento linear; o que nos permite prosseguir com o ajuste do modelo:


```python
my_regression = LinearRegression()
my_regression.fit(X, y)
print("Weights vector: {}.".format(my_regression.w_))
```

    Weights vector: [-1.83001214  0.15326859  0.98686684].


Veja que o vetor de pesos, nesse caso com $3$ coordenadas ($w_0$, $w_1$ e $w_2$, respectivamente) foi facilmente determinado. Para conseguirmos visualizar o plano definido por esse vetor, podemos plotar o seguinte gráfico:


```python
g1 = np.arange(0, 20, 0.1)
g2 = np.arange(0, 20, 0.1)
g1, g2 = np.meshgrid(g1, g2)
g3 = my_regression.predict(np.c_[np.ravel(g1), np.ravel(g2)])
g3 = g3.reshape(g1.shape)

fig = plt.figure(figsize = (6, 6))
ax  = fig.add_subplot(111, projection = '3d')
ax.plot_surface(g1, g2, g3, color = "red", alpha = 0.5)
ax.scatter(X[:, 0], X[:, 1], y, color = "red", marker = "o", edgecolor = "black")

ax.set_xlim3d(0, 20)
ax.set_ylim3d(0, 20)
ax.set_zlim3d(0, 20)

ticks = np.arange(0, 24, 4)
ax.xaxis.set(ticks = ticks)
ax.yaxis.set(ticks = ticks)
ax.zaxis.set(ticks = ticks)

ax.set_title("Fitted model with raw implementation", pad = 15)
ax.set_xlabel("G1")
ax.set_ylabel("G2")
ax.set_zlabel("G3")

ax.view_init(30, 150)
# ax.invert_xaxis()

plt.tight_layout()
plt.show()
```


![png]({{ site.baseurl }}/assets/images/modelo-de-regressao-linear_files/modelo-de-regressao-linear_21_0.png)


Nesse caso, o plano parece representar bem o conjunto de pontos. Entretanto, ainda não temos uma análise quantitativa desse tipo de medida $-$ como dito em posts anteriores, ainda vamos chegar lá: na discussão da acurácia dos modelos que escrevemos.

Um pequeno teste que podemos fazer é o de predizer, de acordo com o modelo ajustado, qual nota dois alunos arbitrários tirariam na prova `G3` (com base em suas notas `G1` e `G2`).


```python
print("O aluno A teve notas G1 = 6.5 e G2 = 17. Assim, espera-se que ele tire {:.2f} pontos na última prova.".format(my_regression.predict(np.array([6.5, 17]))))
print("\n")
print("O aluno B teve notas G1 = 17 e G2 = 6.5. Assim, espera-se que ele tire {:.2f} pontos na última prova.".format(my_regression.predict(np.array([17, 6.5]))))
```

    O aluno A teve notas G1 = 6.5 e G2 = 17. Assim, espera-se que ele tire 15.94 pontos na última prova.
    
    
    O aluno B teve notas G1 = 17 e G2 = 6.5. Assim, espera-se que ele tire 7.19 pontos na última prova.


Uma observação interessante nesse caso é a de que a nota `G3` é mais afetada pela nota `G2` do que por `G1` $-$ o que faz total sentido, já que, como vimos, $w_2 > w_1$.

Por fim, vamos utilizar a implementação do Sklearn para o modelo de regressão linear. Confira o código a seguir:


```python
# Sklearn usage
from sklearn.linear_model import LinearRegression

sklearn_regression = LinearRegression()
sklearn_regression.fit(X, y)

print("Weights vector - intercept: {} & coefficients: {}.".format(sklearn_regression.intercept_, sklearn_regression.coef_))
```

    Weights vector - intercept: -1.8300121405807381 & coefficients: [0.15326859 0.98686684].


Nesse caso, como obtivemos uma solução analítica para $\mathbf{w}$, os resultados são exatamente iguais aos que encontramos.

## Conclusão

Nesse texto discutimos o importante modelo de regressão linear, que nos permite então, trabalhar com uma função alvo $f: \mathcal{X} \rightarrow \mathcal{Y}$, com $\mathcal{Y} \subset \mathbb{R}$. Como vimos, nesse caso há uma solução analítica para $\mathbf{w}$, o que nem sempre é o caso. Além disso, fizemos a implementação do algoritmo em Python $-$ em contraponto à utilização direta do Sklearn (que também foi feita no final do texto). Na próxima postagem, discutiremos como trabalhar com transformações não lineares.

Qualquer dúvida, sugestão ou *feedback*, por favor, deixe um comentário abaixo.

> Essa postagem faz parte de uma [série]({{ site.baseurl }}/categories/aprendizado-de-máquina-learning-from-data/) de textos que tem o objetivo de estudar, principalmente, o curso "[Learning from Data](http://www.work.caltech.edu/telecourse.html)", ministrado pelo Professor Yaser Abu-Mostafa. Outros materiais utilizados serão sempre referenciados.