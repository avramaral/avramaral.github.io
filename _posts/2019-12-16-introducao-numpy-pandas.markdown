---
title: "Uma Introdução ao NumPy e Pandas para Análise de Dados"
categories: Tutorial
tags: [python, ciencia-de-dados]
---


Python oferece muitas bibliotecas que extendem suas funcionalidades padrão. Por esse motivo, quando vamos trabalhar com processamento de dados, existem dois pacotes muito úteis que podemos utilizar: **Numpy** e **Pandas**.

**Numpy** cria um novo objeto array multidimensional que nos permite estruturar e operar nosso conjunto de dados de maneira fácil (e muito mais rápido que as tradicionais listas do Python). Além disso, nós também temos o **Pandas**, uma biblioteca de alto-nível construída sobre o código do Numpy que oferece dois novos tipos de estrutura de dados: `Series` e `DataFrame`.  

A primeira coisa que temos que fazer antes de usar essas bibliotecas é importá-las. Para isso, veja as duas linhas de código a seguir:


```python
import numpy  as np
import pandas as pd
```

Agora, nós podemos conferir se os pacotes foram importadas de maneira correta imprimindo suas versões:


```python
print("Numpy  -", np.__version__)
print("Pandas -", pd.__version__)
```

    Numpy  - 1.17.4
    Pandas - 0.25.3


Como dito anteriormente, NumPy é uma biblioteca de baixo-nível, se comparada ao Pandas. Por esse motivo, nós vamos começar com ela, criando algumas arrays.

A fim de criar uma array de 1 dimensão (1D), digite:


```python
my_array = np.array([0, 1, 2], dtype = "int16")
my_array
```




    array([0, 1, 2], dtype=int16)



Note que nós criamos uma variável chamada `my_array` e atribuímos a ela uma array de três elementos: $\lbrace0, 1, 2\rbrace$. Em adição, nós também determinamos o parâmetro `dtype` $-$ isso é importante porque, diferente do que acontece com as variáveis padrão no Python, a biblioteca NumPy nos permite dizer quanto de espaço será alocado na memória para salvar um determinado objeto; nesse sentido, NumPy é mais parecido com a linguagem C, na qual você deve que determinar a quantidade de bits que será utilizada para armazenar algo.

Continuando com a mesma sintaxe, nós podemos criar arrays $n$ dimensionais.


```python
# 2D array (2, 2)
my2D_array = np.array([[11, 12], [21, 22]])
print("2D array, with dtype = {}: \n{}\n".format(my2D_array.dtype, my2D_array))

# 3D array (2, 2, 2)
my3D_array = np.array([[[111, 112], [121, 122]], [[211, 212], [221, 222]]])
print("3D array, with dtype = {}: \n{}\n".format(my3D_array.dtype, my3D_array))
```

    2D array, with dtype = int64: 
    [[11 12]
     [21 22]]
    
    3D array, with dtype = int64: 
    [[[111 112]
      [121 122]]
    
     [[211 212]
      [221 222]]]
    


Note que, para o caso `my2D_array`, nós criamos uma array de 2 dimensões com 2 linhas e 2 colunas; i.e., $(2 \times 2)$. Por outro lado, considerando a variável `my3D_array`, uma array de 3 dimensões, com 2 linhas, 2 colunas e 2 camadas de produnfidade $-$ ou seja, $(2 \times 2 \times 2)$ $-$, foi criada. Por fim, perceba que, como não determinamos manualmente o atributo `dtype`, a biblioteca Numpy o definiu como `int64`, justificado pelos valores que cada umas das arrays assumiu.

Dessa forma, se quisermos fatiar alguns desses elementos, podemos tratar essas arrays de maneira similar às listas do Python:


```python
# on the 2D array, we want to slice the first row: (11, 12)
print("My 2D sliced array:", my2D_array[0, :])

# on the 3D array, we want to slice the elements of the second row and second column for both layers of depth: (221, 222)
print("My 3D sliced array:", my3D_array[1, 1, :])

```

    My 2D sliced array: [11 12]
    My 3D sliced array: [221 222]


Nesse exemplo, é importante notar que os índices começam de 0 (como de costume).

Além desses comandos simples, Numpy também disponibiliza uma grande quantidade de métodos e atributos que podem ser utilizados para realizar tarefas específicas. Vou demonstrar alguns deles, mas para uma lista completa, acesse a documentação oficial ([Numpy Doc](https://docs.scipy.org/doc/numpy-dev/reference/)).

Em relação às operações matemáticas, esses são os métodos mais utilizados:


```python
# create two arrays that will be used on the math operations
a = np.array([[1, 2], [3, 4]]) # array([[1, 2],
                               #        [3, 4]])

b = np.array([[4, 3], [2, 1]]) # array([[4, 3],
                               #        [2, 1]])

# Element by element addition
print("Addition (element by element): \n{}\n".format(np.add(a, b)))

# Element by element subtraction
print("Subraction (element by element): \n{}\n".format(np.subtract(a, b)))

# Element by element multiplication
print("Multiplication (element by element): \n{}\n".format(np.multiply(a, b)))

# Matrix multiplication
print("Matrix multiplication: \n{}\n".format(np.dot(a, b)))

# Element by element division
print("Division (element by element): \n{}\n".format(np.divide(a, b)))
```

    Addition (element by element): 
    [[5 5]
     [5 5]]
    
    Subraction (element by element): 
    [[-3 -1]
     [ 1  3]]
    
    Multiplication (element by element): 
    [[4 6]
     [6 4]]
    
    Matrix multiplication: 
    [[ 8  5]
     [20 13]]
    
    Division (element by element): 
    [[0.25       0.66666667]
     [1.5        4.        ]]
    


O método mais importante aqui é o `np.dot(array_1, array_2)`, que faz a multiplicação "adequada" de duas matrizes; em contraponto ao método `np.multiply(array_1, array_2)`, que realiza o produto termo a termo das duas arrays.

Por fim, antes de começarmos a trabalhar com a biblioteca Pandas, tem mais um exemplo que gostaria de apresentar. Veja a seguir:


```python
# create a 1D array with elements [0, 25[; then reshape it to a 2D array with 5 rows and 5 cols
an_array = np.array(np.arange(0, 25)).reshape(5, 5)
an_array
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])




```python
print("Shape of the array:", an_array.shape)
print("Number of dimensions:", an_array.ndim)
```

    Shape of the array: (5, 5)
    Number of dimensions: 2


Com esses dois pedaços de código acima, nós conseguimos aprender um pouco mais sobre algumas funcionalidades do Numpy. Primeiro, nós criarmos uma array unidimensioal com 25 elementos $\lbrace 0, 1, \cdots, 23, 24 \rbrace$; então, na mesma linha de código, nós utilizamos o método `reshape()` e transformamos esse objeto em uma array $(5 \times 5)$. Finalmente, o "formato" e o número de dimensões da array foram impressos utilizando os atributos `shape` e `ndim`, respectivamente.

Exitem dezenas de outros métodos e atributos diferentes para se explorar com o Numpy, mas isso foi suficiente para uma introdução. Vamos agora trabalhar com o Pandas.

A biblioteca Pandas, como dito no começo do tutorial, fornece duas novas estruturas que serão extremamente importantes para o processamento de dados. Enquanto a estrutura `Series` equivale a uma array com rótulos de 1 dimensão, um `DataFrame` é uma array de duas dimensões que pode ter colunas heterogêneas (o que significa que podemos ter cada coluna de um `Data Frame` armazenando um tipo de dado diferente). Sendo assim, como é possível de se imaginar, um conjunto de `Series` forma um  `DataFrame`.

Agora, podemos começar a escrever algum código utilizando Pandas. Vamos ver como criar e utilizar essas novas ferramentas:


```python
# create a pair of new Series
s1 = pd.Series(['A', 'B', 'C'])
s2 = pd.Series([1.2, 0.7, 3.0])

print("Series 1: \n{}".format(s1))
print()
print("Series 2: \n{}".format(s2))
```

    Series 1: 
    0    A
    1    B
    2    C
    dtype: object
    
    Series 2: 
    0    1.2
    1    0.7
    2    3.0
    dtype: float64


Note que a sintaxe é bem intuitiva; entretanto, o aspecto mais importantes vem do fato de que agora temos duas sequências explicitamente rotuladas e que podem ser combinadas para criar um `DataFrame`.

Um `DataFrame` pode ser encarado como um dicinário de `Series`; assim, a fim de criar um objeto desse tipo, podemos utilizar o seguinte código:


```python
# create a dictionary with the previous Series (s1 and s2)
data = {'1st col': s1, '2nd col': s2}

# create a DataFrame with this dictionary
my_df = pd.DataFrame(data)
my_df
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
      <th>1st col</th>
      <th>2nd col</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



Note que nós começamos com dois objetos do tipo `Series`, e então os combinamos para criar um `DataFrame`. Temos, agora, um conjunto tabulado com informações heterogêneas.

Nesse segundo exemplo, vamos criar um `DataFrame` preenchendo-o com os elementos de uma array criada utilizando Numpy.


```python
my_dataFrame = pd.DataFrame(np.arange(0, 50).reshape(10, 5), columns = ['1st', '2nd', '3rd', '4th', '5th'])
my_dataFrame
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
      <th>1st</th>
      <th>2nd</th>
      <th>3rd</th>
      <th>4th</th>
      <th>5th</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>13</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
      <td>24</td>
    </tr>
    <tr>
      <th>5</th>
      <td>25</td>
      <td>26</td>
      <td>27</td>
      <td>28</td>
      <td>29</td>
    </tr>
    <tr>
      <th>6</th>
      <td>30</td>
      <td>31</td>
      <td>32</td>
      <td>33</td>
      <td>34</td>
    </tr>
    <tr>
      <th>7</th>
      <td>35</td>
      <td>36</td>
      <td>37</td>
      <td>38</td>
      <td>39</td>
    </tr>
    <tr>
      <th>8</th>
      <td>40</td>
      <td>41</td>
      <td>42</td>
      <td>43</td>
      <td>44</td>
    </tr>
    <tr>
      <th>9</th>
      <td>45</td>
      <td>46</td>
      <td>47</td>
      <td>48</td>
      <td>49</td>
    </tr>
  </tbody>
</table>
</div>



Como é possível ver, nós criamos `DataFrame` a partir uma array de duas dimensões gerada utilizando Numpy. 

Uma das operações mais úteis que podemos fazer com essa nova estrutura é, mais uma vez, fatiá-la. Para fazer isso, podemos utilizar o atributo `iloc[]` a fim de selecionar uma porção do `DataFrame` original. 


```python
# using the same "my_dataFrame" DataFrame
# select the 3rd and 4th columns and the rows with indexes from 4 to 8
my_dataFrame[['3rd', '4th']].iloc[4:9]
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
      <th>3rd</th>
      <th>4th</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>23</td>
    </tr>
    <tr>
      <th>5</th>
      <td>27</td>
      <td>28</td>
    </tr>
    <tr>
      <th>6</th>
      <td>32</td>
      <td>33</td>
    </tr>
    <tr>
      <th>7</th>
      <td>37</td>
      <td>38</td>
    </tr>
    <tr>
      <th>8</th>
      <td>42</td>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>



Perceba que, se você quiser selecionar mais de uma coluna, terá que utilizar um lista para agrupá-las.

A próxima alternativa para fatiar um `DataFrame` é criando uma máscara que utiliza algum tipo de condicional; por exemplo, se nós quisermos recuperar, na terceira coluna (`3rd`), os valores que são maiores que 20, nós podemos fazer o seguinte:


```python
# create a mask
mask = my_dataFrame['3rd'] > 30

# apply the mask to the "3rd" column in order to slice the DataFrame considering the given condition
my_dataFrame['3rd'][mask]
```




    6    32
    7    37
    8    42
    9    47
    Name: 3rd, dtype: int64



Note que nós primeiro criamos a máscara, e então a aplicamos sobre o `DataFrame`, escolhendo tanto as colunas (`3rd`) quanto as linhas (aquelas que tem valor maior que 30) desejadas.

Finalmente, nós vamos ver alguns métodos da biblioteca Pandas; porém, como já dito anteriormente, eu recomendo fortemente que você leia a [documentação oficial do Pandas](http://pandas.pydata.org/pandas-docs/stable/index.html).

A seguir, vamos criar um conjunto de dados para os nossos próximos exemplos.


```python
# let's create a DataFrame for the next demonstrations

col_labels = ['Name', 'Age', 'Nationality']

name        = pd.Series(['André', 'James', 'Agata', 'María', 'Pedro', 'Juan', 'Paul'])
age         = pd.Series([23, 27, 21, None, 27, 22, 25])
nationality = pd.Series(['Brazilian', 'American', 'Greek', 'Mexican', 'Brazilian', 'Mexican', 'British'])

people = {col_labels[0]: name,
          col_labels[1]: age,
          col_labels[2]: nationality}

df_people = pd.DataFrame(people)
df_people.head()
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
      <th>Name</th>
      <th>Age</th>
      <th>Nationality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>André</td>
      <td>23.0</td>
      <td>Brazilian</td>
    </tr>
    <tr>
      <th>1</th>
      <td>James</td>
      <td>27.0</td>
      <td>American</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Agata</td>
      <td>21.0</td>
      <td>Greek</td>
    </tr>
    <tr>
      <th>3</th>
      <td>María</td>
      <td>NaN</td>
      <td>Mexican</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pedro</td>
      <td>27.0</td>
      <td>Brazilian</td>
    </tr>
  </tbody>
</table>
</div>



A primeira coisa a observar, é a ulização do médoto `head()`, que retorna, por padrão, apenas as 5 primeiras linhas do nosso banco de dados. É mais conveniente visualizar apenas as primeiras linhas do `DataFrame` quando o conjunto de dados com o qual se está trabalhando é grande demais. Além disso, uma das idades (`Age`) está "em branco" (isso é muito comum em aplicações do mundo real, e nós vamos ver como tratar esse tipo de problema).

Nós podemos começar lidando com o valor `None`. Existem algumas estratégias diferentes que podemos tomar; entretanto, a fim de manter esse tutorial o mais simples possível, vamos apenas eliminar a linha que contém esse problema. Para fazer isso, podemos utilizar o método `dropna()`:


```python
df_people.dropna(inplace = True)
df_people
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
      <th>Name</th>
      <th>Age</th>
      <th>Nationality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>André</td>
      <td>23.0</td>
      <td>Brazilian</td>
    </tr>
    <tr>
      <th>1</th>
      <td>James</td>
      <td>27.0</td>
      <td>American</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Agata</td>
      <td>21.0</td>
      <td>Greek</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pedro</td>
      <td>27.0</td>
      <td>Brazilian</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Juan</td>
      <td>22.0</td>
      <td>Mexican</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Paul</td>
      <td>25.0</td>
      <td>British</td>
    </tr>
  </tbody>
</table>
</div>



Como é possível de ser visto, nós removemos a linha com índice 3. Para fazer isso de forma permanente, foi necessário atribuir o valor `True` ao parâmetro `inplace`.

Agora, nós podemos ter uma visão geral do conjunto de dados utilizando o método `info()`.


```python
df_people.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 6 entries, 0 to 6
    Data columns (total 3 columns):
    Name           6 non-null object
    Age            6 non-null float64
    Nationality    6 non-null object
    dtypes: float64(1), object(2)
    memory usage: 192.0+ bytes


Veja que o método utlizado nós mostrou que temos 6 entradas não nulas para cada uma das 3 colunas.

A seguir, vamos assumir que queremos saber a média de idade das pessoas com nacionalidade brasileira (`Brazilian`). Para essa situação hipotética, a primeira coisa que temos que fazer é criar uma máscara para selecionar os invíduos que nasceram no Brasil.


```python
# create a mask
n_mask = df_people['Nationality'] == 'Brazilian'

# apply the mask
df_people[n_mask]
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
      <th>Name</th>
      <th>Age</th>
      <th>Nationality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>André</td>
      <td>23.0</td>
      <td>Brazilian</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pedro</td>
      <td>27.0</td>
      <td>Brazilian</td>
    </tr>
  </tbody>
</table>
</div>



Com essas linhas de código, nós filtramos o `DataFrame` para mostrar apenas as pessoas com nacionalidade brasileira. Falta, então, calcular a média de suas idades:


```python
average_age = df_people[n_mask].mean()
average_age
```




    Age    25.0
    dtype: float64



Note que o resultado é uma estrutura do tipo `Series`. Dessa forma, se quisermos formatá-lo, podemos utilizar o atributo `values`, que retorna os valores do objeto em questão como uma array Numpy.


```python
print("The average age of the Brazilian citizens is {:.0f} years.".format(average_age.values[0]))
```

    The average age of the Brazilian citizens is 25 years.


## Conclusão

**Numpy** e **Pandas** são duas bibliotecas essenciais para se trabalhar com análise de dados. **Numpy** introduz objetos do tipo `ndarray` (arrays $n$ dimensionais) e o **Pandas** implementa duas novas estruturas de dados: `Series` e `DataFrame`. Dessa forma, se você quiser utilizar os conceitos de ciência de dados, aprendizagem de máquina, etc. nos seus projetos com Python, você deve aprender a utilizar essas excelentes ferramentas.

Qualquer dúvida, sugestão ou *feedback*, por favor, deixe um comentário abaixo.





> Eu havia escrito esse texto, originalmente em inglês, em uma antiga versão do blog que já não existe mais. Pequenas correções e atualizações, além da própria tradução, foram feitas para que o conteúdo continuasse relevante.