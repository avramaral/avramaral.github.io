---
title: "Visualização de Dados com Matplotlib"
categories: Tutorial
tags: [python, ciencia-de-dados]
---

Uma parte essencial do processo de análise de dados é a visualização parcial e final dos resultados que foram alcançados. Na maior parte das vezes, é muito mais fácil interpretar esse tipo de informação graficamente do que através de, apenas, tabelas ou números.

Entretanto, mais do que apenas ser capaz de plotar gráficos, é importante que o pesquisador (ou estudante, ou cientista de dados, etc.) represente esse conjunto de dados de uma forma que sua audiência consiga entender. Você precisa ter gráficos acessíveis, confiáveis e elegantes ([Kirk, A](https://www.amazon.com/Data-Visualisation-Handbook-Driven-Design/dp/1473912148). citado em [Curso edX](https://www.edx.org/course/python-data-science-uc-san-diegox-dse200x)).

Em Python, o pacote mais básico para visualização de dados é o **Matplotlib**. Ele nos permite plotar diferentes tipos de gráficos com dezenas de opções de customização. Por esse motivo, nós o escolhemos para esse tutorial.

![Tela principal]({{ site.baseurl }}/assets/images/visualizacao-de-dados-com-matplotlib_files/matplotlib-principal.png)

Toda a imagem mostrada acima é uma `Figure` criada pelo módulo `matplotlib.pyplot`; dentro dela, na área branca, estão compreendidos os eixos `x` e `y`, chamadas de `Axes` ou `Subplot` (existe uma pequena diferença entre esses dois termos, mas para a maioria dos casos, eles podem ser tratados como sinônimos). Além disso, no canto superior esquerdo, existe um pequeno menu que nos permite realizar algumas ações $-$ incluindo exportar o gráfico como imagem.

**Observação:** A partir desse ponto, eu não vou mais apresentar toda a janela mostrada na figura acima; ao invés disso, apenas os gráficos serão gerados.

Agora que sabemos o básico, vamos importar a biblioteca e verificar se ela está funcionando.


```python
%matplotlib inline
import matplotlib.pyplot as plt
```

Perceba que, na maior parte das situações, nós só temos que importar o módulo `pyplot`. Assim, se nenhuma mensagem de erro apareceu, o Matplotlib foi corretamente importado.

**Observação 2:** se você está utilizando o Jupyter Notebook para realizar seus testes, é necessário que o comando `%matplotlib inline` seja incluído antes de importar a biblioteca.

Como você deve imaginar, a primeira coisa que temos que fazer é criar uma `Figure`:


```python
fig = plt.figure(figsize = (3, 6))
```


    <Figure size 216x432 with 0 Axes>


Além de criar um objeto `Figure` e atribuí-lo à variável `fig`, nós defimos os valores para o parâmetro `figsize`, que, como o nome sugere, determina o tamanho da janela: $3 \times 6$ polegadas.

Agora nós criamos, dentro do "container" `fig`, um objeto `Axes`. Para isso, basta utilizar o método `add_subplot()`:


```python
ax1 = fig.add_subplot(111)
```

Note que, mais uma vez, nós tivemos que passar o valor de um parâmetro para a função. Nesse caso, o `111` representa o número de objetos `Axes` que serão criados dentro de `fig`: uma array com `1` linha, `1` coluna e índice `1`.

O próximo passo é definir os dados que serão utilizados para plotar o gráfico. Para esse primeiro exemplo, eu vou criar duas listas que ilustram o comportamento da função $f(x) = 2x$ com domínio em $\lbrace 0, 1, \cdots, 10 \rbrace$:


```python
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
```

Feito isso, agora só precisamos plotar esses valores em um sistema de coordenadas utilizando o comando `plot()`; e, por fim, chamar o método `show()` para mostrá-lo na tela:


```python
# all the code togheter
fig = plt.figure(figsize = (3, 6))
ax1 = fig.add_subplot(111)
ax1.scatter(x, y)
plt.show()
```


![png]({{ site.baseurl }}/assets/images/visualizacao-de-dados-com-matplotlib_files/visualizacao-de-dados-com-matplotlib_19_0.png)


**Observação 3**: mais uma vez, se você está utilizando Jupyter Notebook, talvez seja necessário que todos os comandos estejam em uma única célula.

Em adição ao que acabamos de fazer, é possível modificar algumas característica do gráfico, veja a seguir:


```python
# all the code togheter
fig = plt.figure(figsize = (2, 4)) # intentionally smaller than the previous one
ax1 = fig.add_subplot(111)
ax1.scatter(x, y, label = 'f(x)', color = 'red')

# ax1 properties
ax1.set_title('My Title')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

plt.legend(loc = 4)
plt.show()
```


![png]({{ site.baseurl }}/assets/images/visualizacao-de-dados-com-matplotlib_files/visualizacao-de-dados-com-matplotlib_22_0.png)


Nós incluímos informações importantes com essas últimas linhas de código. No método `scatter()`, nós adicionamos dois parâmetros: `label`e `color`. O primeiro deles é usado para identificar a curva que está sendo plotada (através do método `legend()`), enquanto que o segundo define a cor dos pontos.

Além disso, definimos algumas novas propriedades para o objeto `ax1`: o método `set_title()` define o título do gráfico, e os métodos `set_xlabel` e `set_ylabel` definem os rótulos que serão utilizados nos eixos $x$ e $y$, respectivamente.

A biblioteca Matplotlib, como mencionado no começo desse texto, é muito flexível; por isso, a fim de extrair todo o seu potencial, é importante que você leia a [documentação oficial](https://matplotlib.org/contents.html).

Finalmente, um outro exemplo, um pouco mais complexo, será apresentado. Primeiro, veja o seguinte código:


```python
import numpy as np

# create a new figure
fig = plt.figure(figsize = (10, 5))

# create two new Axes objects in the format of a '1 row x 2 columns' array
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# create the data set that will be plotted
x = np.linspace(0, np.pi * 2)
y = np.sin(x) # y = f(x) = sin(x)
z = np.cos(x) # z = g(x) = cos(x)

# plot f(x) and g(x)
ax1.plot(x, y, label = 'sin', color = 'b') # 'b' stands for 'blue'
ax2.plot(x, z, label = 'cos', color = 'r') # 'r' stands for 'red'

# set 'ax1' properties
ax1.set_title('Trigonometric Functions - SINE')
ax1.set_xlabel('Angle (in radian)')
ax1.set_ylabel('Magnitude')
ax1.axis([0, np.pi * 2, -1.25, 1.25]) # define the range of both axes
# define ticks for each axis (3 auxiliary variables)
xticks = [0, np.pi / 2, np.pi, np.pi * 1.5, np.pi * 2]                         # define the ticks values - 'X' axis
xlabel = ['0', '$\\frac{\\pi}{2}$', '$\\pi$', '$\\frac{3\\pi}{2}$', '$2\\pi$'] # define the label values - 'X' axis
yticks = [-1, -0.5, 0, 0.5, 1]                                                 # define the ticks values - 'Y' axis
ax1.xaxis.set(ticks = xticks, ticklabels = xlabel)
ax1.yaxis.set(ticks = yticks)
ax1.legend(loc = 1)

# set 'ax1' properties
ax2.set_title('Trigonometric Functions - COSINE')
ax2.set_xlabel('Angle (in radian)')
ax2.set_ylabel('Magnitude')
ax2.axis([0, np.pi * 2, -1.25, 1.25]) # define the range of both axes
# define ticks for each axis (3 auxiliary variables)
xticks = [0, np.pi / 2, np.pi, np.pi * 1.5, np.pi * 2]                         # define the ticks values - 'X' axis
xlabel = ['0', '$\\frac{\\pi}{2}$', '$\\pi$', '$\\frac{3\\pi}{2}$', '$2\\pi$'] # define the label values - 'X' axis
yticks = [-1, -0.5, 0, 0.5, 1]                                                 # define the ticks values - 'Y' axis
ax2.xaxis.set(ticks = xticks, ticklabels = xlabel)
ax2.yaxis.set(ticks = yticks)
ax2.legend(loc = 1)

plt.tight_layout() # fix the spaces between the two Axes
plt.show()
```


![png]({{ site.baseurl }}/assets/images/visualizacao-de-dados-com-matplotlib_files/visualizacao-de-dados-com-matplotlib_27_0.png)


Nós acabamos de criar uma `Figure` com dois `Subplot`'s dentro dela. As imagens da esquerda e da direita são definidas pelas funções seno e cosseno, respectivametne. Além disso, há algumas coisas novas aqui; por isso, os próximos parágrafos serão dedicados a esses detalhes.

Depois de importar, além do Matplotlib, o Numpy (veja [aqui](/introducao-numpy-pandas/) um pequeno tutorial sobre Numpy e Pandas), nós criamos uma `Figure` com um tamanho específico. Agora, utilizando esse objeto, que foi armazenado na variável `fig`, foram criados dois `Subplot`'s $-$ **Importante:** note que para o primeiro objeto, o valor do parâmetro foi `121`(que significa `1` linha, `2` colunas e o índice `1`), enquanto que para o segundo, esse número teve que ser definido como `122`. 

Nas linhas seguintes, a única coisa que fizemos foi criar os nossos vetores de dados `x`, `y` e `z`. Com o Numpy, nós geramos uma lista, atribuída a `x`, de valores igualmente espaçados entre $0$ e $2\pi$; então, nas variáveis `y` e `z`, nós calculamos e salvamos os valores de seno e cosseno de `x`, respectivamente.

Depois disso tudo, diferentemente do primeiro exemplo, utilizamos o método `plot()` (ao invés de `scatter()`). Essa modificação faz com que o gráfico de pontos seja, agora, exibido como gráfico de linha.

Por fim, sobre as propriedades de `ax1` e `ax2`, descreverei apenas os atributos e métodos que ainda não foram discutidos. A primeira novidade é o método `axis()`, que define os valores mínimo e máximo que serão mostrados em cada eixo do gráfico $-$ perceba que é necessário passar uma **lista** de números. Além disso, nós também criamos algumas variáveis que foram utilizadas para definir os parametros `ticks` e `ticklabes` dos métodos `xaxis.set()` e `yaxis.set()`. Esses métodos definem em quais pontos ao longo dos eixos você terá um indicador de valor.

Um pequeno detalhe é o de que foi necessário, antes de chamar o método `show()`, utilizar a função `tight_layout()` a fim de ajustar o tamanho e espaçamento dos gráficos.

## Conclusão

**Matplotlib** é uma biblioteca muito flexível e poderosa que nos permite criar diferentes tipos de gráficos. Além disso, como mencionado anteriormente, existe uma galeria oficial cheia de exemplos reais criados a partir dessa ferramenta $-$ confira, e veja todas as possibilidades.

Qualquer dúvida, sugestão ou *feedback*, por favor, deixe um comentário abaixo.



> Eu havia escrito esse texto, originalmente em inglês, em uma antiga versão do blog que já não existe mais. Pequenas correções e atualizações, além da própria tradução, foram feitas para que o conteúdo continuasse relevante.