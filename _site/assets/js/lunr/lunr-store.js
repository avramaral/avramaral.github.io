var store = [{
        "title": "Uma Introdução ao NumPy e Pandas para Análise de Dados",
        "excerpt":"Python oferece muitas bibliotecas que extendem suas funcionalidades padrão. Por esse motivo, quando vamos trabalhar com processamento de dados, existem dois pacotes muito úteis que podemos utilizar: Numpy e Pandas.   Numpy cria um novo objeto array multidimensional que nos permite estruturar e operar nosso conjunto de dados de maneira fácil (e muito mais rápido que as tradicionais listas do Python). Além disso, nós também temos o Pandas, uma biblioteca de alto-nível construída sobre o código do Numpy que oferece dois novos tipos de estrutura de dados: Series e DataFrame.   A primeira coisa que temos que fazer antes de usar essas bibliotecas é importá-las. Para isso, veja as duas linhas de código a seguir:   import numpy  as np import pandas as pd   Agora, nós podemos conferir se os pacotes foram importadas de maneira correta imprimindo suas versões:   print(\"Numpy  -\", np.__version__) print(\"Pandas -\", pd.__version__)   Numpy  - 1.17.4 Pandas - 0.25.3   Como dito anteriormente, NumPy é uma biblioteca de baixo-nível, se comparada ao Pandas. Por esse motivo, nós vamos começar com ela, criando algumas arrays.   A fim de criar uma array de 1 dimensão (1D), digite:   my_array = np.array([0, 1, 2], dtype = \"int16\") my_array   array([0, 1, 2], dtype=int16)   Note que nós criamos uma variável chamada my_array e atribuímos a ela uma array de três elementos: $\\lbrace0, 1, 2\\rbrace$. Em adição, nós também determinamos o parâmetro dtype $-$ isso é importante porque, diferente do que acontece com as variáveis padrão no Python, a biblioteca NumPy nos permite dizer quanto de espaço será alocado na memória para salvar um determinado objeto; nesse sentido, NumPy é mais parecido com a linguagem C, na qual você deve que determinar a quantidade de bits que será utilizada para armazenar algo.   Continuando com a mesma sintaxe, nós podemos criar arrays $n$ dimensionais.   # 2D array (2, 2) my2D_array = np.array([[11, 12], [21, 22]]) print(\"2D array, with dtype = {}: \\n{}\\n\".format(my2D_array.dtype, my2D_array))  # 3D array (2, 2, 2) my3D_array = np.array([[[111, 112], [121, 122]], [[211, 212], [221, 222]]]) print(\"3D array, with dtype = {}: \\n{}\\n\".format(my3D_array.dtype, my3D_array))   2D array, with dtype = int64:  [[11 12]  [21 22]]  3D array, with dtype = int64:  [[[111 112]   [121 122]]   [[211 212]   [221 222]]]   Note que, para o caso my2D_array, nós criamos uma array de 2 dimensões com 2 linhas e 2 colunas; i.e., $(2 \\times 2)$. Por outro lado, considerando a variável my3D_array, uma array de 3 dimensões, com 2 linhas, 2 colunas e 2 camadas de produnfidade $-$ ou seja, $(2 \\times 2 \\times 2)$ $-$, foi criada. Por fim, perceba que, como não determinamos manualmente o atributo dtype, a biblioteca Numpy o definiu como int64, justificado pelos valores que cada umas das arrays assumiu.   Dessa forma, se quisermos fatiar alguns desses elementos, podemos tratar essas arrays de maneira similar às listas do Python:   # on the 2D array, we want to slice the first row: (11, 12) print(\"My 2D sliced array:\", my2D_array[0, :])  # on the 3D array, we want to slice the elements of the second row and second column for both layers of depth: (221, 222) print(\"My 3D sliced array:\", my3D_array[1, 1, :])    My 2D sliced array: [11 12] My 3D sliced array: [221 222]   Nesse exemplo, é importante notar que os índices começam de 0 (como de costume).   Além desses comandos simples, Numpy também disponibiliza uma grande quantidade de métodos e atributos que podem ser utilizados para realizar tarefas específicas. Vou demonstrar alguns deles, mas para uma lista completa, acesse a documentação oficial (Numpy Doc).   Em relação às operações matemáticas, esses são os métodos mais utilizados:   # create two arrays that will be used on the math operations a = np.array([[1, 2], [3, 4]]) # array([[1, 2],                                #        [3, 4]])  b = np.array([[4, 3], [2, 1]]) # array([[4, 3],                                #        [2, 1]])  # Element by element addition print(\"Addition (element by element): \\n{}\\n\".format(np.add(a, b)))  # Element by element subtraction print(\"Subraction (element by element): \\n{}\\n\".format(np.subtract(a, b)))  # Element by element multiplication print(\"Multiplication (element by element): \\n{}\\n\".format(np.multiply(a, b)))  # Matrix multiplication print(\"Matrix multiplication: \\n{}\\n\".format(np.dot(a, b)))  # Element by element division print(\"Division (element by element): \\n{}\\n\".format(np.divide(a, b)))   Addition (element by element):  [[5 5]  [5 5]]  Subraction (element by element):  [[-3 -1]  [ 1  3]]  Multiplication (element by element):  [[4 6]  [6 4]]  Matrix multiplication:  [[ 8  5]  [20 13]]  Division (element by element):  [[0.25       0.66666667]  [1.5        4.        ]]   O método mais importante aqui é o np.dot(array_1, array_2), que faz a multiplicação “adequada” de duas matrizes; em contraponto ao método np.multiply(array_1, array_2), que realiza o produto termo a termo das duas arrays.   Por fim, antes de começarmos a trabalhar com a biblioteca Pandas, tem mais um exemplo que gostaria de apresentar. Veja a seguir:   # create a 1D array with elements [0, 25[; then reshape it to a 2D array with 5 rows and 5 cols an_array = np.array(np.arange(0, 25)).reshape(5, 5) an_array   array([[ 0,  1,  2,  3,  4],        [ 5,  6,  7,  8,  9],        [10, 11, 12, 13, 14],        [15, 16, 17, 18, 19],        [20, 21, 22, 23, 24]])   print(\"Shape of the array:\", an_array.shape) print(\"Number of dimensions:\", an_array.ndim)   Shape of the array: (5, 5) Number of dimensions: 2   Com esses dois pedaços de código acima, nós conseguimos aprender um pouco mais sobre algumas funcionalidades do Numpy. Primeiro, nós criarmos uma array unidimensioal com 25 elementos $\\lbrace 0, 1, \\cdots, 23, 24 \\rbrace$; então, na mesma linha de código, nós utilizamos o método reshape() e transformamos esse objeto em uma array $(5 \\times 5)$. Finalmente, o “formato” e o número de dimensões da array foram impressos utilizando os atributos shape e ndim, respectivamente.   Exitem dezenas de outros métodos e atributos diferentes para se explorar com o Numpy, mas isso foi suficiente para uma introdução. Vamos agora trabalhar com o Pandas.   A biblioteca Pandas, como dito no começo do tutorial, fornece duas novas estruturas que serão extremamente importantes para o processamento de dados. Enquanto a estrutura Series equivale a uma array com rótulos de 1 dimensão, um DataFrame é uma array de duas dimensões que pode ter colunas heterogêneas (o que significa que podemos ter cada coluna de um Data Frame armazenando um tipo de dado diferente). Sendo assim, como é possível de se imaginar, um conjunto de Series forma um  DataFrame.   Agora, podemos começar a escrever algum código utilizando Pandas. Vamos ver como criar e utilizar essas novas ferramentas:   # create a pair of new Series s1 = pd.Series(['A', 'B', 'C']) s2 = pd.Series([1.2, 0.7, 3.0])  print(\"Series 1: \\n{}\".format(s1)) print() print(\"Series 2: \\n{}\".format(s2))   Series 1:  0    A 1    B 2    C dtype: object  Series 2:  0    1.2 1    0.7 2    3.0 dtype: float64   Note que a sintaxe é bem intuitiva; entretanto, o aspecto mais importantes vem do fato de que agora temos duas sequências explicitamente rotuladas e que podem ser combinadas para criar um DataFrame.   Um DataFrame pode ser encarado como um dicinário de Series; assim, a fim de criar um objeto desse tipo, podemos utilizar o seguinte código:   # create a dictionary with the previous Series (s1 and s2) data = {'1st col': s1, '2nd col': s2}  # create a DataFrame with this dictionary my_df = pd.DataFrame(data) my_df                           1st col       2nd col                       0       A       1.2                 1       B       0.7                 2       C       3.0            Note que nós começamos com dois objetos do tipo Series, e então os combinamos para criar um DataFrame. Temos, agora, um conjunto tabulado com informações heterogêneas.   Nesse segundo exemplo, vamos criar um DataFrame preenchendo-o com os elementos de uma array criada utilizando Numpy.   my_dataFrame = pd.DataFrame(np.arange(0, 50).reshape(10, 5), columns = ['1st', '2nd', '3rd', '4th', '5th']) my_dataFrame                           1st       2nd       3rd       4th       5th                       0       0       1       2       3       4                 1       5       6       7       8       9                 2       10       11       12       13       14                 3       15       16       17       18       19                 4       20       21       22       23       24                 5       25       26       27       28       29                 6       30       31       32       33       34                 7       35       36       37       38       39                 8       40       41       42       43       44                 9       45       46       47       48       49            Como é possível ver, nós criamos DataFrame a partir uma array de duas dimensões gerada utilizando Numpy.   Uma das operações mais úteis que podemos fazer com essa nova estrutura é, mais uma vez, fatiá-la. Para fazer isso, podemos utilizar o atributo iloc[] a fim de selecionar uma porção do DataFrame original.   # using the same \"my_dataFrame\" DataFrame # select the 3rd and 4th columns and the rows with indexes from 4 to 8 my_dataFrame[['3rd', '4th']].iloc[4:9]                           3rd       4th                       4       22       23                 5       27       28                 6       32       33                 7       37       38                 8       42       43            Perceba que, se você quiser selecionar mais de uma coluna, terá que utilizar um lista para agrupá-las.   A próxima alternativa para fatiar um DataFrame é criando uma máscara que utiliza algum tipo de condicional; por exemplo, se nós quisermos recuperar, na terceira coluna (3rd), os valores que são maiores que 20, nós podemos fazer o seguinte:   # create a mask mask = my_dataFrame['3rd'] &gt; 30  # apply the mask to the \"3rd\" column in order to slice the DataFrame considering the given condition my_dataFrame['3rd'][mask]   6    32 7    37 8    42 9    47 Name: 3rd, dtype: int64   Note que nós primeiro criamos a máscara, e então a aplicamos sobre o DataFrame, escolhendo tanto as colunas (3rd) quanto as linhas (aquelas que tem valor maior que 30) desejadas.   Finalmente, nós vamos ver alguns métodos da biblioteca Pandas; porém, como já dito anteriormente, eu recomendo fortemente que você leia a documentação oficial do Pandas.   A seguir, vamos criar um conjunto de dados para os nossos próximos exemplos.   # let's create a DataFrame for the next demonstrations  col_labels = ['Name', 'Age', 'Nationality']  name        = pd.Series(['André', 'James', 'Agata', 'María', 'Pedro', 'Juan', 'Paul']) age         = pd.Series([23, 27, 21, None, 27, 22, 25]) nationality = pd.Series(['Brazilian', 'American', 'Greek', 'Mexican', 'Brazilian', 'Mexican', 'British'])  people = {col_labels[0]: name,           col_labels[1]: age,           col_labels[2]: nationality}  df_people = pd.DataFrame(people) df_people.head()                           Name       Age       Nationality                       0       André       23.0       Brazilian                 1       James       27.0       American                 2       Agata       21.0       Greek                 3       María       NaN       Mexican                 4       Pedro       27.0       Brazilian            A primeira coisa a observar, é a ulização do médoto head(), que retorna, por padrão, apenas as 5 primeiras linhas do nosso banco de dados. É mais conveniente visualizar apenas as primeiras linhas do DataFrame quando o conjunto de dados com o qual se está trabalhando é grande demais. Além disso, uma das idades (Age) está “em branco” (isso é muito comum em aplicações do mundo real, e nós vamos ver como tratar esse tipo de problema).   Nós podemos começar lidando com o valor None. Existem algumas estratégias diferentes que podemos tomar; entretanto, a fim de manter esse tutorial o mais simples possível, vamos apenas eliminar a linha que contém esse problema. Para fazer isso, podemos utilizar o método dropna():   df_people.dropna(inplace = True) df_people                           Name       Age       Nationality                       0       André       23.0       Brazilian                 1       James       27.0       American                 2       Agata       21.0       Greek                 4       Pedro       27.0       Brazilian                 5       Juan       22.0       Mexican                 6       Paul       25.0       British            Como é possível de ser visto, nós removemos a linha com índice 3. Para fazer isso de forma permanente, foi necessário atribuir o valor True ao parâmetro inplace.   Agora, nós podemos ter uma visão geral do conjunto de dados utilizando o método info().   df_people.info()   &lt;class 'pandas.core.frame.DataFrame'&gt; Int64Index: 6 entries, 0 to 6 Data columns (total 3 columns): Name           6 non-null object Age            6 non-null float64 Nationality    6 non-null object dtypes: float64(1), object(2) memory usage: 192.0+ bytes   Veja que o método utlizado nós mostrou que temos 6 entradas não nulas para cada uma das 3 colunas.   A seguir, vamos assumir que queremos saber a média de idade das pessoas com nacionalidade brasileira (Brazilian). Para essa situação hipotética, a primeira coisa que temos que fazer é criar uma máscara para selecionar os invíduos que nasceram no Brasil.   # create a mask n_mask = df_people['Nationality'] == 'Brazilian'  # apply the mask df_people[n_mask]                           Name       Age       Nationality                       0       André       23.0       Brazilian                 4       Pedro       27.0       Brazilian            Com essas linhas de código, nós filtramos o DataFrame para mostrar apenas as pessoas com nacionalidade brasileira. Falta, então, calcular a média de suas idades:   average_age = df_people[n_mask].mean() average_age   Age    25.0 dtype: float64   Note que o resultado é uma estrutura do tipo Series. Dessa forma, se quisermos formatá-lo, podemos utilizar o atributo values, que retorna os valores do objeto em questão como uma array Numpy.   print(\"The average age of the Brazilian citizens is {:.0f} years.\".format(average_age.values[0]))   The average age of the Brazilian citizens is 25 years.   Conclusão   Numpy e Pandas são duas bibliotecas essenciais para se trabalhar com análise de dados. Numpy introduz objetos do tipo ndarray (arrays $n$ dimensionais) e o Pandas implementa duas novas estruturas de dados: Series e DataFrame. Dessa forma, se você quiser utilizar os conceitos de ciência de dados, aprendizagem de máquina, etc. nos seus projetos com Python, você deve aprender a utilizar essas excelentes ferramentas.   Qualquer dúvida, sugestão ou feedback, por favor, deixe um comentário abaixo.      Eu havia escrito esse texto, originalmente em inglês, em uma antiga versão do blog que já não existe mais. Pequenas correções e atualizações, além da própria tradução, foram feitas para que o conteúdo continuasse relevante.   ","categories": ["Tutorial"],
        "tags": ["python","ciencia-de-dados"],
        "url": "http://localhost:4000/introducao-numpy-pandas/",
        "teaser":null},{
        "title": "Visualização de Dados com Matplotlib",
        "excerpt":"Uma parte essencial do processo de análise de dados é a visualização parcial e final dos resultados que foram alcançados. Na maior parte das vezes, é muito mais fácil interpretar esse tipo de informação graficamente do que através de, apenas, tabelas ou números.   Entretanto, mais do que apenas ser capaz de plotar gráficos, é importante que o pesquisador (ou estudante, ou cientista de dados, etc.) represente esse conjunto de dados de uma forma que sua audiência consiga entender. Você precisa ter gráficos acessíveis, confiáveis e elegantes (Kirk, A. citado em Curso edX).   Em Python, o pacote mais básico para visualização de dados é o Matplotlib. Ele nos permite plotar diferentes tipos de gráficos com dezenas de opções de customização. Por esse motivo, nós o escolhemos para esse tutorial.      Toda a imagem mostrada acima é uma Figure criada pelo módulo matplotlib.pyplot; dentro dela, na área branca, estão compreendidos os eixos x e y, chamadas de Axes ou Subplot (existe uma pequena diferença entre esses dois termos, mas para a maioria dos casos, eles podem ser tratados como sinônimos). Além disso, no canto superior esquerdo, existe um pequeno menu que nos permite realizar algumas ações $-$ incluindo exportar o gráfico como imagem.   Observação: A partir desse ponto, eu não vou mais apresentar toda a janela mostrada na figura acima; ao invés disso, apenas os gráficos serão gerados.   Agora que sabemos o básico, vamos importar a biblioteca e verificar se ela está funcionando.   %matplotlib inline import matplotlib.pyplot as plt   Perceba que, na maior parte das situações, nós só temos que importar o módulo pyplot. Assim, se nenhuma mensagem de erro apareceu, o Matplotlib foi corretamente importado.   Observação 2: se você está utilizando o Jupyter Notebook para realizar seus testes, é necessário que o comando %matplotlib inline seja incluído antes de importar a biblioteca.   Como você deve imaginar, a primeira coisa que temos que fazer é criar uma Figure:   fig = plt.figure(figsize = (3, 6))   &lt;Figure size 216x432 with 0 Axes&gt;   Além de criar um objeto Figure e atribuí-lo à variável fig, nós defimos os valores para o parâmetro figsize, que, como o nome sugere, determina o tamanho da janela: $3 \\times 6$ polegadas.   Agora nós criamos, dentro do “container” fig, um objeto Axes. Para isso, basta utilizar o método add_subplot():   ax1 = fig.add_subplot(111)   Note que, mais uma vez, nós tivemos que passar o valor de um parâmetro para a função. Nesse caso, o 111 representa o número de objetos Axes que serão criados dentro de fig: uma array com 1 linha, 1 coluna e índice 1.   O próximo passo é definir os dados que serão utilizados para plotar o gráfico. Para esse primeiro exemplo, eu vou criar duas listas que ilustram o comportamento da função $f(x) = 2x$ com domínio em $\\lbrace 0, 1, \\cdots, 10 \\rbrace$:   x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] y = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]   Feito isso, agora só precisamos plotar esses valores em um sistema de coordenadas utilizando o comando plot(); e, por fim, chamar o método show() para mostrá-lo na tela:   # all the code togheter fig = plt.figure(figsize = (3, 6)) ax1 = fig.add_subplot(111) ax1.scatter(x, y) plt.show()      Observação 3: mais uma vez, se você está utilizando Jupyter Notebook, talvez seja necessário que todos os comandos estejam em uma única célula.   Em adição ao que acabamos de fazer, é possível modificar algumas característica do gráfico, veja a seguir:   # all the code togheter fig = plt.figure(figsize = (2, 4)) # intentionally smaller than the previous one ax1 = fig.add_subplot(111) ax1.scatter(x, y, label = 'f(x)', color = 'red')  # ax1 properties ax1.set_title('My Title') ax1.set_xlabel('X') ax1.set_ylabel('Y')  plt.legend(loc = 4) plt.show()      Nós incluímos informações importantes com essas últimas linhas de código. No método scatter(), nós adicionamos dois parâmetros: labele color. O primeiro deles é usado para identificar a curva que está sendo plotada (através do método legend()), enquanto que o segundo define a cor dos pontos.   Além disso, definimos algumas novas propriedades para o objeto ax1: o método set_title() define o título do gráfico, e os métodos set_xlabel e set_ylabel definem os rótulos que serão utilizados nos eixos $x$ e $y$, respectivamente.   A biblioteca Matplotlib, como mencionado no começo desse texto, é muito flexível; por isso, a fim de extrair todo o seu potencial, é importante que você leia a documentação oficial.   Finalmente, um outro exemplo, um pouco mais complexo, será apresentado. Primeiro, veja o seguinte código:   import numpy as np  # create a new figure fig = plt.figure(figsize = (10, 5))  # create two new Axes objects in the format of a '1 row x 2 columns' array ax1 = fig.add_subplot(121) ax2 = fig.add_subplot(122)  # create the data set that will be plotted x = np.linspace(0, np.pi * 2) y = np.sin(x) # y = f(x) = sin(x) z = np.cos(x) # z = g(x) = cos(x)  # plot f(x) and g(x) ax1.plot(x, y, label = 'sin', color = 'b') # 'b' stands for 'blue' ax2.plot(x, z, label = 'cos', color = 'r') # 'r' stands for 'red'  # set 'ax1' properties ax1.set_title('Trigonometric Functions - SINE') ax1.set_xlabel('Angle (in radian)') ax1.set_ylabel('Magnitude') ax1.axis([0, np.pi * 2, -1.25, 1.25]) # define the range of both axes # define ticks for each axis (3 auxiliary variables) xticks = [0, np.pi / 2, np.pi, np.pi * 1.5, np.pi * 2]                         # define the ticks values - 'X' axis xlabel = ['0', '$\\\\frac{\\\\pi}{2}$', '$\\\\pi$', '$\\\\frac{3\\\\pi}{2}$', '$2\\\\pi$'] # define the label values - 'X' axis yticks = [-1, -0.5, 0, 0.5, 1]                                                 # define the ticks values - 'Y' axis ax1.xaxis.set(ticks = xticks, ticklabels = xlabel) ax1.yaxis.set(ticks = yticks) ax1.legend(loc = 1)  # set 'ax1' properties ax2.set_title('Trigonometric Functions - COSINE') ax2.set_xlabel('Angle (in radian)') ax2.set_ylabel('Magnitude') ax2.axis([0, np.pi * 2, -1.25, 1.25]) # define the range of both axes # define ticks for each axis (3 auxiliary variables) xticks = [0, np.pi / 2, np.pi, np.pi * 1.5, np.pi * 2]                         # define the ticks values - 'X' axis xlabel = ['0', '$\\\\frac{\\\\pi}{2}$', '$\\\\pi$', '$\\\\frac{3\\\\pi}{2}$', '$2\\\\pi$'] # define the label values - 'X' axis yticks = [-1, -0.5, 0, 0.5, 1]                                                 # define the ticks values - 'Y' axis ax2.xaxis.set(ticks = xticks, ticklabels = xlabel) ax2.yaxis.set(ticks = yticks) ax2.legend(loc = 1)  plt.tight_layout() # fix the spaces between the two Axes plt.show()      Nós acabamos de criar uma Figure com dois Subplot’s dentro dela. As imagens da esquerda e da direita são definidas pelas funções seno e cosseno, respectivametne. Além disso, há algumas coisas novas aqui; por isso, os próximos parágrafos serão dedicados a esses detalhes.   Depois de importar, além do Matplotlib, o Numpy (veja aqui um pequeno tutorial sobre Numpy e Pandas), nós criamos uma Figure com um tamanho específico. Agora, utilizando esse objeto, que foi armazenado na variável fig, foram criados dois Subplot’s $-$ Importante: note que para o primeiro objeto, o valor do parâmetro foi 121(que significa 1 linha, 2 colunas e o índice 1), enquanto que para o segundo, esse número teve que ser definido como 122.   Nas linhas seguintes, a única coisa que fizemos foi criar os nossos vetores de dados x, y e z. Com o Numpy, nós geramos uma lista, atribuída a x, de valores igualmente espaçados entre $0$ e $2\\pi$; então, nas variáveis y e z, nós calculamos e salvamos os valores de seno e cosseno de x, respectivamente.   Depois disso tudo, diferentemente do primeiro exemplo, utilizamos o método plot() (ao invés de scatter()). Essa modificação faz com que o gráfico de pontos seja, agora, exibido como gráfico de linha.   Por fim, sobre as propriedades de ax1 e ax2, descreverei apenas os atributos e métodos que ainda não foram discutidos. A primeira novidade é o método axis(), que define os valores mínimo e máximo que serão mostrados em cada eixo do gráfico $-$ perceba que é necessário passar uma lista de números. Além disso, nós também criamos algumas variáveis que foram utilizadas para definir os parametros ticks e ticklabes dos métodos xaxis.set() e yaxis.set(). Esses métodos definem em quais pontos ao longo dos eixos você terá um indicador de valor.   Um pequeno detalhe é o de que foi necessário, antes de chamar o método show(), utilizar a função tight_layout() a fim de ajustar o tamanho e espaçamento dos gráficos.   Conclusão   Matplotlib é uma biblioteca muito flexível e poderosa que nos permite criar diferentes tipos de gráficos. Além disso, como mencionado anteriormente, existe uma galeria oficial cheia de exemplos reais criados a partir dessa ferramenta $-$ confira, e veja todas as possibilidades.   Qualquer dúvida, sugestão ou feedback, por favor, deixe um comentário abaixo.      Eu havia escrito esse texto, originalmente em inglês, em uma antiga versão do blog que já não existe mais. Pequenas correções e atualizações, além da própria tradução, foram feitas para que o conteúdo continuasse relevante.   ","categories": ["Tutorial"],
        "tags": ["python","ciencia-de-dados"],
        "url": "http://localhost:4000/visualizacao-de-dados-com-matplotlib/",
        "teaser":null},{
        "title": "[Parte 01] O que é Aprendizado (de Máquina) & PLA",
        "excerpt":"Quando se fala de Aprendizado de Máquina, há pelo menos duas interpretações para o que isso significa: a primeira delas diz respeito a um conjunto de técnicas e modelos estatísticos que são usados, primariamente, para se fazer inferência (estimação pontual dos parâmetros de interesse, contrução de intervalo de confiança, teste de hipótese, etc.); enquanto que a segunda abordagem, que será a utilizada daqui para frente, se preocupa em, principalmente, fazer predição sobre novas observações $-$ nesse caso, o(s) modelo(s) utilizado(s) pode(m), inclusive, não ter forma explícita (Izbicki, R.; dos Santos, T. M. Machine Learning sob a ótica estatística). Dito isso, podemos começar.   Componentes do aprendizado   A fim de dar contexto ao que vem a seguir, tome o seguinte exemplo: imagine que um banco contratou uma empresa de consultoria para ajudá-lo a determinar se, para cada cliente que pede um empréstimo, o banco deve ou não conceder esse crédito (aqui, um cliente é bom se, de alguma forma, faz com que o banco ganhe dinheiro; e é ruim, caso contrário). Para construir um modelo que resolve esse tipo de problema, o banco tem um conjunto de dados de clientes antigos, com várias de suas características (salário, estado civil, bens materiais, etc.), além de, se no passado, esses clientes fizeram a instituição ganhar ou perder dinheiro. De posse dessas informações, a empresa de consultoria pode escrever um modelo que ajuda o banco a predizer se clientes futuros darão (ou não) algum tipo de lucro.   Considerando esse exemplo, podemos dar nomes às componentes (do apredizado) de interesse. Seja $\\mathbf{x}$ o vetor de entrada (informação que o banco tem de um cliente), então $f: \\mathcal{X} \\longrightarrow \\mathcal{Y}$ é a função alvo (que é desconhecida) $-$ onde $\\mathcal{X}$ é o espaço de entrada (para o caso onde existem $d$ características sobre um cliente, $\\mathcal{X}$ é o Espaço Euclidiano $d$-dimensional) e $\\mathcal{Y}$ é o espaço de saída (no exemplo, $\\mathcal{Y} = \\lbrace +1, -1 \\rbrace$; ou seja, o banco concede ou não o empréstimo). Além disso, temos o conjunto de dados com entradas e saídas, definido por $\\mathcal{D} = (\\mathbf{x}_1, y_1), \\cdots, (\\mathbf{x}_N, y_N)$ $-$ onde $y_n = f(\\mathbf{x}_n)$, tal que $n = 1, \\cdots, N$ $-$, e um algoritmo de aprendizagem $\\mathcal{A}$ que usa $\\mathcal{D}$ para determinar uma função $g: \\mathcal{X} \\longrightarrow \\mathcal{Y}$ que aproxima $f$. Nesse caso, o algoritmo “escolhe” uma função $g$ a partir de uma classe de funções que são relevantes para o problema (esse conjunto que contempla $g$ será denotado por $\\mathcal{H}$, e receberá o nome de Hypothesis Set).   Voltando ao exemplo, o banco irá, baseado em $g$, decidir para quais clientes realizará o empréstimo (lembre-se que $f$ é desconhecida). Nesse caso, o algoritmo $\\mathcal{A}$ selecionou $g \\in \\mathcal{H}$ a partir da análise do conjunto de dados $\\mathcal{D}$; na esperança que o comportamento de clientes futuros ainda possa ser modelado por essa função escolhida. A figura a seguir ilustra a relação entre todas essas componentes.    Figura 1 [fonte: “Learning from Data”] $-$ Componentes do aprendizado.   A Fig. 1 será utilizada como framework para tratarmos o “problema do aprendizado”. Mais tarde, esse esquema vai passar por alguns refinamentos, mas a base será a mesma:           Existe uma $f$ para ser aprendida (que é, e continuará sendo desconhecida para nós).            Temos um conjunto de dados $\\mathcal{D}$ gerados por essa função alvo.            O algoritmo de aprendizagem $\\mathcal{A}$ utiliza $\\mathcal{D}$ para encontrar uma função $g \\in \\mathcal{H}$ que aproxima bem $f$.       Um modelo de aprendizado simples   Sobre as componentes do aprendizado que acabamos de discutir, a função alvo $f$ e o conjunto de dados $\\mathcal{D}$ vêm do problema com o qual estamos lidando. Entretanto, o conjunto de possíveis soluções $\\mathcal{H}$ e o algoritmo de aprendizagem $\\mathcal{A}$ são ferramentas nós temos que escolher para determinar $g$; nesse sentido, eles ($\\mathcal{H}$ e $\\mathcal{A}$) são chamados de modelo de aprendizado.   Vejamos, então, um modelo de aprendizado simples: seja $\\mathcal{X} = \\mathbb{R}^d$ e $\\mathcal{Y} = \\lbrace +1, -1 \\rbrace$; onde $+1$ e $-1$, denotam “sim” e “não”, respectivamente. No exemplo do banco, diferentes coordenadas de $\\mathbf{x} \\in \\mathcal{X}$ representam cada uma das características do cliente (salário, estado civil, bens materiais, etc.); enquanto que o espaço de saída $\\mathcal{Y}$ faz referência ao fato de o banco conceder ou não o empréstimo. Em adição, vamos dizer que $\\mathcal{H}$ é composto por todas as funções $h \\in \\mathcal{H}$ que têm forma ditada por:     onde $\\mathbf{w}$ é um vetor de “pesos” e $\\mathbf{x} \\in \\mathcal{X}$, com $\\mathcal{X} = \\lbrace 1 \\rbrace \\times \\mathbb{R}^d$. O que está sendo feito aqui é simples: a família de funções $h(\\cdot)$ $-$ perceba que $h$ não está completamente definida, já $\\textbf{w}$ não é parâmetro da função $-$, atribui pesos $w_i$ para cada uma das $d$ características dos indivíduos. Dessa forma, podemos determinar a regra de que, se $\\sum_{i = 1}^{d} w_i x_i &gt; \\text{threshold}$, então o banco aprova o empréstimo; caso contrário, não. $h(\\mathbf{x})$ traduz essa ideia; além de assumir os valores $+1$ ou $-1$, como gostaríamos que fosse.   O modelo que acabamos de descrever é chamado de perceptron. O algoritmo de aprendizagem $\\mathcal{A}$ vai procurar por valores de $\\mathbf{w}$ ($w_0$ incluído) que se adaptam bem as dados. A escolha ótima será a nossa função $g$.   Observação: se o conjunto de dados for linearmente separável, então existirá um $\\mathbf{w}$ que classifica todas as observações corretamente.   Por fim, vamos ver então qual é esse algoritmo $\\mathcal{A}$. O algoritmo de aprendizagem nesse caso é chamado de perceptron learning algorithm (PLA), e funciona como descrito abaixo.   Para encontrar $\\mathbf{w}$ tal que todos os pontos estão corretamente classificados, vamos considerar um processo de iteração em $t$ $-$ tal que $t = 0, 1, 2, \\cdots$. Nesse caso, o vetor de “pesos” no $t$-ésimo instante será denotado por $\\mathbf{w}(t)$; aqui, $\\mathbf{w}(0)$ é escolhido arbitrariamente. Para cada etapa do processo, o algoritmo seleciona uma das obervações que não está classificada corretamente $-$ vamos chamá-la de $(\\mathbf{x}(t), y(t))$ $-$, e aplica a seguinte regra:     O que a regra que acabamos de definir faz é “mover” a reta $w_0 + w_1 x_1 + w_2 x_2 = 0$ (para o caso com $2$ dimensões) que divide os pontos do conjunto de dados; a fim de classificar corretamente a observação $(\\mathbf{x}(t), y(t))$.   Como dito anteriormente, se os dados são linearmente separáveis, o PLA converge, classificando corretamente todas as observações; o que implica em duas coisas interessantes:           Repare que, para cada etapa do processo de iteração, apesar de corrigir a observação que está sendo considerada, o algoritmo pode “bagunçar” a classificação (em um primeiro momento, correta) dos outros pontos; mas mesmo assim, sob a hipótese de que os dados são linearmente separáveis, o algoritmo converge (a demonstração pode ser vista aqui).            Para esse tipo de classificador que acabamos de detalhar, $\\mathcal{H}$ é uma classe infinita de funções; assim, dizer que, nesse caso, o algorimo converge, é o mesmo que dizer que, mesmo em um conjunto de cardinalidade infinita, foi necessário uma quantidade finita de iterações para encontrarmos uma solução ótima (no sentido de classificar corretamente todos os pontos) para o problema $-$ o que é, no mínimo, interessante.       Conclusão   Na primeira parte dessa série de textos, vimos o framework básico com o qual vamos trabalhar quando queremos modelar o processo de aprendizado (de máquina). Essas ideias serão revisitadas à exaustão e, por isso, são tão importantes. Por fim, vimos também como essas componentes do apredizado podem nos guiar na construção de um modelo de classificação simples. Tal modelo, o “perceptron”, é de fato limitado; mas é um excelente primeiro passo que podemos dar. A próxima postagem será dedicada à implementação desse classificador em Python, bem como à discussão de um exemplo.   Qualquer dúvida, sugestão ou feedback, por favor, deixe um comentário abaixo.      Essa postagem faz parte de uma série de textos que tem o objetivo de estudar, principalmente, o curso “Learning from Data”, ministrado pelo Professor Yaser Abu-Mostafa. Outros materiais utilizados serão sempre referenciados.   ","categories": ["Machine Learning - Learning from Data"],
        "tags": [],
        "url": "http://localhost:4000/o-que-e-aprendizado/",
        "teaser":null}]
