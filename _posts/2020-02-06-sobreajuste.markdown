---
title: "[Parte 14] Sobreajuste (ou Overfitting)"
categories: [Machine Learning - Learning from Data]
---


Continuando com a nossa [série de textos]({{ site.baseurl }}/categorias/#machine-learning-learning-from-data), e agora falando sobre um assunto um pouco diferente do que vínhamos discutindo até então; vamos tratar do problema de **sobreajuste** (ou *overfitting*, do inglês $-$ como é mais conhecido). Veremos como a ideia de "ruído" (apresentada na [parte 07](/erro-e-ruido/)) é causa direta desse tipo de fenômento e vamos, ainda, introduzir um novo conceito, o de "ruído determinístico". 

***Overfitting*** pode ser entendido como o fenômeno no qual um bom ajuste do modelo escolhido com dados pertencentes ao conjunto de trainamento NÃO se traduz em $E_{out}$ proporcionalmente pequeno; na verdade, o contrário pode acontecer: o erro fora da amostra pode aumentar conforme o erro dentro da amostra diminui. 

Para conseguirmos enxergar o que definimos como *overfitting* acontecendo, considere o exercício a seguir. 

**Exemplo:** trabalhando com conjunto de dados em $1$ dimensão e ajustando modelos da classe de regressão polinomial (esse é só um nome diferente para uma regressão linear com uma transformação do tipo $x \mapsto (1, x, x^2, \cdots)$ para dados unidimensionais), vamos definir dois cenários:

1. A função alvo é um polinômio de ordem $10$ **com** ruído associado. Aqui, $\mathcal{D}$ contém 15 pontos.
2. A função alvo é um polinômio de ordem $50$ **sem** ruído associado. Aqui, $\mathcal{D}$ também contém 15 pontos.

A figura a seguir ilustra o que acabamos de descrever:

![Ilustração do exemplo]({{ site.baseurl }}/assets/images/sobreajuste_files/exemplo_inicial.png)
*Figura 1 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Ilustração do exemplo.*

Agora suponha que, para lidar com essas funções alvo, vamos ajustar dois modelos de regressão: um de ordem $2$ e outro de ordem $10$.

![Modelos ajustados]({{ site.baseurl }}/assets/images/sobreajuste_files/dados_ajustados.png)
*Figura 2 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Modelos ajustados.*

Em seguida, vamos ver como os erros ($E_{in}$ e $E_{out}$) se comportam para esses dois modelos ajustados. **Observação:** lembre-se de que, como ainda estamos trabalhando com modelos de regressão, a medida de erro mais utilizada é o "erro quadrático", como vimos pela primeira vez na [parte 05](/modelo-de-regressao-linear/).

![Tabela de erros]({{ site.baseurl }}/assets/images/sobreajuste_files/tabela_erros.png)
*Figura 3 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Tabela de erros para todos os cenários considerados.*

Vamos, primeiro, analisar a situação na qual a função alvo é um polinômio de grau $10$ **com** ruído. Nesse caso, note que o modelo mais simples (polinômio de grau $2$) teve erro fora da amostra ($$E_{out}$$) menor $-$ se comparado ao modelo ajustado com polinômio de grau $10$. Isso aconteceu porque, apesar de o conjunto de hipóteses $$\mathcal{H}_{10}$$ conter a função alvo (**a menos do ruído**), a quantidade de dados de treinamento não foi suficiente para permitir generalização; sendo assim, o modelo apenas "memorizou" $\mathcal{D}$ e, por isso, teve $E_{in}$ muito pequeno. Dizemos que, aqui, houve **sobreajuste**. Observe, por fim, que, contrário à ideia de que "mais informações" se traduz em um modelo melhor, vimos que um modelo mais simples apresentou erro fora da amostra (que, no final das contas, é o que importa) bem mais interessante. 

Agora, vamos olhar para a situação onde a função alvo é um polinômio de grau $50$ **sem** ruído. Aqui, mais uma vez, o modelo mais complexo $-$ o polinômio de grau $10$ - teve erro dentro da amostra menor; porém, perdeu "muito feio" para o desempenho do modelo mais simples para $x \not\in \mathcal{D}$. Nesse caso, também houve **sobreajuste**. Entretanto, a razão aqui é outra: no primeiro cenário, onde existia ruído associado a $f$, o que aconteceu foi que o modelo com mais parâmetros incorporou $\epsilon$ como parte do que seria a função alvo (uma amostra muito maior preveniria esse comportamento); agora, nesse segundo cenário, não há ruído. Sendo assim, o que aconteceu? A resposta é que $f$ é muito mais complexa que os dois possíveis conjuntos de hipóteses ($\mathcal{H_2}$ e $$\mathcal{H}_{10}$$); dessa forma, o algoritmo $\mathcal{A}$ tenta usar $\mathcal{H}_{10}$ para  modelar uma função que ele não é capaz, e, por isso, acaba "memorizando" os dados ao invés de, de novo, "aprender".

A ideia aqui é que o *overfitting* pode estar relacionado a, principalmente, duas coisas: o nível de ruído (que denotaremos por $\sigma^2$) e a complexidade da função alvo (que denotaremos por $Q_f$). Ao primeiro distúrbio, danos o nome de ***ruído estocástico*** (**não há nada de novo aqui**, estamos apenas utilizando um termo maior para falar do mesmo "ruído" que temos considerado até então); ao passo que, ao segundo, damos o nome de ***ruído determinístico***. Em ambos os casos, $g \in \mathcal{H}$ perde o poder de generalização para dados fora da amostra.

Dito isso, podemos tentar estebeler uma **medida de sobreajuste**. Nesse sentido, defina:

$$
\begin{align}
\text{Medida de sobreajuste } (\mathcal{M}_s) = E_{out}(g_{10}) - E_{out}(g_{2}),
\end{align}
$$

onde $g_{10}$ e $g_{2}$ podem ser substituídas pelas funções (do tipo: $g \in \mathcal{H}$) escolhidas pelo modelo mais complexo e pelo modelo mais simples, respectivamente. 

Assim, se $\mathcal{M}_s$ for positivo, quer dizer que o modelo mais simples ganha (que, em outras palavras, é o mesmo que dizer que o modelo complexo generaliza mal dados fora de $\mathcal{D}$); e o inverso acontece no caso de $\mathcal{M}_s$ ser negativo. Perceba que essa medida é fundamentalmente de comparação. 

A imagem a seguir apresenta o resultado de um processo de simulação que estuda essas quantidades. A explicação do procedimento vem imediatamente abaixo.

![Simulação para a quantidade "medida de sobreajuste"]({{ site.baseurl }}/assets/images/sobreajuste_files/heat_map.png)
*Figura 4 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Simulação para a quantidade "$E_{out}(g_{10}) - E_{out}(g_{2})$".*

No mapa de calor da esquerda, vemos como a medida de sobreajuste (definida pela cor) depende de $\sigma^2$ e do tamanho da amostra (com $Q_f = 20$ fixo). Perceba que, nesse caso, se o ruído estocástico aumenta e o $N$ é pequeno, modelos mais complexos (nesse caso, $\mathcal{H}_{10}$) tem desempenho ruim; entretanto, se o tamanho da amostra aumenta, esse efeito é corrigido.

Já no mapa de calor da direita, é possível enxergar como a medida de sobreajuste depende da complexidade da função alvo e, mais uma vez, do tamanho da amostra (com $\sigma^2 = 0.1$ fixo). Note que, quando $Q_f > 10$, a classe de modelos polinomiais de ordem $10$ começa a perder capacidade de aprendizado, abrindo espaço para que o modelo mais simples tenha melhor desempenho no que se diz respeito a $E_{out}$. Isso, de novo, é corrigido com o aumento do tamanho da amostra.

Aqui, é possível perceber o porquê do "ruído determinístico" receber esse nome. Quando a classe de modelos passa a não ser capaz mais de representar adequadamente a função alvo, a porção de dados não explicada pelo modelo é tratada como uma espécie de ruído.

### *Overfitting* e o *trade-off* entre viés e variância

Antes de finalizarmos, vamos ver como a questão do "*trade-off* entre viés e variância" se relaciona com as quantidades que acabamos de estudar. Relembrando o que foi apresentado na [parte 10](/vies-variancia-tradeoff/), podemos decompor o valor esperado do erro fora da amostra em duas componentes: viés e variância. A equação a seguir representa essa relação:

$$
\begin{align*}
\mathbb{E}_{\mathcal{D}}\left[E_{out}(g^{(\mathcal{D})})\right] = \text{Var} + \text{Viés},
\end{align*}
$$

onde $$\text{Viés} = \mathbb{E}_{\mathbf{x}}(\bar{g}(\mathbf{x}) - f(\mathbf{x}))^2$$ e que $$\text{Var} = \mathbb{E}_{\mathcal{D},\mathbf{x}}\left[( g^{(\mathcal{D})}(\mathbf{x}) - \bar{g}(\mathbf{x}))^2\right]$$.

Lembre-se, porém, de que na equação acima, **não** consideramos que existia ruído. Alternativamente, se agora dissermos que $y = f(x) + \epsilon$ (tal que $\mathbb{E}(\epsilon) = 0$ e $\mathbb{V}(\epsilon) = \sigma^2$), então, de maneira análoga, podemos deduzir que

$$
\begin{align*}
\mathbb{E}_{\mathcal{D}, \epsilon}\left[E_{out}(g^{(\mathcal{D})})\right] = \text{Var} + \text{Viés} + \sigma^2,
\end{align*}
$$

com $$\sigma^2 = \mathbf{E}_{\epsilon,\mathbf{x}}\left[(\epsilon(\mathbf{x}))^2\right]$$.


Da equação acima, perceba que o $$\text{Viés} = \mathbb{E}_{\mathbf{x}}(\bar{g}(\mathbf{x}) - f(\mathbf{x}))^2$$ pode ser visto como o que chamamos de "ruído determinístico" $-$ à medida que essa quantidade captura a inabilidade do modelo de aproximar a função alvo $f$. Aqui, lembre-se de que $$\bar{g}(\mathbf{x}) = \mathbb{E}_{\mathcal{D}}\left[g^{(\mathcal{D})}(\mathbf{x})\right]$$.

No final das contas, o que toda essa última seção quer dizer é que, similarmente ao que já fizemos antes, o valor esperado para o erro fora da amostra pode ser decomposto em:

- "Ruído determinístico" e "ruído aleatório" que, dado um conjunto de hipóteses $\mathcal{H}$, são quantidades fixas; e
- "$\text{Var}$", que é afetada indiretamente pelos dois tipos de ruídos - no sentido de que o modelo torna-se mais suscetível às variações advindas das componentes de ruído.

## Conclusão

Nesse post, introduzimos a ideia de **sobreajuste**, bem como quais quantidades estão relacionadas a esse fenômeno: o "ruído estocástico" e o "ruído determinístico" (que tem ligação direta com a complexidade da função alvo). Via de regra, vimos que: ${}^{1)}$ se o tamanho de $N$ cresce, então $\mathcal{M}_s$ decresce, ${}^{2)}$ se o ruído estocástico aumenta, $\mathcal{M}_s$ também assume valores maiores; e, por fim, ${}^{3)}$ se $f$ é arbitrariamente complexa, então $\mathcal{M}_s$ é potencialmente maior (vide Fig. 4). Esse, como deve ter ficado bem claro no ponto do texto em que estamos, é um problema bastante importante no que se diz respeito ao poder de generalização dos nossos modelos $-$ e, por isso, é importante estudarmos formas de mitigá-lo. A próxima postagem fará isso, apresentando a técnica de regularização.


Qualquer dúvida, sugestão ou *feedback*, por favor, deixe um comentário abaixo.

> Essa postagem faz parte de uma [série]({{ site.baseurl }}/categorias/#machine-learning-learning-from-data) de textos que tem o objetivo de estudar, principalmente, o curso "[Learning from Data](http://www.work.caltech.edu/telecourse.html)", ministrado pelo Professor Yaser Abu-Mostafa. Outros materiais utilizados serão sempre referenciados.