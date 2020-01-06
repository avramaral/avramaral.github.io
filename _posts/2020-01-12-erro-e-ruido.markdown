---
title: "[Parte 07] Erro e Ruído"
categories: [Aprendizado de Máquina - Learning from Data]
tags: [aprendizado-de-maquina]
---


Formalizando alguns conceitos sobre os quais já falamos ao longo [dessa série de textos]({{ site.baseurl }}/categories/aprendizado-de-máquina-learning-from-data/), essa postagem será dedicada à discussão de: 1) Como quantificar o quão bem a hipótese final $g \in \mathcal{H}$ se aproxima da função alvo $f$? 2) Como lidar com o "ruído" associado à $f$?

### Medidas de Erro

Dando sequência às ideias sobre as quais começamos a discutir na [parte 03](/memorizar-nao-e-aprender/), e relembrando que $g$ é apenas uma aproximação de $f$, é importante que sejamos capazes de definir uma medida de erro para essa diferença. Vamos, então, começar por definir "erro" da seguinte forma:

$$
\begin{align*}
\text{Erro} = E(h, f).
\end{align*}
$$

Aqui, note que o erro é uma função de cada uma das possíveis $h \in \mathcal{H}$ e de $f$. Mais explicitamente, dado um ponto $\mathbf{x}$, podemos definir $e(h(\mathbf{x}), f(\mathbf{x}))$ como uma medida pontual de erro. Assim, $E$ será determinado pela média dos erros pontuais $e$'s, para todo $\mathbf{x} \in \mathcal{D}$. 

Nesse sentido, temos como exemplo: ${}^{*)}$ $e(h(\mathbf{x}), f(\mathbf{x})) = \mathbb{I}_{\lbrace h(\mathbf{x}) \neq f(\mathbf{x}) \rbrace}$, quando falamos de classificadores; ou ${}^{**)}$ $e(h(\mathbf{x}), f(\mathbf{x})) = (h(\mathbf{x}) - f(\mathbf{x}))^2$, quando estudamos o modelo de regressão.

Um ponto importante a se discutir é que, em uma situação ideal, a medida de erro ($E$) deve ser escolhida de acordo com o problema com o qual se está trabalhando. Entretanto, isso nem sempre é praticável $-$ pode acontecer de a função erro (ou "função custo", ou "função perda") especificada ser de difícil otimização (no sentido matemático da palavra; i.e., pode não ser possível determinar o vetor de pesos $\mathbf{w}$ que minimiza essa medida).

### Ruído

Em situações reais, é possível que não sejamos capazes de obter toda a informação que determina unicamente, através da função $f$, o valor do ponto $\mathbf{x} \in \mathcal{X}$; ou seja, existem características que não são observáveis, e, portanto, não podem ser **explicitamante** incluídas no modelo que escrevemos. Assim, temos que acomodar no nosso *framework* essa diferença, denomimanda por **ruído**, entre os valores observado e real (não observável) de $y$.

Assim, ao invés de definirmos $y = f(\mathbf{x})$, podemos enxergar $y$ como variável aleatória. Formalmente, ao invés de uma função alvo $f$, teremos uma distribuição alvo $P(y \mid \mathbf{x})$. Dessa forma, um ponto ($\mathbf{x}$, y) é gerado a partir da distribuição conjunta $P(\mathbf{x}, y) = P(\mathbf{x}) P(y \mid \mathbf{x})$. Seguindo esse raciocínio, teremos que $y = f(\mathbf{x}) + \epsilon$; onde, para o modelo de regressão, $f(\mathbf{x}) = \mathbb{E}(y \mid \mathbf{x})$. Aqui, chamaremos de $\epsilon$ de "ruído", mas existem outros nomes que podem ser encontrados na literatura ("erro", por exemplo).

O esquema abaixo introduz essa nova compomente $-$ distribuição alvo $-$, bem como a medida de erro (discutida no começo do texto) no diagrama de aprendizado que estamos construindo.

![Framework completo de aprendizado]({{ site.baseurl }}/assets/images/erro-e-ruido_files/framework-completo-aprendizado.png)
*Figura 1 \[fonte: "[Learning from Data](http://www.work.caltech.edu/textbook.html)"\] $-$ Framework completo de aprendizado.*

### Observações importantes

Note que existe uma diferença no papel das distribuição $P(y \mid \mathbf{x})$ e $P(\mathbf{x})$ em relação ao problema do aprendizado que temos discutido até agora. A primeira distruição é a que estamos tentanto aprender, ao passo que a segunda diz respeito à quantificação do quão bem estamos aprendendo (como explicado em detalhes na [parte 03](/memorizar-nao-e-aprender/)).

Por fim, podemos nos perguntar se aprender a distribuição alvo é tão fácil quanto aprender a função alvo. E a resposta é: "não necessariamente". Perceba que a aproximação de $E_{out}$ por $E_{in}$ ainda pode ser igual para os dois cenário (a construção que fizemos na [parte 03](/memorizar-nao-e-aprender/) continua válida); entretanto, o erro amostral ($E_{in}$) pode ser potencialmente maior $-$ pela dificuldade de incorporar o ruído no modelo ajustado. 

## Conclusão

Nesse post, discutimos com um pouco mais de profundidade a quantificação que podemos fazer sobre o quão bem uma função $g \in \mathcal{H}$ pode aproximar $f$; além de como medidas de erro podem ser definidas nesse contexto. Em adição, incorporamos o conceito de **ruído** ao nosso *framework* de aprendizado, nos permitindo incluir no nosso modelo a diferença entre os valores observado e real (não observável) da variável dependente. Nos próximos textos, vamos generalizar vários dos conceitos que estudamos até agora.

Qualquer dúvida, sugestão ou *feedback*, por favor, deixe um comentário abaixo.

> Essa postagem faz parte de uma [série]({{ site.baseurl }}/categories/aprendizado-de-máquina-learning-from-data/) de textos que tem o objetivo de estudar, principalmente, o curso "[Learning from Data](http://www.work.caltech.edu/telecourse.html)", ministrado pelo Professor Yaser Abu-Mostafa. Outros materiais utilizados serão sempre referenciados.