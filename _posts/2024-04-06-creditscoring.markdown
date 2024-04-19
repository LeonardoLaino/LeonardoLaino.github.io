---
layout: default
title: Credit Scoring
author: Credit Scoring
date: 2024-04-06
---


<h1>Introdução</h1>

<p>Neste post eu irei explorar como um modelo estatístico pode servir para orientar decisões na <b>área de concessão de crédito</b>, mais especificamente quando falamos <b>controlar os riscos de concessão</b>. Para isso, irei resumir um projeto que resolvi recentemente (que pode ser encontrado <a href="https://github.com/LeonardoLaino/credit-scoring" target="_blank">aqui</a>), testando duas abordagens diferentes: Uma mais tradicional e outra mais branda.</p>

<p><b>Obs:</b> Os dados que irei utilizar neste projeto podem ser encontrados <a href="https://www.kaggle.com/c/home-credit-default-risk/data" target="_blank">aqui</a>.</p>

<h1>O Problema</h1>

<p>Na concessão de crédito, os bancos e instituições financeiras enfrentam o dilema de avaliar o risco associado a cada cliente. Tradicionalmente, esse processo baseava-se em análises manuais e em poucos critérios, como histórico de crédito, renda e histórico de pagamento. No entanto, essa abordagem muitas vezes resultava em decisões subjetivas, falhando na identificação de potenciais inadimplentes.</p>

<p>Historicamente a concessão de crédito passou por várias transformações: No inicio, era baseada principalmente em relações pessoais e na reputação dos clientes. Porém com passar do tempo e o aumento da complexidade dos mercados financeiros, o volume de dados cresceu exponencialmente e, consequentemente, obrigou à introdução de métodos mais estruturados de avaliação de crédito.</p>

<p>As instituições ainda enfrentam desafios na concessão de crédito, principalmente quando olhamos para a questão do equilibrio entre concessão e risco, onde quer-se aprovar o máximo possível com o menor risco associado. Outra questão associada é a <b>política de crédito</b>: Quais outros fatores precisam ser levados em consideração na hora de aprovação ou recusa? Esses fatores ajudam na inclusão de pessoas historicamente ignoradas? Vamos explorar um pouco deste assunto.</p>


<h1>1. Análise Exploratória</h1>

Aqui irei me ater à base de `application train`, que é onde temos a base de clientes.

Nessa base iremos encontrar informações gerais dos clientes (como idade, quantidade de filhos, salário, etc.), o produto que eles contrataram (`empréstimo` ou `cartão de crédito`), informações de bureaus de crédito e o nosso <b>target</b> (1 para `inadimplente/evento` ou 0 para `adimplente/não evento`).

Uma passada rápida pelos dados, podemos ter uma ideia do público que estamos lidando:

<div class="container">
    <img class= "centered-image" src="/assets/images/inadimplencia.png" alt="EDA">
</div>

Com uma ideia do público em mente, meu foco agora é tentar descobrir se alguma das <b>122</b> variáveis consegue, mesmo que sutilmente durante sua faixa de valores, separar o público adimplente do inadimplente. Em outras palavras, queremos saber <b>quais variáveis são potencialmente boas preditoras da variável dependente(<i>target</i>)</b>.

Irei adotar a seguinte abordagem:

<ol>
    <li>Para variáveis <mark style="background-color: #F5F2F0;"><code>contínuas</code></mark>: KDE Plot das Variáveis x <i>Target</i></li>
    <li>Para variáveis <mark style="background-color: #F5F2F0;"><code>categóricas</code></mark>: Gráfico de colunas empilhadas x <i>Target</i></li>
</ol>


<h2>1.1. Variáveis Contínuas</h2>

`Obs`: A curva em <font color = 'red'><b>vermelho</b></font> representa a densidade dos clientes inadimplentes, enquanto a curva <font color = 'green'><b>verde</b></font> represente a densidade dos clientes adimplentes.


<div class="container">
    <img class= "centered-image" src="/assets/images/numericas_vs_target.png" alt="EDA">
</div>


Observe o comportamento da variável `YEARS BUILD AVG` e compare-o com a variável `DAYS BIRTH`.

Enquanto na primeira as duas curvas estão praticamente sobrepostas conforme o valor da variável cresce, a segunda apresenta um desbalanceamento em uma faixa específica de valores. Isso nos revela um recorte para investigação mais profunda, onde há possibilidade de extração de mais insights. Por outro lado, também pode nos indicar que esta variável é uma boa preditora do nosso <i>target</i>.

Outras variáveis que nos chamam atenção são `EXT SOURCE 1`, `EXT SOURCE 2` e `EXT SOURCE 3`. Essas variáveis são scores de crédito provenientes de 3 bureaus de crédito distintos. Note como elas possuem alta capacidade de segmentar o <i>target</i>.


<h2>1.2. Variáveis Categóricas</h2>

Aqui foi aplicado o mesmo conceito, porém para as variáveis categóricas.


<div class="container">
    <img class= "centered-image" src="/assets/images/categoricas_vs_target.png" alt="EDA">
</div>


Não conseguimos observar nenhuma feature com desbalanceamento do <i>target</i> em um de seus domínios.

O mais próximo que podemos observar é na feature "`CODE GENDER`" (Gênero), onde os clientes de label "M" (Masculino) apresentam uma taxa de evento ligeiramente superior.

Aqui, porém, precisamos nos atentar ao próprio balanceamento dos domínios. Como vimos anteriormente, a quantidade de mulheres é quase duas vezes maior que a quantidade de homens.

Um conceito interessante que podemos aplicar aqui é o de `Information Value (IV)`, como uma forma de conseguirmos validar nossas conclusões sobre as variáveis categóricas.

Definindo-o brevemente, o `IV` nos indica o poder preditivo de uma feature em relação a um target, comparando a distribuição da taxa de evento e taxa de não evento para os valores daquela feature.

A aplicação é mais fácil para features categóricas, uma vez que não há necessidade de criar faixas de intervalos, como nas features numéricas.


O information value é calculado da seguinte forma:


$$IV = \sum(p_{evento} \times p_{não-evento}) \times WoE$$

Onde, no nosso caso:

$$p_{evento}:$$ Proporção de "1" (inadimplentes)

$$p_{não-evento}:$$ Proporção de "0" (adimplentes)

$$WoE = ln(\frac{p_{evento}}{p_{não-evento}}):$$ (Weight of Evidence)


Aqui estamos interessados em features com IV entre 0.1 e 0.5, que possuem poder preditivo moderado/forte.

IVs muito altos podem indicar features propensas a causar overfitting no modelo.

Aplicando este conceito, temos o seguinte resultado:


<table>
    <thead>
        <tr>
            <th>Feature</th>
            <th>IV</th>
            <th>Categoria IV</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>NAME_CONTRACT_TYPE</td>
            <td>0.02</td>
            <td>not useful for prediction</td>
        </tr>
        <tr>
            <td>CODE_GENDER</td>
            <td>inf</td>
            <td>suspicious predictor</td>
        </tr>
        <tr>
            <td>FLAG_OWN_CAR</td>
            <td>0.01</td>
            <td>not useful for prediction</td>
        </tr>
        <tr>
            <td>FLAG_OWN_REALTY</td>
            <td>0.00</td>
            <td>not useful for prediction</td>
        </tr>
        <tr>
            <td>NAME_TYPE_SUITE</td>
            <td>0.00</td>
            <td>not useful for prediction</td>
        </tr>
        <tr>
            <td>NAME_INCOME_TYPE</td>
            <td>inf</td>
            <td>suspicious predictor</td>
        </tr>
        <tr>
            <td>NAME_EDUCATION_TYPE</td>
            <td>0.05</td>
            <td>weak predictor</td>
        </tr>
        <tr>
            <td>NAME_FAMILY_STATUS</td>
            <td>inf</td>
            <td>suspicious predictor</td>
        </tr>
        <tr>
            <td>NAME_HOUSING_TYPE</td>
            <td>0.01</td>
            <td>not useful for prediction</td>
        </tr>
        <tr>
            <td>OCCUPATION_TYPE</td>
            <td>0.09</td>
            <td>weak predictor</td>
        </tr>
        <tr>
            <td>WEEKDAY_APPR_PROCESS_START</td>
            <td>0.00</td>
            <td>not useful for prediction</td>
        </tr>
        <tr>
            <td>ORGANIZATION_TYPE</td>
            <td>0.08</td>
            <td>weak predictor</td>
        </tr>
        <tr>
            <td>FONDKAPREMONT_MODE</td>
            <td>0.00</td>
            <td>not useful for prediction</td>
        </tr>
        <tr>
            <td>HOUSETYPE_MODE</td>
            <td>0.00</td>
            <td>not useful for prediction</td>
        </tr>
        <tr>
            <td>WALLSMATERIAL_MODE</td>
            <td>0.01</td>
            <td>not useful for prediction</td>
        </tr>
        <tr>
            <td>EMERGENCYSTATE_MODE</td>
            <td>0.00</td>
            <td>not useful for prediction</td>
        </tr>
    </tbody>
</table>

Comprovando as observações anteriores, nenhuma feature categórica se mostra capaz de prever bem o nosso target.


<h1>2. Feature Engineering</h1>

Aqui iniciamos a parte de criação do nosso book de variáveis.

Olhando os `metadados` que temos em mãos, percebemos que todas as tabelas possuem alguma forma de registrar a `data`. Isso é muito valioso, porque nos permite criar janelas temporais para tentar capturar padrões nos dados.

Outro ponto importante a se destacar é que, para esta volumetria de dados, utilizar o Pandas está fora de cogitação. Para esta etapa, irei utilizar o `PySpark` e o `Spark SQL`. Irei passar brevemente pelo processo de desenvolvimento.

Iniciando com o básico, aqui eu carrego os dados e as bibliotecas que irei utilizar. Neste exemplo, estou manipulando a base de cartões de crédito (`credit card balance`).


<div style="text-align: center; overflow-x: auto;">
  <pre class="language-python"><code>
  # Carregando as bibliotecas
  import os
  import findspark
  findspark.init()
  from pyspark.sql import SparkSession
  spark = SparkSession.builder \
      .appName("FeatureEng") \
      .config("spark.executor.memory", "14g") \
      .config("spark.driver.memory", "14g") \
      .getOrCreate()
  from warnings import filterwarnings
  filterwarnings('ignore')

  # Carregando a Tabela
  credit_balance = spark.read.csv(
      './DATASETS/credit_card_balance.csv', 
      header= True,
      inferSchema= True
  )

  # Checando as Dimensões
  (credit_balance.count(), len(credit_balance.columns))
  >> (3.840.312, 23)

  # Criando uma View (Para usar o SQL)
  credit_balance.createOrReplaceTempView('credit_balance')
  </code></pre>
</div>


Com os dados devidamente carregados, eu vou criar `flags temporais` que futuramente me permitirão fazer agregações em janelas de tempo específicas. 

Para esta tabela em específico, a coluna que nos indica a data dos registros é a `MONTHS_BALANCE`. Em casos de crédito como este, geralmente, agregar transações dos últimos **3**, **6**, **9** e **12** meses já é de grande valor.


<div style="text-align: center; overflow-x: auto;">
  <pre class="language-python"><code>
temp01 = spark.sql("""
SELECT
    *,
        CASE
            WHEN MONTHS_BALANCE >= -3 THEN 1
        ELSE 0
    END AS FL_U3M,
        CASE
            WHEN MONTHS_BALANCE >= -6 THEN 1
        ELSE 0
    END AS FL_U6M,
        CASE
            WHEN MONTHS_BALANCE >= -9 THEN 1
        ELSE 0
    END AS FL_U9M,
        CASE
            WHEN MONTHS_BALANCE >= -12 THEN 1
        ELSE 0
    END AS FL_U12M
FROM
    credit_balance
ORDER BY
    `SK_ID_PREV`;
""")

# Atualizando a View
temp01.createOrReplaceTempView('temp01')
  </code></pre>
</div>

Com as flags criadas e munido de funções básicas de agregação, diminuí a granularidade da tabela para <b>nível cliente</b>.


<div style="text-align: center; overflow-x: auto;">
  <pre class="language-python"><code>
# Importando as funções de Agregação
from pyspark.sql.functions import when, min, max, sum, round, col, median

# Selecionando as variáveis que serão agregadas (exceto Flags e IDs)
agg_cols = [col for col in temp02.columns if ("FL_" not in col) & ("SK_ID" not in col)]

# Removendo a Coluna de Janela Temporal
agg_cols.remove('MONTHS_BALANCE')

# Selecionando as flags temporais
flags_temporais = ['FL_U3M', 'FL_U6M', 'FL_U9M','FL_U12M']

new_cols = []

for flag_temp in flags_temporais:

    for agg_col in agg_cols:

        if 'DPD' in agg_col:
            new_cols.append(round(max(when(col(flag_temp) == 1, col(agg_col))),2).alias(f"QT_MAX_{agg_col}_{flag_temp}_CREDITCARDBALANCE"))
            new_cols.append(round(min(when(col(flag_temp) == 1, col(agg_col))),2).alias(f"QT_MIN_{agg_col}_{flag_temp}_CREDITCARDBALANCE"))
        else:
            new_cols.append(round(max(when(col(flag_temp) == 1, col(agg_col))),2).alias(f"VL_MAX_{agg_col}_{flag_temp}_CREDITCARDBALANCE"))
            new_cols.append(round(min(when(col(flag_temp) == 1, col(agg_col))),2).alias(f"VL_MIN_{agg_col}_{flag_temp}_CREDITCARDBALANCE"))
            new_cols.append(round(sum(when(col(flag_temp) == 1, col(agg_col))),2).alias(f"VL_SUM_{agg_col}_{flag_temp}_CREDITCARDBALANCE"))
            new_cols.append(round(median(when(col(flag_temp) == 1, col(agg_col))),2).alias(f"VL_MD_{agg_col}_{flag_temp}_CREDITCARDBALANCE"))

new_cols = tuple(new_cols)

# Unpacking
temp02 = temp01.groupBy("SK_ID_PREV").agg(*new_cols).orderBy("SK_ID_PREV")
  </code></pre>
</div>

Com este processo, podemos obter um total de ``289 variáveis``!

Há possibilidade, também, de criar `flags categóricas` e gerar mais variáveis combinando-as com as <b>flags temporais</b>. Em uma primeira simulação, consegui expandir o total de variáveis para `793`.

No entando, devido a limitações computacionais (e de tempo), não irei estressar muito este processo.

Após aplicar os mesmos conceitos nas tabelas disponíveis, e conectando-as devidamente, obtive uma tabela com `10.964 variáveis`.

<div style="text-align: center; overflow-x: auto;">
  <pre class="language-python"><code>
# Tabela com os dados do Bureau, após feature engineering
bureau = spark.read.parquet('./BASES_TREINO_TESTE/BUREAU_FEAT_ENG')

# Tabela com os dados de Previous Application (POSCASH Balance, Instalments Payments e Credit Card Balance)
# Após feat. eng.
prev_app = spark.read.parquet('./BASES_TREINO_TESTE/PREV_APP_AGG_FEAT_ENG')

# Base de Treino (Application Train)
app_train = spark.read.csv('./DATASETS/application_train.csv', header= True, inferSchema= True)

# Realizando os Joins
app_train_final = app_train.join(other= prev_app, on= "SK_ID_CURR", how= 'left').join(other= bureau, on= "SK_ID_CURR", how= 'left')

# Observando a dimensão final da tabela após os joins
display(app_train_final.count(), len(app_train_final.columns))
>> (215.257, 10.964)
  </code></pre>
</div>


Com essa tabela criada, vou fazer uma feature selection simples para facilitar a persistência desses dados em disco.


Para esta etapa, irei remover colunas com alta taxas de nulos e features que possuam alta correlação entre si. Também será necessário adaptar este processo para a volumetria de dados que estou lidando, uma vez que esta tabela não pode ser carregada pelo Pandas. Com isso em mente, irei criar lotes de N variáveis e aplicar as funções de feature selection. Ao final deste procedimento, devemos ter uma quantidade razoável de variáveis para as próximas operações.


<div style="text-align: center; overflow-x: auto;">
  <pre class="language-python"><code>
from tqdm import tqdm
# Carregando as funções de feature selection
from feature_selection_functions import amostragem, remove_highly_correlated_features, generate_metadata, variancia, vars_selection

# Função para criar lotes de colunas
def criar_lotes(lista, tamanho):
    
    qtd = len(lista) // tamanho
    residuo = len(lista) % tamanho
    lotes = []
    slice_start = 0
    slice_end = tamanho

    for _ in range(1,qtd+2):
        if _ == (qtd+2):
            lotes.append(lista[len(lista) - residuo:len(lista)-1])
            break

        lotes.append(lista[slice_start:slice_end])
        slice_start+= tamanho
        slice_end+= tamanho
    
    return lotes

lotes = criar_lotes(app_train_final_cols, 500)

variaveis_selecionadas = []

for lote in tqdm(lotes):
    lote.append('SK_ID_CURR')
    lote.append('TARGET')

    temp01 = app_train_final.select(*lote).toPandas()

    selecao = vars_selection(temp01, percentual_preenchimento= 80, threshold= 0.5, tamanho_amostragem= 90000)

    if not selecao.empty:
        for variavel in selecao['Variável']:
            variaveis_selecionadas.append(variavel)
    else:
        pass
  </code></pre>
</div>

Após o término de execução, reduzimos a quantidade de variáveis de `10.963` para `218`.


<h1>3. Modelagem Estatística</h1>

Nesta seção irei criar e avaliar dos modelos estatísticos. O critério de sucesso  consiste em assegurar que o modelo seja capaz de classificar os clientes em faixas de score, permitindo a distinção entre bons e maus pagadores. Este processo é fundamental para que a área de negócios consiga controlar o risco, mantendo o maior valor possível de aprovação.

Não irei entrar nas etapas de `data preparation` e `feature selection`, apenas irei menciona-las supercialmente a seguir:

1. <b><i>Missings</i></b>: Preenchimento por Mediana para variáveis contínuas e valores mais frequentes para variáveis categóricas;

2. <b><i>Transformações</i></b>: Padronização para as variáveis contínuas e Label Encoder para as variáveis categóricas;

3. <b><i>Feature Selection</i></b>: Novamente removi variáveis com alta correlação. Para a Regressão Logística, selecionei as 20 melhores variáveis via Feature Importance com XGBoost. Para o LGBM, utilizei todas as variáveis.


<h2>3.1. Baseline</h2>
Aqui irei testar alguns modelos apenas com os dados de treino (conforme fornecido no problema), com intuito de estabelecer uma referencia de desempenho.

<div style="text-align: center; overflow-x: auto;">
  <pre class="language-python"><code>
from model_metrics_functions import plot_metrics, calculate_metrics

models = [
    DecisionTreeClassifier(criterion= 'gini', random_state=1),
    LogisticRegression(solver= 'liblinear', random_state=1),
    RandomForestClassifier(random_state=1),
    GradientBoostingClassifier(random_state=1),
    XGBClassifier(random_state=1),
    lgb.LGBMClassifier(random_state=1)
]

for model in models:
    model_name = str(model)[:str(model).find("(")]
    # Treinamento
    model.fit(X_train_processed, y_train)

    # Avaliação
    metrics = calculate_metrics(model_name, model, X_train_processed, y_train, X_test_processed, y_test)
    display(metrics)
  </code></pre>
</div>

Após o término da execução, estes foram os resultados:

<table border="1">
  <tr>
    <th>Algoritmo</th>
    <th>Conjunto</th>
    <th>Acuracia</th>
    <th>Precisao</th>
    <th>Recall</th>
    <th>AUC_ROC</th>
    <th>GINI</th>
    <th>KS</th>
  </tr>
  <tr>
    <td>DecisionTreeClassifier</td>
    <td>Treino</td>
    <td>1.000000</td>
    <td>1.000000</td>
    <td>1.000000</td>
    <td>1.000000</td>
    <td>1.000000</td>
    <td>1.000000</td>
  </tr>
  <tr>
    <td>DecisionTreeClassifier</td>
    <td>Teste</td>
    <td>0.851993</td>
    <td>0.134017</td>
    <td>0.157772</td>
    <td>0.534861</td>
    <td>0.069722</td>
    <td>0.070959</td>
  </tr>
</table>

<br>

<table border="1">
  <tr>
    <th>Algoritmo</th>
    <th>Conjunto</th>
    <th>Acuracia</th>
    <th>Precisao</th>
    <th>Recall</th>
    <th>AUC_ROC</th>
    <th>GINI</th>
    <th>KS</th>
  </tr>
  <tr>
    <td>RandomForestClassifier</td>
    <td>Treino</td>
    <td>0.999973</td>
    <td>1.0</td>
    <td>0.999674</td>
    <td>1.000000</td>
    <td>1.000000</td>
    <td>1.000000</td>
  </tr>
  <tr>
    <td>RandomForestClassifier</td>
    <td>Teste</td>
    <td>0.920499</td>
    <td>0.0</td>
    <td>0.000000</td>
    <td>0.692609</td>
    <td>0.385219</td>
    <td>0.285866</td>
  </tr>
</table>

<br>

<table border="1">
  <tr>
    <th>Algoritmo</th>
    <th>Conjunto</th>
    <th>Acuracia</th>
    <th>Precisao</th>
    <th>Recall</th>
    <th>AUC_ROC</th>
    <th>GINI</th>
    <th>KS</th>
  </tr>
  <tr>
    <td>GradientBoostingClassifier</td>
    <td>Treino</td>
    <td>0.919265</td>
    <td>0.687708</td>
    <td>0.016859</td>
    <td>0.766100</td>
    <td>0.532200</td>
    <td>0.396157</td>
  </tr>
  <tr>
    <td>GradientBoostingClassifier</td>
    <td>Teste</td>
    <td>0.920840</td>
    <td>0.593220</td>
    <td>0.013635</td>
    <td>0.751106</td>
    <td>0.502211</td>
    <td>0.373384</td>
  </tr>
</table>

<br>

<table border="1">
  <tr>
    <th>Algoritmo</th>
    <th>Conjunto</th>
    <th>Acuracia</th>
    <th>Precisao</th>
    <th>Recall</th>
    <th>AUC_ROC</th>
    <th>GINI</th>
    <th>KS</th>
  </tr>
  <tr>
    <td>XGBClassifier</td>
    <td>Treino</td>
    <td>0.929147</td>
    <td>0.956670</td>
    <td>0.136667</td>
    <td>0.908577</td>
    <td>0.817155</td>
    <td>0.650010</td>
  </tr>
  <tr>
    <td>XGBClassifier</td>
    <td>Teste</td>
    <td>0.919895</td>
    <td>0.452323</td>
    <td>0.036034</td>
    <td>0.731318</td>
    <td>0.462636</td>
    <td>0.347898</td>
  </tr>
</table>

<br>

<table border="1">
  <tr>
    <th>Algoritmo</th>
    <th>Conjunto</th>
    <th>Acuracia</th>
    <th>Precisao</th>
    <th>Recall</th>
    <th>AUC_ROC</th>
    <th>GINI</th>
    <th>KS</th>
  </tr>
  <tr>
    <td>LGBMClassifier</td>
    <td>Treino</td>
    <td>0.920407</td>
    <td>0.862595</td>
    <td>0.027610</td>
    <td>0.833379</td>
    <td>0.666758</td>
    <td>0.508259</td>
  </tr>
  <tr>
    <td>LGBMClassifier</td>
    <td>Teste</td>
    <td>0.920747</td>
    <td>0.559701</td>
    <td>0.014608</td>
    <td>0.750472</td>
    <td>0.500945</td>
    <td>0.376406</td>
  </tr>
</table>

Os modelos baseados em <i>bagging</i> sofreram de overfitting, enquanto os modelos baseados em <i>boosting</i> (como o `LGBM` e o `GradientBoosting`) se saíram bem melhor.

<h2>3.2. Regressão Logística</h2>

<h3>3.2.1. Um pouco sobre a Regressão Logística</h3>

A regressão logística é um dos modelos mais conhecidos e utilizados no setor de crédito devido à sua grande estabilidade e facilidade de interpretação. Por meio dos coeficientes $$\beta$$ e seus respectivos $$p_{valor}$$, é possível compreender como o aumento ou diminuição de uma variável afeta o <i>score</i> de crédito.


A ideia da regressão logística é modelar a relação entre uma variável `dependente` binária (neste caso, adimplente/inadimplente ou 0/1) e $$n$$ variáveis `independentes`. Além disso, ela é capaz de fornecer a `probabilidade` de cada observação pertencer a uma determinada classe.


A cara da equação é a seguinte:

$$ln(\frac{p}{1-p}) = \beta_{0} + \beta_{1}x_{1} + \beta_{2}x_{2} + ... + \beta_{n}x_{n}$$

<b>Onde</b>:

$$p:$$ é a probabilidade de evento (ou de pertencer a uma classe);

$$\frac{p}{1-p}:$$ é a razão das probabilidades (odds);

$$\beta_{0}:$$ é o intercepto;

$$\beta_{1}, \beta_{2}, ..., \beta_{n}:$$ são os coeficientes que representam o efeito das variáveis independentes $$x_{1}, x_{2}, ..., x_{n}$$ sobre a log-odds do evento ocorrer.

Os $$\beta$$ são calculados maximizando a equação da `verossimilhança` usando técnicas de aproximação numérica (Como <b> Newton-Raphson</b> ou <b>Gradiente Descendente</b>). De forma resumida, chuta-se um valor inicial pro Beta, calcula-se o valor da função, atualiza-se o valor de $$\beta$$ e o processo é repetido até convergir em um valor máximo da função.

Após o processo de cálculo dos $$\beta$$, tem-se um valor $$z$$ (conhecido como ``logito`` ou `log-odds`) que representa a equação linear $$\beta_{0} + \sum_{n=1}^{N}(\beta_{n}x_{n})$$.

Para calcular a probabilidade, utilizamos a função sigmoide (ou função logística), da seguinte forma:

$$p = \frac{1}{1+e^{-z}}$$

Que, em síntese, transforma o valor de $$z$$ em um valor entre $$0$$ e $$1$$.


<h3>3.2.2. O Modelo</h3>
TBA

<h2>3.3. LightGBM</h2>
TBA