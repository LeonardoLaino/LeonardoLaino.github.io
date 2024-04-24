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

Outras variáveis que nos chamam atenção são `EXT SOURCE 1`, `EXT SOURCE 2` e `EXT SOURCE 3`. Essas variáveis são <i>scores</i> de crédito provenientes de 3 bureaus de crédito distintos. Note como elas possuem alta capacidade de segmentar o <i>target</i>.


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

Nesta seção irei criar e avaliar dos modelos estatísticos. O critério de sucesso  consiste em assegurar que o modelo seja capaz de classificar os clientes em faixas de <i>score</i>, permitindo a distinção entre bons e maus pagadores. Este processo é fundamental para que a área de negócios consiga controlar o risco, mantendo o maior valor possível de aprovação.

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

<h3>3.2.1. Sobre a Regressão Logística</h3>

Antes de partir para a modelagem em si, vou entrar um pouco nos bastidores da regressão logística.


A regressão logística é um dos modelos mais conhecidos e utilizados no setor de crédito devido à sua grande estabilidade e facilidade de interpretação. Por meio dos coeficientes $$\beta$$, é possível compreender como o aumento ou diminuição de uma variável afeta o <i>score</i> de crédito.


A ideia da regressão logística é modelar a relação entre uma variável `dependente` binária (neste caso, adimplente/inadimplente ou 0/1) e $$n$$ variáveis `independentes`. Além disso, ela é capaz de fornecer a `probabilidade` de cada observação pertencer a uma determinada classe, o que a torna uma ferramenta muito poderosa.


A Regressão Logística pode ser escrita da seguinte maneira:

$$ln(\frac{p}{1-p}) = \beta_{0} + \beta_{1}x_{1} + \beta_{2}x_{2} + ... + \beta_{n}x_{n}$$

<b>Onde</b>:

$$p:$$ é a probabilidade de evento (ou de pertencer a uma classe);

$$\beta_{0}:$$ é o intercepto;

$$\beta_{1}, \beta_{2}, ..., \beta_{n}:$$ são os coeficientes que representam o efeito das variáveis independentes $$x_{1}, x_{2}, ..., x_{n}$$ sobre a log-odds do evento ocorrer.


O método para encontrar os valores de $$\beta$$ ótimos envolve otimizar uma função de custo através de aproximações numéricas, aplicando processos iterativos como <b>Newton-Raphson</b> ou o <b>Gradiente Descendente</b>.


Quanto as funções de custo para este caso, temos: 

a) Maximizar a <b>Verossimilhança</b> (<i>Maximum Likelihood Estimation - MLE</i>);

$$argmax \quad L(\beta_{0}, \beta_{1}, ..., \beta_{n}) = \prod_{i=1}^{n} P(y_{i}|x_{i};\beta_{0}, \beta_{1}, ..., \beta_{n})$$

Como esta função faz o produto das probabilidades condicionais das observações pertencerem a uma determinada classe, encontrar os valores de $$\beta$$ que maximizem a função significa encontrar os valores de $$\beta$$ que maximizam a <b>probabilidade</b> das observações pertencerem a uma determinada classe.

b) Minimizar a <b>Entropia Cruzada</b> (<i>Cross-Entropy</i>).

$$H(y, \hat{y}) = -\frac{1}{n} \sum_{i=1}^{n} \left( y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right)$$


A entropia cruzada mede a diferença entre as probabilidades preditas e as probabilidades reais. Logo, é natural pensar que se estamos minimizando o valor da entropia cruzada, estamos minimizando o <i>erro</i>.


Em ambos os casos, o calculo da probabilidade condicional é feito através da <b>função logística</b> (ou <i>função sigmoide</i>), aplicando os valores de $$\beta_{0}, \beta_{1}, ..., \beta_{n}$$ de uma i-ésima iteração:


$$\hat{y_{i}} = P(y_{i} = 1|x_{i}) = \frac{1}{1 + e^{-(\beta_{0}, \beta_{1}x_{1}, ..., \beta_{n}x_{n})}}$$


Uma vez que os valores de $$\beta$$ foram otimizados, pode-se encontrar o valor da <b>probabilidade</b> de qualquer observação pertencer a classe "1". No entando, apenas realizar as predições não é suficiente quando olhamos do ponto de vista do negócio. Os <i>insights</i> e respostas obtidas através do modelo precisam ser interpretáveis! E para esta tarefa, entramos na parte da equação que ainda não abordei: o <b><i>log-odds</i></b>.


$$\ln{(\frac{p}{1-p})}$$


É preciso lembrar, porém, que <i>odds</i> (chance) e probabilidade são conceitos diferentes. Enquanto a <b>probabilidade</b> é quantificada como a razão entre um evento ocorrer e todos os possíveis resultados para este evento, a <i>odds</i> é a razão entre um evento ocorrer e este mesmo evento <b>não</b> ocorrer.


Por exemplo, se um determinado evento possui 5 sucessos em 8 tentativas, podemos dizer que a probabilidade de se obter um sucesso ou um fracasso são, respectivamente:


$$P_{sucesso} = \frac{Qtd.Sucessos}{Qtd.Tentativas} = \frac{5}{8} = 0.625$$

$$P_{fracasso} = 1 - P_{sucesso} = 1 - 0.625 = 0.375$$

Por outro lado, se estamos interessados na <i>odds</i> de sucesso ou fracasso, temos que:


$$odds_{sucesso} = \frac{Qtd.Sucessos}{Qtd.Fracassos} = \frac{5}{3} \approx 1.7$$

$$odds_{fracasso} = \frac{Qtd.Fracassos}{Qtd.Sucessos} = \frac{3}{5} = 0.6$$


Também podemos chegar ao mesmo resultado para as <i>odds</i> utilizando os conceitos de probabilidade (e assim chegando bem próximo do que vemos na Regressão Logística):


$$odds_{sucesso} = \frac{P_{sucesso}}{P_{fracasso}} \iff \frac{P_{sucesso}}{1 - P_{sucesso}} = \frac{0.625}{1 - 0.625} \approx 1.7$$


Olhando para os resultados percebemos que enquanto a probabilidade é sempre um valor compreendido no intervalo $$[0,1]$$, não podemos afirmar o mesmo em relação a <i>odds</i>, que possui intervalo de valores compreendido no intervalo $$[0,\infty]$$. Isso pode ser observado em casos que a probabilidade se aproxime muito de valores extremos (0 ou 1). 


Como a interpretabilidade do resultado da Regressão Logística reside em utilizar os valores dos coeficientes $$\beta$$, para avaliar se uma variação em $$x_{n}$$ aumenta ou diminui a probabilidade (ou, no nosso caso, aumenta ou diminui o <i>score</i> de um cliente), obrigatoriamente precisamos ter uma relação de <b>linearidade</b> entre nossas variáveis independentes $$x_{n}$$ e o nosso meio de quantificar uma observação. Essa linearidade nunca existirá se utilizarmos somente o valor da probabilidade, uma vez que os resultados obrigatoriamente estarão entre $$[0,1]$$. 


Embora os valores possíveis para a <i>odds</i> englobem mais valores, ele ainda ainda possui um problema: ele é <b>assimétrico</b>, uma vez que para valores de $$P\le0.5$$ geram valores de <i>odds</i> entre 0 e 1, enquanto valores de $$P > 0.5$$ podem ir até $$\infty$$. É aqui que tudo muda quando introduzimos o <b>logaritmo</b> do valor da <i>odds</i> (ou <b>log-odds</b>), que além de tornar este intervalo totalmente simétrico, ainda é capaz de lidar com variáveis em escalas diferentes.


Aplicando a <b>log-odds</b> no nosso exemplo anterior, temos que:

$$\log{(odds_{sucesso})} = \log{(\frac{5}{3})} = 0.2218$$

$$\log{(odds_{fracasso})} = \log{(\frac{3}{5})} = -0.2218$$


O que, sem dúvidas, torna a interpretação do modelo muito mais prática.


<h3>3.2.2. O Modelo</h3>
Agora que já cobri a ideia de funcionamento por trás da Regressão Logística, irei aplicar o modelo com a ABT obtida após a etapa de Feature Engineering.

Para iniciar, carreguei os dados e fiz o split de ``treino`` e ``teste``, removendo as colunas de ID e o Target.

Após o split, iniciei a etapa de `Data Preparation`, aplicando as transformações que havia mencionado anteriormente. Como ainda existem muitas variáveis na nossa ABT, removi as variáveis que possuíam <b>correlação de pearson</b> maior que ``0.5``, e filtrei as 20 melhores variáveis segundo a <b>Feature Importance</b> do ``XGBoost``.

<div style="text-align: center; overflow-x: auto;">
  <pre class="language-python"><code>
# Instanciando o modelo
xgb_model = XGBClassifier(random_state = 1)

# Fittando o modelo
xgb_model.fit(abt_train, y_train)

# Criando uma tabela com a feature importance
importances_df = pd.DataFrame({
    'vars' : abt_train.columns,
    'importance' : xgb_model.feature_importances_
})

# Selecionando as 20 melhores variáveis
best_vars = importances_df.sort_values(by= 'importance', ascending= False)[:20].vars.tolist()

# Atualizando nossa abt
abt_train = abt_train[best_vars].copy()
abt_test = abt_test[best_vars].copy()
  </code></pre>
</div>

Aqui começa a parte mais importante da aplicação da Regressão Logística. O primeiro passo é verificar a linearidade das variáveis restantes com a <b><i>log-odds</i></b>, pelos motivos abordados anteriormente. Para isso, calculei o $$R^{2}$$ e gerei gráficos para cada variável. Não irei trazer todos os gráficos aqui, mas apenas um exemplo de um fit muito bom e outro muito ruim para ilustrar a situação.

a) Exemplo de Fit Ótimo:
<div class="container">
    <img class= "centered-image" src="/assets/images/good_r2_logodds.png" alt="EDA">
</div>

Se repararmos, esta é uma das variáveis provenientes de `bureaus de crédito`. Conforme vimos anteriormente durante o EDA, esta variável se mostrava uma boa preditora para o <i>target</i>.

b) Exemplo de Fit Ruim:

<div class="container">
    <img class= "centered-image" src="/assets/images/bad_r2_logodds.png" alt="EDA">
</div>

Para as variáveis com baixo $$R^{2}$$, tentarei aplicar algumas transformações (como logaritmo, raiz quadrada ou exponencial) e verificar novamente a linearidade com a <b><i>log-odds</i></b>.

<div style="overflow-x: auto;">
  <table>
    <tr>
      <th>Variable</th>
      <th>Best Transformation</th>
      <th>R^2 of Transformation</th>
      <th>Feat Eng</th>
      <th>Transformation Equation</th>
    </tr>
    <tr>
      <td>VL_TOT_VL_MAX_AMT_CREDIT_SUM_OVERDUE_CREDIT_AC...</td>
      <td>AbsLog</td>
      <td>0.018054</td>
      <td>Categorizar</td>
      <td>AbsLog(VL_TOT_VL_MAX_AMT_CREDIT_SUM_OVERDUE_CR...)</td>
    </tr>
    <tr>
      <td>AMT_GOODS_PRICE</td>
      <td>Quadratic</td>
      <td>0.379131</td>
      <td>Categorizar</td>
      <td>Quadratic(AMT_GOODS_PRICE)</td>
    </tr>
    <tr>
      <td>OWN_CAR_AGE</td>
      <td>AbsLog</td>
      <td>0.389392</td>
      <td>Categorizar</td>
      <td>AbsLog(OWN_CAR_AGE)</td>
    </tr>
    <tr>
      <td>VL_MAX_VL_SUM_NUM_INSTALMENT_NUMBER_U3M_INSTAL...</td>
      <td>Quadratic</td>
      <td>0.639222</td>
      <td>Categorizar</td>
      <td>Quadratic(VL_MAX_VL_SUM_NUM_INSTALMENT_NUMBER_...)</td>
    </tr>
    <tr>
      <td>VL_TOT_VL_MAX_AMT_CREDIT_SUM_DEBT_CREDIT_TYPE_...</td>
      <td>AbsLog</td>
      <td>0.535312</td>
      <td>Categorizar</td>
      <td>AbsLog(VL_TOT_VL_MAX_AMT_CREDIT_SUM_DEBT_CREDI...)</td>
    </tr>
  </table>
</div>

Mesmo após as transformações, os novos valores de $$R^{2}$$ estão longes de mostrarem alguma linearidade com a <b><i>log-odds</i></b>, não restando outra alternativa a não ser <b>categorizar</b> os valores dessas variáveis. A categorização normalmente deve ser evitada, devido a perda de informação e poder estatístico. Porém, no caso específico da Regressão Logística, é uma alternativa viável para lidar com a violação da premissa de linearidade com a <b><i>log-odds</i></b>.

Aqui o processo de categorização das variáveis foi feito através do treinamento de uma arvore de decisão simples, usando as quebras dos nós com parâmetro. Após o término da categorização, precisamos verificar novamente os valores de $$R^{2}$$ das variáveis para avaliar se o procedimento cumpriu com seu objetivo. Novamente, não irei trazer aqui todos os resultados, porém podemos comparar utilizando a variável que observamos um baixo $$R^{2}$$ anteriormente:

<div class="container">
    <img class= "centered-image" src="/assets/images/new_r2_good_fit.png" alt="EDA">
</div>

O mesmo resultado foi observado nas demais variáveis.

Apliquei os mesmos procedimentos da etapa de `Data Preparation`, agora contemplando os dados categorizados. O próximo passo é gerar o <i><b>scorecard</b></i> das variáveis e verificar o $$p-valor$$.

<div style="overflow-x: auto;">
  <table>
    <tr>
      <th>Variavel</th>
      <th>Beta Coefficient</th>
      <th>P-Value</th>
      <th>Wald Statistic</th>
    </tr>
    <tr>
      <td>EXT_SOURCE_2</td>
      <td>-2.145261e+00</td>
      <td>0.000000e+00</td>
      <td>1897.672562</td>
    </tr>
    <tr>
      <td>EXT_SOURCE_3</td>
      <td>-2.497780e+00</td>
      <td>0.000000e+00</td>
      <td>1874.889747</td>
    </tr>
    <tr>
      <td>const</td>
      <td>-5.469212e+00</td>
      <td>4.184963e-96</td>
      <td>432.707103</td>
    </tr>
    <tr>
      <td>EXT_SOURCE_1</td>
      <td>-1.222648e+00</td>
      <td>1.200664e-65</td>
      <td>292.832357</td>
    </tr>
    <tr>
      <td>ORGANIZATION_TYPE</td>
      <td>8.399421e+00</td>
      <td>1.314801e-53</td>
      <td>237.596163</td>
    </tr>
    <tr>
      <td>CODE_GENDER</td>
      <td>9.394816e+00</td>
      <td>3.376465e-43</td>
      <td>189.880285</td>
    </tr>
    <tr>
      <td>TFT_VL_TOT_VL_MAX_AMT_CREDIT_SUM_DEBT_CREDIT_TYPE_CREDIT_CARD_3.0</td>
      <td>4.391254e-01</td>
      <td>1.776173e-30</td>
      <td>131.659469</td>
    </tr>
    <tr>
      <td>TFT_VL_TOT_VL_MAX_AMT_CREDIT_SUM_DEBT_CREDIT_TYPE_CREDIT_CARD_1.0</td>
      <td>2.918998e-01</td>
      <td>3.460126e-24</td>
      <td>102.936798</td>
    </tr>
    <tr>
      <td>NAME_CONTRACT_TYPE</td>
      <td>1.312104e+01</td>
      <td>2.469341e-22</td>
      <td>94.485181</td>
    </tr>
    <tr>
      <td>NAME_EDUCATION_TYPE</td>
      <td>7.430119e+00</td>
      <td>6.449095e-22</td>
      <td>92.585106</td>
    </tr>
    <tr>
      <td>TFT_AMT_GOODS_PRICE_2.0</td>
      <td>2.355020e-01</td>
      <td>7.541621e-22</td>
      <td>92.275393</td>
    </tr>
    <tr>
      <td>TFT_AMT_GOODS_PRICE_1.0</td>
      <td>2.801600e-01</td>
      <td>5.941381e-18</td>
      <td>74.540272</td>
    </tr>
    <tr>
      <td>FLAG_OWN_CAR</td>
      <td>2.070341e+01</td>
      <td>1.029334e-16</td>
      <td>68.912441</td>
    </tr>
    <tr>
      <td>DEF_60_CNT_SOCIAL_CIRCLE</td>
      <td>1.823345e-01</td>
      <td>5.084611e-15</td>
      <td>61.227509</td>
    </tr>
    <tr>
      <td>REGION_RATING_CLIENT_W_CITY</td>
      <td>1.542734e-01</td>
      <td>3.686974e-14</td>
      <td>57.328901</td>
    </tr>
    <tr>
      <td>TFT_VL_MAX_VL_SUM_NUM_INSTALMENT_NUMBER_U3M_INSTALMENTS_FL_U12M_PREVIOUS_APPLICATION_3.0</td>
      <td>3.594390e-01</td>
      <td>3.914482e-13</td>
      <td>52.685612</td>
    </tr>
    <tr>
      <td>OCCUPATION_TYPE</td>
      <td>3.402880e+00</td>
      <td>4.690592e-12</td>
      <td>47.812220</td>
    </tr>
    <tr>
      <td>TFT_VL_TOT_VL_MAX_AMT_CREDIT_SUM_DEBT_CREDIT_TYPE_CREDIT_CARD_2.0</td>
      <td>2.304362e-01</td>
      <td>3.424755e-11</td>
      <td>43.917712</td>
    </tr>
    <tr>
      <td>QT_MAX_DAYS_FIRST_DRAWING_FL_U12M_PREVIOUS_APPLICATION</td>
      <td>-9.306779e-07</td>
      <td>9.356356e-09</td>
      <td>32.970590</td>
    </tr>
    <tr>
      <td>TFT_VL_MAX_VL_SUM_NUM_INSTALMENT_NUMBER_U3M_INSTALMENTS_FL_U12M_PREVIOUS_APPLICATION_1.0</td>
      <td>1.686962e-01</td>
      <td>1.350309e-07</td>
      <td>27.792754</td>
    </tr>
    <tr>
      <td>TFT_OWN_CAR_AGE_4.0</td>
      <td>2.817931e-01</td>
      <td>3.363984e-07</td>
      <td>26.028592</td>
    </tr>
    <tr>
      <td>REG_CITY_NOT_LIVE_CITY</td>
      <td>1.609482e-01</td>
      <td>3.840242e-07</td>
      <td>25.772995</td>
    </tr>
    <tr>
      <td>TFT_AMT_GOODS_PRICE_3.0</td>
      <td>-1.277319e-01</td>
      <td>1.567337e-05</td>
      <td>18.653810</td>
    </tr>
    <tr>
      <td>TFT_OWN_CAR_AGE_3.0</td>
      <td>2.329940e-01</td>
      <td>3.490027e-05</td>
      <td>17.130342</td>
    </tr>
    <tr>
      <td>TFT_OWN_CAR_AGE_2.0</td>
      <td>1.584614e-01</td>
      <td>1.799706e-04</td>
      <td>14.029441</td>
    </tr>
    <tr>
      <td>QT_MIN_QT_MAX_SK_DPD_DEF_NAME_CONTRACT_STATUS_ACTIVE_POSCASH_FL_U12M_PREVIOUS_APPLICATION</td>
      <td>3.870633e-02</td>
      <td>5.803978e-04</td>
      <td>11.837819</td>
    </tr>
    <tr>
      <td>FLAG_WORK_PHONE</td>
      <td>6.051035e-02</td>
      <td>9.791794e-03</td>
      <td>6.672387</td>
    </tr>
    <tr>
      <td>TFT_VL_MAX_VL_SUM_NUM_INSTALMENT_NUMBER_U3M_INSTALMENTS_FL_U12M_PREVIOUS_APPLICATION_2.0</td>
      <td>4.839981e-01</td>
      <td>3.666035e-02</td>
      <td>4.366121</td>
    </tr>
    <tr>
      <td>TFT_VL_TOT_VL_MAX_AMT_CREDIT_SUM_OVERDUE_CREDIT_ACTIVE_CLOSED_2.0</td>
      <td>3.830969e-01</td>
      <td>5.400200e-01</td>
      <td>0.375503</td>
    </tr>
  </table>
</div>

Das variáveis presentes no <i><b>scorecard</b></i>, apenas a <b>última</b> será removida, já que seu $$p-valor$$ ficou acima do nivel de significancia (95%).

Finalmente, apliquei o modelo usando a ABT com as variáveis restantes, separando a taxa de evento em decis. Neste momento é importante observar a capacidade de <b>ordenação</b> do modelo, uma vez que quanto mais o <i>score</i> cresce, é natural esperar que a taxa de evento (inadimplencia) diminua.

<div class="container">
    <img class= "centered-image" src="/assets/images/event_rate_decile.png" alt="EDA">
</div>

O modelo conseguiu ordenar as faixas de score conforme o esperado. Este gráfico nos mostra a base de clientes divida em 10 partes aproximadamente iguais, usando como critério o <i>score</i> de cada cliente, de forma que cada parte contenha uma <b>faixa de <i>score</i></b>. Para os clientes da primeira faixa de score, podemos observar uma taxa de evento (no caso deste projeto, <b>inadimplencia</b>) de 25%.


<table>
  <tr>
    <th>Metric</th>
    <th>Train Value</th>
    <th>Test Value</th>
  </tr>
  <tr>
    <td>KS</td>
    <td>0.363068</td>
    <td>0.366113</td>
  </tr>
  <tr>
    <td>AUC</td>
    <td>0.743848</td>
    <td>0.745137</td>
  </tr>
  <tr>
    <td>Gini</td>
    <td>0.487695</td>
    <td>0.490274</td>
  </tr>
</table>

Olhando para as métricas do modelo, vemos que ele performou bem e variou muito pouco entre o `treino` e `teste`.


<h2>3.3. LightGBM</h2>

Outro modelo que eu gostaria de trazer aqui é o <b>Light Gradient Boosting Machine</b>, um algoritmo de <i>gradient boosting tree</i> que, como o próprio nome diz, funciona em forma de árvores de decisão.

A matemática por trás também envolve a otimização de uma função de perda, através do gradiente descendente, onde as árvores são adicionadas sequencialmente para minimizar a perda (conceito de <i>boosting trees</i>). Para problemas de classificação binária (como o que estamos lidando aqui) frequentemente utiliza, assim como a Regressão Logística, a função de entropia cruzada como função de perda.

Sua implementação é bem mais direta do que a Regressão Logística, uma vez que não precisamos nos preocupar com questões de linearidade entre a <i><b>log-odds</b></i> e as variáveis, além possuir desempenho computacional bem superior, que nos permite trabalhar com uma quantidade maior de variáveis. Tudo isso vem com um custo: os resultados do algoritmo não possuem uma interpretação tão clara quanto a Regressão.

Após carregar os dados e realizar o split, repliquei os procedimentos já feitos anteriormente para a etapa de `Data Preparation`. Com os dados prontos para uso, utilizei o `optuna` para a etapa de <i>tunning</i> dos <b>hiperparametros</b> e apliquei a melhor combinação no modelo.


<div style="text-align: center; overflow-x: auto;">
  <pre class="language-python"><code>

study = optuna.create_study(
  direction= 'maximize',
  study_name= 'LGBM_M3',
  storage= 'sqlite:///LGBM_M3.db'
)

model = lgb.LGBMClassifier(**study.best_params, random_state= 1, verbosity= -1)

model.fit(X_train_dataprep, y_train)

metricas = calculate_metrics(str(model)[:str(model).find("(")], model, X_train_dataprep, y_train, X_test_dataprep, y_test)

  </code></pre>
</div>

<br>

Com o término da execução desta parte, podemos avaliar as métricas do modelo (tabela abaixo).

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
    <td>0.919750</td>
    <td>0.726829</td>
    <td>0.024271</td>
    <td>0.800839</td>
    <td>0.601677</td>
    <td>0.450380</td>
  </tr>
  <tr>
    <td>LGBMClassifier</td>
    <td>Teste</td>
    <td>0.920778</td>
    <td>0.565217</td>
    <td>0.015193</td>
    <td>0.759623</td>
    <td>0.519246</td>
    <td>0.387033</td>
  </tr>
</table>

Comparando com o modelo <b>baseline</b>, notamos uma maior estabilidade entre os conjuntos e uma melhora significativa durante o `teste`.

Conforme vimos na Regressão Logística, aqui também estamos interessados na capacidade do modelo de ordenar a taxa de evento nas faixas de <i>score</i>.

<div class="container">
    <img class= "centered-image" src="/assets/images/event_rate_decile_lgbm.png" alt="EDA">
</div>

O LGBM conseguiu um acumulo maior de taxa de evento no primeiro decil que a Regressão Logística (algo em torno de 1.5%). As implicações para o <b>negócio</b> serão discutidas em detalhes na seção a seguir.


<h1>4. Avaliando os Resultados</h1>

Nesta última seção irei abordar os ganhos financeiros potenciais com a aplicação dos modelos abordados até aqui, considerando uma política de crédito vigente que aprova 100% dos clientes (para facilitar a comparação, já que não temos esta informação). Também iremos considerar que o primeiro decil será <b>rejeitado</b>, uma vez que é o decil com maior concentração de inadimplencia.

Lembrando que esta abordagem é apenas uma tentativa de quantificar os resultados do modelo. Como nossas informações estão limitadas somente ao momento da ``aplicação``, não conseguimos trazer aqui efeitos como <b>swap-in</b> ou <b>swap-out</b>.

Para estimar os ganhos, vou considerar o <b>ticket médio</b> e a <b>taxa de juros</b> dos dois produtos (Cash Loans e Revolving Loans), da seguinte maneira:

<table border="1">
  <tr>
    <th>Produto</th>
    <th>% Volume</th>
    <th>Ticket Médio</th>
    <th>Taxa de Juros</th>
  </tr>
  <tr>
    <td>Cash Loans</td>
    <td>91,00%</td>
    <td>R$ 628.524,55</td>
    <td>22,00%</td>
  </tr>
  <tr>
    <td>Revolving Loans</td>
    <td>9,00%</td>
    <td>R$ 325.106,09</td>
    <td>49,00%</td>
  </tr>
</table>

<b>Obs.</b> Irei representar os valores monetários a seguir em reais por conveniência, porém esta não é a moeda original dos dados.

<h2>4.1. Modelo Vigente</h2>

Conforme definimos na seção anterior, irei considerar que o modelo vigente não recusa nenhum cliente, o que simula um cenário onde não há uma política de crédito em vigor. Na amostragem que estou utilizando (baseado no conjunto de teste dos modelos), estamos olhando para um total de 64578 clientes. Olhando para o modelo vigente, temos o seguinte situação:


<table border="1">
  <tr>
    <th>Modelo</th>
    <th>Volume Público</th>
    <th>Volume Aprovado</th>
    <th>Taxa de Aprovação</th>
    <th>Volume Inadimp.</th>
    <th>Taxa Inadimp.</th>
  </tr>
  <tr>
    <td>Vigente</td>
    <td>64578</td>
    <td>64578</td>
    <td>100%</td>
    <td>5134</td>
    <td>7,95%</td>
  </tr>
</table>

A taxa de inadimplencia está aproximadamente igual ao valor que descobrimos durante a etapa de análise exploratória, sendo um indicativo de que não há problemas com a nossa amostra. Com o número de <b>clientes</b> e a <b>taxa de aprovação</b>, podemos estimar a receita total gerada e a perda proveniente dos clientes inadimplentes.

<table border="1">
  <tr>
    <th>Modelo</th>
    <th>Receita</th>
    <th>Perda</th>
    <th>Balanço</th>
    <th>% Ganho</th>
  </tr>
  <tr>
    <td>Vigente</td>
    <td>R$ 9.051.755.767</td>
    <td>R$ 3.086.651.305</td>
    <td>R$ 5.965.104.462</td>
    <td>0,00%</td>
  </tr>
</table>

Feito este processo para o modelo vigente, vamos analisar os modelos propostos e quantificar o ganho em relação ao atual.

<h2>4.2. Modelos Propostos</h2>

Iniciando com o resultado da ordenação por decis da ``Regressão Logística``, temos o seguinte cenário:

<table border="1">
  <tr>
    <th>Decil</th>
    <th>Event Rate</th>
    <th>Volume Clientes</th>
    <th>Qtd. Inadimp.</th>
  </tr>
  <tr>
    <td>1</td>
    <td>25,59%</td>
    <td>6460</td>
    <td>1653</td>
  </tr>
  <tr>
    <td>2</td>
    <td>14,45%</td>
    <td>6459</td>
    <td>933</td>
  </tr>
  <tr>
    <td>3</td>
    <td>10,40%</td>
    <td>6470</td>
    <td>673</td>
  </tr>
  <tr>
    <td>4</td>
    <td>7,91%</td>
    <td>6461</td>
    <td>511</td>
  </tr>
  <tr>
    <td>5</td>
    <td>6,10%</td>
    <td>6457</td>
    <td>394</td>
  </tr>
  <tr>
    <td>6</td>
    <td>4,67%</td>
    <td>6493</td>
    <td>303</td>
  </tr>
  <tr>
    <td>7</td>
    <td>3,67%</td>
    <td>6429</td>
    <td>236</td>
  </tr>
  <tr>
    <td>8</td>
    <td>2,80%</td>
    <td>6502</td>
    <td>182</td>
  </tr>
  <tr>
    <td>9</td>
    <td>2,28%</td>
    <td>6446</td>
    <td>147</td>
  </tr>
  <tr>
    <td>10</td>
    <td>1,59%</td>
    <td>6401</td>
    <td>102</td>
  </tr>
</table>

Conforme nossas premissas, caso o primeiro decil seja automaticamente rejeitado, estaremos negando a concessão para 6460 clientes, 1653 (25.59%) deles considerados "maus". A tabela abaixo compara a Regressão Logística com o Vigente:


<table border="1">
  <tr>
    <th>Cenário</th>
    <th>Volume Público</th>
    <th>Volume Aprovado</th>
    <th>Taxa de Aprovação</th>
    <th>Volume Inadimp.</th>
    <th>Taxa Inadimp.</th>
  </tr>
  <tr>
    <td>Regressão Logística</td>
    <td>64578</td>
    <td>58118</td>
    <td>90,0%</td>
    <td>3481</td>
    <td>5,99%</td>
  </tr>
  <tr>
    <td>Vigente x Reg. Log</td>
    <td>0</td>
    <td>-6460</td>
    <td>-10,0%</td>
    <td>-1653</td>
    <td>-1,96%</td>
  </tr>
</table>


Aqui podemos observar que reduzir 10% da aprovação, neste cenário, resultaria em um <b>risco</b> 1.96% menor de inadimplência. Para este modelo ser utilizado, o valor da perda proveniente da redução na aprovação precisa ser menor que o valor da perda por inadimplência, sugerindo que mesmo com uma menor aprovação, o banco ainda teria lucros maiores. A tabela abaixo traz esta comparação:


<table border="1">
  <tr>
    <th>Cenário</th>
    <th>Receita</th>
    <th>Perda</th>
    <th>Balanço</th>
    <th>% Ganho</th>
  </tr>
  <tr>
    <td>Reg. Log.</td>
    <td>R$ 8.146.271.821</td>
    <td>R$ 2.092.841.159</td>
    <td>R$ 6.053.430.662</td>
    <td>1,481%</td>
  </tr>
  <tr>
    <td>Vigente X Reg. Log</td>
    <td>-R$ 905.483.945</td>
    <td>-R$ 993.810.146</td>
    <td>R$ 88.326.200</td>
    <td>1,481%</td>
  </tr>
</table>

Mesmo com uma receita menor, ainda foi possível ter um lucro de `R$ 88.326.200` (1.48%), somente com a eliminação do primeiro decil. É possível que fazendo uma separação um pouco menor das faixas de score (15 faixas ao invés de 10, por exemplo), consigamos uma concentração ainda maior de inadimplentes em uma única faixa, tornando este lucro ainda maior, desde que as premissas de ordenação das faixas se mantenha.


Olhando agora para o ``LGBM``, temos o seguinte cenário:

<table border="1">
  <tr>
    <th>Decil</th>
    <th>Event Rate</th>
    <th>Volume</th>
    <th>Qtd. Inadimp.</th>
  </tr>
  <tr>
    <td>1</td>
    <td>26,15%</td>
    <td>6458</td>
    <td>6458</td>
  </tr>
  <tr>
    <td>2</td>
    <td>14,56%</td>
    <td>6458</td>
    <td>12916</td>
  </tr>
  <tr>
    <td>3</td>
    <td>10,56%</td>
    <td>6458</td>
    <td>19374</td>
  </tr>
  <tr>
    <td>4</td>
    <td>7,60%</td>
    <td>6457</td>
    <td>25828</td>
  </tr>
  <tr>
    <td>5</td>
    <td>6,01%</td>
    <td>6458</td>
    <td>32290</td>
  </tr>
  <tr>
    <td>6</td>
    <td>4,41%</td>
    <td>6458</td>
    <td>38748</td>
  </tr>
  <tr>
    <td>7</td>
    <td>3,69%</td>
    <td>6457</td>
    <td>45199</td>
  </tr>
  <tr>
    <td>8</td>
    <td>2,85%</td>
    <td>6458</td>
    <td>51664</td>
  </tr>
  <tr>
    <td>9</td>
    <td>2,21%</td>
    <td>6458</td>
    <td>58122</td>
  </tr>
  <tr>
    <td>10</td>
    <td>1,46%</td>
    <td>6458</td>
    <td>64580</td>
  </tr>
</table>

Este modelo se saiu ligeiramente melhor em concentrar a taxa de evento no primeiro decil. Aplicando os mesmos conceitos anteriores, temos o seguinte resultado:

<table border="1">
  <tr>
    <th>Cenário</th>
    <th>Receita</th>
    <th>Perda</th>
    <th>Balanço</th>
    <th>% Ganho</th>
  </tr>
  <tr>
    <td>LGBM</td>
    <td>R$ 8.146.552.157</td>
    <td>R$ 2.071.194.627,58</td>
    <td>R$ 6.075.357.529,42</td>
    <td>1,85%</td>
  </tr>
  <tr>
    <td>Vigente x LGBM</td>
    <td>-R$ 905.203.610</td>
    <td>-R$ 1.015.456.677</td>
    <td>R$ 110.253.067</td>
    <td>1,85%</td>
  </tr>
</table>

Este modelo se saiu ainda melhor que a ``Regressão Logística``, gerando um lucro de `R$ 110.253.067` em relação ao modelo vigente. É importante, porém, observar se a performance do modelo irá se manter conforme novos dados forem introduzidos.