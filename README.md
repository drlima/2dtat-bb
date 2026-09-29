# Construção de TARGET e Modelagem de Crédito com Classes Desbalanceadas

Tutorial hands-on sobre um problema clássico de risco de crédito: **aprovação de cartão de crédito**, a partir do dataset público [`rikdifos/credit-card-approval-prediction`](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) no Kaggle.

Este repositório foi pensado para quem está começando e não tem contexto prévio sobre o desafio — os notebooks são standalone, comentados em detalhe e podem ser lidos como um tutorial guiado, não só executados. Eles também não escondem os resultados desconfortáveis: quando uma técnica não funciona, ou quando os dados não sustentam o que se esperava deles, isso é mostrado e discutido, porque é parte do que se quer ensinar.

## O que este repositório ensina

Os dados brutos não trazem uma variável de resposta pronta — o TARGET ("bom" ou "mau pagador") precisa ser construído a partir do histórico de pagamentos, e a forma como isso é feito muda drasticamente o problema. A partir daí, a sequência mostra como modelar uma classe rara sem se enganar com as métricas, e o que fazer quando a avaliação honesta revela que o sinal não está onde se procurava.

0. **[`notebooks/00_bootstrap_dados.ipynb`](notebooks/00_bootstrap_dados.ipynb)** — baixa automaticamente os CSVs do Kaggle via `kagglehub` e os coloca em `data/raw/`. Rode uma vez, antes dos demais (opcional — veja "Obtendo os dados" abaixo).
1. **[`notebooks/demo_criacao_target.ipynb`](notebooks/demo_criacao_target.ipynb)** — como construir o TARGET a partir do histórico de atrasos (`credit_record.csv`), comparando um critério permissivo (atraso a partir de 30 dias) com um critério rígido (atraso de 60 dias ou mais). Mostra como a escolha do critério muda o desbalanceamento das classes (de ~88/12 para ~98/2), defende que essa escolha é de negócio e não estatística, e expõe os vieses escondidos nessa etapa: o viés de exposição da agregação "algum dia atrasou", o viés de seleção do `inner join` com `application_record.csv`, e dois tipos de duplicata na base — IDs colidentes e, mais grave, o mesmo cadastro repetido sob IDs diferentes.
2. **[`notebooks/demo_modelagem_desbalanceada.ipynb`](notebooks/demo_modelagem_desbalanceada.ipynb)** — continuação do primeiro: pré-processamento mínimo, split treino/teste **por grupo de perfil** (para que clientes com cadastro idêntico não vazem entre treino e teste) e antes de qualquer balanceamento, comparação entre baseline, `class_weight="balanced"` e SMOTENC, otimização de hiperparâmetros com validação cruzada também por grupo e por diferentes métricas (f1, recall, average precision), e calibração do limiar de decisão. Dois fios condutores: **acurácia alta não significa modelo útil** quando a classe de interesse é rara, e **técnicas de balanceamento redistribuem erros, não criam sinal** — avaliado de forma honesta, o cadastro do solicitante quase não prevê quem vai atrasar.
3. **[`notebooks/demo_tecnicas_avancadas.ipynb`](notebooks/demo_tecnicas_avancadas.ipynb)** — organiza os problemas encontrados no caminho (perfis duplicados e vazamento de avaliação, atributos sem poder preditivo, viés de exposição, otimismo da validação cruzada com poucos positivos) e mostra onde o sinal de fato está: no comportamento de pagamento. Reformula o problema como *behavioral scoring*, com janelas fixas de observação e performance e snapshots rolantes, e discute como ler o resultado da forma que uma área de crédito leria — captura de risco nos clientes mais arriscados, e por que o F1 de um evento raro mede a prevalência tanto quanto mede o modelo. Fecha com uma rodada sobre os dados deduplicados, no nível da pessoa.

Os notebooks devem ser lidos/executados nessa ordem.

## Principais achados

Alguns resultados que valem ser conhecidos antes de abrir os notebooks, porque contrariam o que a maioria dos exemplos públicos sobre este dataset sugere:

- O critério rígido de inadimplência produz ~1,7% de maus pagadores. Prever "bom pagador" para todo mundo dá 98,3% de acurácia — e um modelo inútil.
- 91% dos clientes têm ao menos um "gêmeo" com cadastro idêntico sob outro `ID`; 68% dos maus pagadores têm um gêmeo que é bom pagador. Com um split aleatório, o mesmo modelo aparenta ROC-AUC de 0,78; com split por grupo, 0,61. A diferença é memorização.
- Avaliado honestamente, o cadastro do solicitante tem poder preditivo praticamente nulo para atraso de 60+ dias (AUC univariada de cada atributo entre 0,45 e 0,51). Nenhuma técnica de balanceamento, busca de hiperparâmetros ou calibração de limiar muda isso.
- Seis meses de comportamento de pagamento levam a ROC-AUC honesta de 0,85-0,88: o 1% mais arriscado da carteira concentra mais de 20% dos futuros inadimplentes. O mesmo modelo tem F1 de ~0,2 — efeito da prevalência de 0,65%, não do modelo, como o notebook demonstra.

## Estrutura do repositório

```text
.
├── data/
│   └── raw/              # CSVs brutos (não versionados — ver "Obtendo os dados")
├── notebooks/
│   ├── 00_bootstrap_dados.ipynb
│   ├── demo_criacao_target.ipynb
│   ├── demo_modelagem_desbalanceada.ipynb
│   └── demo_tecnicas_avancadas.ipynb
└── requirements.txt
```

## Ambiente

Requer Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

As versões estão pinadas em `requirements.txt`. Os notebooks 2 e 3 usam `imbalanced-learn`; o notebook 3 usa também `xgboost`.

## Obtendo os dados

Os notebooks esperam os arquivos `application_record.csv` e `credit_record.csv` em `data/raw/`. Duas formas de obtê-los:

**Opção 1 — notebook de bootstrap (recomendado)**
Rode [`notebooks/00_bootstrap_dados.ipynb`](notebooks/00_bootstrap_dados.ipynb): ele baixa o dataset via `kagglehub` e copia os CSVs para `data/raw/` automaticamente. Requer uma conta Kaggle e credenciais de API configuradas — o próprio notebook explica as opções de autenticação.

**Opção 2 — download manual**
Baixe o dataset em [kaggle.com/datasets/rikdifos/credit-card-approval-prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) e extraia os dois CSVs para `data/raw/`.

## Executando

```bash
source .venv/bin/activate
jupyter lab
```

Abra os notebooks em `notebooks/` na ordem listada acima. Todos rodam de ponta a ponta com o kernel reiniciado; o notebook 2 leva alguns minutos (três buscas de hiperparâmetros) e o notebook 3 também (construção dos snapshots e validação cruzada de vários modelos). Os resultados são reprodutíveis pela semente fixa (`RANDOM_STATE = 42`), com uma ressalva discutida no notebook 2: com paralelismo (`n_jobs=-1`), métricas de classificação podem variar na segunda ou terceira casa decimal entre máquinas; métricas de ranking (ROC-AUC, average precision) e as conclusões não mudam.

## Licença

Código sob licença [MIT](LICENSE). O dataset é de terceiros (Kaggle, ver link acima) e não está incluído neste repositório — consulte os termos do próprio dataset antes de redistribuí-lo.
