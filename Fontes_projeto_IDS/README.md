# Fontes do projeto — IDS IoT / Generalização

Esta pasta reúne as fontes centrais usadas como referência para a continuação do projeto.

## 1. Artigo-base

**Arquivo:** `artigo/Artigo_SBCUP_2026.pdf`

Artigo: *Abordagem em duas fases para detecção e classificação de ataques em redes IoT com computação em névoa*.

É a fonte principal para o problema original, metodologia, arquitetura IoT–névoa–nuvem e trabalhos futuros.

## 2. Repositório original

**Arquivo:** `repositorios/GitHub_original.txt`

Link:
https://github.com/Math-BUG/iot-fog-ids-two-phase

Contém o código e os artefatos associados ao trabalho original apresentado no artigo.

## 3. Repositório estendido

**Arquivo:** `repositorios/GitHub_estendido.txt`

Link:
https://github.com/Math-BUG/ids_generalization_pipeline

É o repositório atual da extensão do trabalho, voltada à generalização, protocolos de split, execução em GPU e futura avaliação da inferência na névoa.

## Direção atual do projeto

O eixo principal da extensão foi definido como:

**generalização do IDS + treinamento offline acelerado em GPU na nuvem/cluster + inferência online na névoa.**

A sequência principal de investigação é:

1. validar o pipeline GPU real;
2. estabelecer o baseline completo no TON_IoT;
3. avaliar generalização por grupos;
4. avaliar generalização temporal;
5. estudar estabilidade do clustering;
6. avaliar generalização cross-dataset;
7. explorar HDBSCAN e SHAP;
8. avaliar a inferência em um nó de névoa.

A linha de seleção/compressão por orçamento B não é atualmente o eixo principal do projeto.
