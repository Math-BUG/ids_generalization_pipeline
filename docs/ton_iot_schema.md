# Esquema semântico TON_IoT — Incremento 1

Versão: `ton_iot_network/1.0.0`. A fonte executável é
`src/ids_pipeline/schema.py`; a normalização central está em
`src/ids_pipeline/dataset_schema.py`. Este incremento corrige representação e
tipagem. Não altera a seleção de features de `behavioral_strict`, splits,
seletores, orçamento, algoritmos ou medição de desempenho.

## Evidência e limites da inspeção

Foram inspecionados somente os cabeçalhos dos 23 `Network_dataset_*.csv` locais
em `D:/IC/Dataset`: a união contém 47 colunas, com `uid` presente apenas na
partição 6; as outras partições contêm 46. Também foram lidos cinco registros
iniciais de cada uma das partições 1, 6 e 23. Essas pequenas amostras confirmam
timestamps numéricos em segundos Unix, bytes como strings numéricas, `-`, flags
`F` e código HTTP `0`. Isso não constitui uma auditoria de todos os valores.
Não foram lidos arquivos de associações individuais aos clusters, datasets
completos ou Parquets reais, nem reconvertidas as 23 partições.

A interpretação usa as instruções do projeto e as definições primárias de
[conexões Zeek](https://docs.zeek.org/en/current/reference/logs/conn.html),
[DNS](https://docs.zeek.org/en/current/scripts/base/protocols/dns/main.zeek.html),
[HTTP](https://docs.zeek.org/en/current/scripts/base/protocols/http/main.zeek.html) e
[weird/notice](https://docs.zeek.org/en/current/scripts/base/frameworks/notice/weird.zeek.html).
As correspondências `src`/`dst` com os campos de origem/resposta do Zeek são a
interpretação adotada para o esquema TON_IoT. O armazenamento observado não é
usado para decidir se um código ou uma quantidade entra como número no modelo.

## Contrato

`normalize_dataset_schema(df, report_path=None)` retorna `(frame, report)` e não
aprende parâmetros. Mantém número, ordem e índice das linhas, incluindo índices
duplicados, e não modifica a entrada. Identificadores são preservados, salvo as
sentinelas de ausência cadastradas. Colunas não registradas são preservadas e
listadas; features TON_IoT desconhecidas são rejeitadas no despacho do modelo.
Colunas opcionais ausentes são listadas, sem serem fabricadas pelo normalizador.
O conversor mantém o alinhamento de colunas entre partições já existente.

Cada coluna registra armazenamento normalizado, semântica, tratamento no modelo,
unidade, domínio, nulabilidade, sentinelas e elegibilidade para `log1p`.
O relatório registra também o dtype físico recebido. `float64` permite ausências
e interoperabilidade; contagens continuam tendo domínio integral. `Int64` é o
inteiro anulável do pandas. `string` não significa necessariamente texto livre:
também armazena códigos nominais e booleanos canônicos.

Somente nulos nativos e os tokens exatos `""` e `"-"` representam ausência.
Não se remove whitespace nem se aplica o vocabulário implícito de NA do pandas.
`"NA"`, `"NULL"`, `"nan"` e `"None"` são tokens literais: inválidos em quantidades,
preservados em texto. Zero permanece um valor; não se transforma ausência em zero.

Quantidades aceitam strings numéricas, inclusive notação científica. São
verificados falha de conversão, infinitos, sinal, integralidade e limites
declarados. Inteiros acima de `2**53 - 1` são rejeitados por risco de perda de
precisão, em vez de arredondados silenciosamente. `ts` permanece em segundos
Unix, sem conversão para datetime; exige valor finito, não negativo e inferior
ao limite de representação em nanossegundos do pandas (`int64.max / 1e9`). Esse
limite é técnico, não um intervalo de datas aprendido com o dataset.

Códigos nominais exigem inteiros não negativos e são normalizados para strings
decimais (`1`, `1.0`, `"01"` -> `"1"`). Portas têm domínio inteiro `[0, 65535]`.
`label` exige 0 ou 1. Ambos os alvos rejeitam ausência quando presentes; `type`
não tem vocabulário aprendido/inferido nesta etapa.

Indicadores aceitam explicitamente `T/F`, `true/false`, `True/False`,
`TRUE/FALSE`, `1/0`, `1.0/0.0` e booleanos nativos, com saída `T/F` ou ausência.
Outros tokens, como `yes`, geram violação de domínio.

Qualquer valor inválido interrompe a normalização com `SchemaValidationError`.
O relatório fica na exceção (`.report`) e, quando solicitado, em JSON antes da
exceção. Não se devolve um dataframe parcialmente corrigido. Cada problema tem
contagem e até cinco exemplos com posição, índice e valor; textos dos exemplos
são limitados a 80 caracteres. O mascaramento interno permite reunir os
diagnósticos e não é uma política de imputação. `missing_after` descreve esse
estado interno em relatórios de erro; esses dados nunca seguem para treino.

## Todas as colunas reais

O tratamento se aplica caso a política de features existente selecione a coluna.
Ser categórica não autoriza incluir um identificador. Alvos nunca são features.
`log1p` abaixo significa elegibilidade pela lista explícita padrão, condicionada
a `log1p_numeric` e à seleção da coluna.

| Coluna | Armazenamento normalizado | Semântica | Tratamento | log1p |
|---|---|---|---|---|
| ts | float64 | Timestamp, segundos Unix | Numérico, conforme política temporal existente | Não |
| uid | string | Identificador de conexão | Categórico | Não |
| src_ip | string | Identificador: IP de origem | Categórico | Não |
| src_port | Int64 | Porta de origem | Categórico | Não |
| dst_ip | string | Identificador: IP de destino | Categórico | Não |
| dst_port | Int64 | Porta de destino | Categórico | Não |
| proto | string | Protocolo nominal | Categórico | Não |
| service | string | Serviço nominal | Categórico | Não |
| duration | float64 | Quantidade contínua: duração em segundos | Numérico | Sim |
| src_bytes | float64 | Contagem de bytes de conteúdo, origem | Numérico | Sim |
| dst_bytes | float64 | Contagem de bytes de conteúdo, destino | Numérico | Sim |
| conn_state | string | Estado nominal da conexão | Categórico | Não |
| missed_bytes | float64 | Contagem de bytes não capturados | Numérico | Sim |
| src_pkts | float64 | Contagem de pacotes, origem | Numérico | Sim |
| src_ip_bytes | float64 | Contagem de bytes IP, origem | Numérico | Sim |
| dst_pkts | float64 | Contagem de pacotes, destino | Numérico | Sim |
| dst_ip_bytes | float64 | Contagem de bytes IP, destino | Numérico | Sim |
| dns_query | string | Texto: consulta DNS | Categórico | Não |
| dns_qclass | string | Código nominal: classe DNS | Categórico | Não |
| dns_qtype | string | Código nominal: tipo DNS | Categórico | Não |
| dns_rcode | string | Código nominal: resposta DNS | Categórico | Não |
| dns_AA | string | Indicador booleano | Categórico T/F | Não |
| dns_RD | string | Indicador booleano | Categórico T/F | Não |
| dns_RA | string | Indicador booleano | Categórico T/F | Não |
| dns_rejected | string | Indicador booleano | Categórico T/F | Não |
| ssl_version | string | Versão nominal SSL/TLS | Categórico | Não |
| ssl_cipher | string | Suíte criptográfica nominal | Categórico | Não |
| ssl_resumed | string | Indicador booleano | Categórico T/F | Não |
| ssl_established | string | Indicador booleano | Categórico T/F | Não |
| ssl_subject | string | Texto: sujeito do certificado | Categórico | Não |
| ssl_issuer | string | Texto: emissor do certificado | Categórico | Não |
| http_trans_depth | float64 | Contagem: profundidade de transação HTTP | Numérico | Sim |
| http_method | string | Método HTTP nominal | Categórico | Não |
| http_uri | string | Texto: URI | Categórico | Não |
| http_referrer | string | Texto: referência HTTP | Categórico | Não |
| http_version | string | Versão HTTP nominal | Categórico | Não |
| http_request_body_len | float64 | Contagem de bytes do corpo da requisição | Numérico | Sim |
| http_response_body_len | float64 | Contagem de bytes do corpo da resposta | Numérico | Sim |
| http_status_code | string | Código nominal: status HTTP | Categórico | Não |
| http_user_agent | string | Texto: agente HTTP | Categórico | Não |
| http_orig_mime_types | string | Texto: tipos MIME serializados | Categórico | Não |
| http_resp_mime_types | string | Texto: tipos MIME serializados | Categórico | Não |
| weird_name | string | Nome nominal de evento Zeek | Categórico | Não |
| weird_addl | string | Texto adicional de evento | Categórico | Não |
| weird_notice | string | Indicador booleano | Categórico T/F | Não |
| label | Int64 | Alvo binário: 0 normal / 1 ataque | Alvo, excluído das features | Não |
| type | string | Alvo nominal: normal/tipo de ataque | Alvo, excluído das features | Não |

As duas saídas já existentes de `extract_causal`, `ts_hour` e
`ts_day_of_week`, têm registro separado (`TON_IOT_DERIVED_SCHEMA`) como componentes
de calendário integrais, numéricos e sem log. Preserva-se o código existente `-1`
para indisponibilidade; limites superiores são 23 e 6. Não são colunas brutas do
TON_IoT. A extração temporal existente não foi corrigida neste incremento; sua
interpretação de segundos Unix ainda deve ser revisada na etapa temporal.

## Integração e compatibilidade

- Loader CSV: preserva tokens TON_IoT como strings antes da normalização. A
  identificação usa a assinatura de cabeçalho `ts, src_ip, dst_ip, src_bytes,
  dns_qtype`; entradas genéricas mantêm o carregamento legado. Fixtures parciais
  podem chamar o normalizador diretamente.
- Loader Parquet: normaliza após a leitura, inclusive no caminho cuDF/pandas.
  Arquivos existentes não precisam ser reconvertidos para corrigir a tipagem
  usada no modelo. Informações já perdidas em conversões antigas não podem ser
  recuperadas, por exemplo strings previamente convertidas em nulo.
- Conversor: usa tipos explícitos para TON_IoT e apenas cabeçalhos para a união
  de colunas. Cada conversão futura escreverá um `.schema.json`; o manifest
  registra a versão. O fallback para datasets genéricos permanece.
- Pré-processamento: valida também chamadas diretas e usa `model_feature_types`
  antes de escolher CPU/GPU. Ambos recebem as mesmas listas numéricas e
  categóricas. Imputação, escala, codificação e redução continuam sendo ajustadas
  somente no treino; as implementações CPU/GPU não precisam ser numericamente
  idênticas.
- Relatórios: a CLI escreve `schema_normalization_report.json`; o
  pré-processamento escreve `preprocessing_schema_report.json`, e os metadados
  de pré-processamento recebem a versão. A segunda validação é sem estatísticas.
- Configuração: `log1p_numeric: false` continua desabilitando log. O campo novo
  `log1p_columns` aceita nomes exatos do conjunto elegível; `null` usa a lista
  padrão e `[]` a desabilita. `log1p_patterns` continua parseável para YAMLs
  existentes, mas é ignorado, com mensagem de log quando a lista padrão é usada.
  Nomes sem domínio explícito de log são rejeitados. Não há clipping.

Métricas e artefatos antigos não são diretamente intercambiáveis com os novos:
`src_bytes` e `http_trans_depth` passam a quantitativos; códigos DNS/HTTP passam a
categorias; flags têm codificação canônica; `-` passa a ausência declarada; os
11 quantitativos têm elegibilidade explícita para log, incluindo comprimentos
HTTP. Portas, quando permitidas pelas políticas existentes, são categóricas.
Pré-processadores ajustados anteriormente precisam ser tratados como artefatos
da representação antiga. Não houve reexecução nem alteração dos resultados.
Entradas inválidas antes toleradas podem agora interromper a execução com
diagnóstico; não existe evidência de que todos os milhões de registros já
satisfaçam o contrato.

## Ambiguidades mantidas explícitas

- `http_status_code=0` está nos exemplos locais; não foi possível decidir se
  indica indisponibilidade no processamento de origem. Preserva-se a categoria
  `0`, sem impor a faixa 100–599. Códigos DNS também preservam zero, cujo
  significado depende do campo, sem vocabulário aprendido.
- `http_orig_mime_types` e `http_resp_mime_types` podem conter coleções
  serializadas. Não foi inferida uma gramática ou transformado o campo em
  múltiplas categorias. `weird_addl` permanece texto opaco.
- Não foi validado um vocabulário completo de `type`, versões, serviços ou
  suítes criptográficas. A semântica nominal é definida; a ontologia completa
  não foi inventada a partir de poucos registros.
- O mapeamento detalhado de nomes `src`/`dst` a originador/respondedor, e a
  proveniência de todos os valores processados, não foram auditados globalmente.
  Essa limitação não altera a evidência de que bytes/pacotes são quantidades.

## Validação pequena

`tests/test_dataset_schema.py` cobre o contrato das 47 colunas, tipos físicos
distintos entre partições, contagens, códigos, indicadores, sentinelas, tokens
inválidos, infinitos, domínios, precisão de inteiros, idempotência, identidade de
linhas, exclusão de alvos, colunas desconhecidas, log explícito, relatórios CSV,
compatibilidade com Parquet antigo e conversão de duas fixtures minúsculas.
Também verifica independência das estatísticas de treino frente a mudanças em
validação/teste, despacho CPU/GPU idêntico e compatibilidade das colunas
temporais derivadas. O teste de helpers GPU substitui CuPy por NumPy; o despacho
GPU usa um stub. Isso não constitui uma execução CUDA.

Comando de validação: `python -B -m pytest -o addopts='' -q -p no:cacheprovider`.
Resultado: **79 testes passaram em 4,64 s**, incluindo **53 casos novos** neste
arquivo de testes. `git diff --check` não encontrou erros de whitespace.
O teste de integração já existente usa apenas 120 registros sintéticos.
