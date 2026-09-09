# Fase 1D: reproduzir o split de grupos no ClusterGPU

Preparação somente. Nenhum job é submetido automaticamente. O algoritmo de split,
schema e quarentena dos Incrementos 1–3 permanecem inalterados.

## Inspeção e escolha do caminho

- `split_protocols.py` já implementa manifesto, hashes, invariantes e carregamento
  congelado com verificação da população e configuração.
- `splitting.py` já salva os três arrays `.npy` e aceita `frozen_splits_dir`.
- `audit_ton_iot_splits.py` lê por blocos somente oito colunas, aplica a quarentena
  e o schema e retém sete colunas compactas. Seu `main` executa três protocolos e
  os caminhos do checkpoint são absolutos; usamos apenas `read_population`.
- `report_split_checkpoint.py` depende de caminhos históricos, `frozen_before.json`
  e `tests.xml` de outra execução. Não será executado nesta fase.
- O config principal GPU herda `timestamp_unit: auto`. A auditoria de referência
  usou `s`, valor que participa do hash. O config exclusivo desta fase registra os
  parâmetros originais explicitamente, sem mudar o config de experimentos.

## Entradas obrigatórias

1. Os mesmos 23 Parquet do Incremento 1, sem reconversão ou reordenação.
2. O **`data_quality_report.json` original da auditoria Parquet do Incremento 1**:
   lista ordenada de fontes, IDs relativos, SHA256 de conteúdo, exclusões e ID da
   população. Não substitua pelo relatório CSV nem por contagens copiadas à mão.

O relatório antigo não está presente neste checkout. Copie-o do arquivo/backup
auditado para um caminho persistente no cluster antes de submeter o job.
Sem ele, a preparação pode ser revisada, mas a reprodução não pode ser atestada.
Os caminhos Windows no JSON são remapeados em uma cópia de trabalho usando
`data-root/source_id`; ordem, IDs e hashes originais permanecem intactos.

## Fluxo proposto

1. Conferir metadados da referência, conjunto e ordem das fontes e saída nova fora
   do repositório e dos dados. Não sobrescrever checkpoints existentes.
2. Em diretório temporário persistente, verificar SHA256 de cada fonte por leitura
   sequencial; ler Parquet em blocos de 100 mil linhas. Aplicar as funções existentes
   de quarentena e normalização. Comparar o ID da população com o Incremento 1.
3. Regenerar somente `group_stratified/2.0.0`, seed 42, grupos `src_ip`, `dst_ip`,
   `service`, `proto`, com os mesmos 16 candidatos e tolerância de 0,02.
4. Verificar 22.339.021 originais, 869 quarentenadas, 22.338.152 elegíveis;
   train 13.575.269, val 3.244.049, test 5.518.834; cobertura completa de linhas,
   ausência de duplicação e sobreposição, grupos disjuntos e as mesmas 10 classes
   `type` presentes em todos os splits. Conferir também cobertura de `label`.
5. Calcular o hash pelo algoritmo existente e exigir igualdade exata com
   `sha256:828c57acef5ca3a953762a2dcf836dafd6b8dd513dd476c88ba924841a5f0a2a`.
   O prefixo `sha256:` é apenas o formato usado pelo código.
6. Salvar arrays, manifesto e mapa de posições originais; recarregar pelo caminho
   congelado para verificar os arrays salvos. Publicar o diretório final somente
   após aprovação. Falhas deixam diagnósticos em `.group_stratified_v2.pending_*`;
   esses diretórios não devem ser usados em experimentos.

A leitura projetada não constitui nova auditoria de todas as features: a auditoria
integral do Incremento 1 continua sendo a evidência, vinculada aos SHA256 exatos das
fontes. A fase não executa preprocessing, KMeans ou RF e não lê assignments antigos.

## Comandos propostos (não executados)

Na raiz do projeto no cluster, usando o caminho real do relatório preservado:

```bash
python scripts/validate_ton_iot_group_split.py \
  --checkpoint "$HOME/dados/ids_generalization_splits/inputs/data_quality_report.json" \
  --data-root "$HOME/dados/datasets/TON_IoT_parquet" \
  --output "$HOME/dados/ids_generalization_splits/group_stratified_v2" \
  --preflight-only

sbatch scripts/validate_ton_iot_group_split.slm \
  "$HOME/dados/ids_generalization_splits/inputs/data_quality_report.json" \
  "$HOME/dados/datasets/TON_IoT_parquet" \
  "$HOME/dados/ids_generalization_splits/group_stratified_v2"
```

O preflight verifica apenas caminhos e metadados; não calcula hashes das fontes
nem gera split. Ative a `.venv` do ambiente validado antes desse comando.

Pedido Slurm: **1 nó, 1 tarefa, 8 CPUs, 128 GiB RAM, 12 horas**, partição
`scientific`, QoS `scientific-qos`. Sem GPU: o split congelado usa pandas/NumPy,
inclusive quando os experimentos posteriores usam GPU. Os módulos Python/3.13.5
e CUDA/12.9.1 e a `.venv` são os validados. CPU/memória/tempo são uma reserva
conservadora, não uma medição; a alocação gulosa é predominantemente sequencial.
O dataset inteiro não é mantido: somente as sete colunas e estruturas dos grupos.

## Saída persistente e reutilização

O diretório final terá `split_manifest.json`, `checkpoint_verification.json`,
`execution_environment.json`, `checkpoint_config.yaml`, `reference_relocated.json`,
`data_quality_population.json`, `reuse_split.yaml`, `eligible_original_positions.npy`
e `splits/{train,val,test}_indices.npy`, além dos pequenos relatórios existentes.
Os quatro arrays int64 ocupam aproximadamente 341 MiB; os manifestos são resumos,
sem CSV individual de milhões de linhas.

O hash do manifesto é SHA256 de JSON canônico de versão, população, parâmetros,
seed e hashes dos três arrays int64 little-endian. Não é o SHA256 do arquivo JSON.
O hash da população inclui ordem, índices e conteúdo das colunas de split e usa
`pandas.util.hash_pandas_object`; versões do ambiente e hashes do código ficam
registrados. Divergência de ambiente, conteúdo, ordem ou representação não será
contornada alterando o hash esperado, seed, tolerância ou critérios.

Para experimentos futuros, incorpore os campos de `reuse_split.yaml` à configuração
completa de experimento, preservando suas features/modelos/backend. O loader YAML
não faz merge automático de arquivos e não expande `$HOME` em valores YAML: use
o caminho absoluto gerado. Não use o fragmento como config completo de treinamento.
`create_splits` então carrega os arrays existentes e revalida população, parâmetros,
hashes e invariantes, sem nova busca de grupos. A validação ainda percorre a
população e tem custo; o que é evitado é a regeneração dos índices.
