# Fase 2.1 — implementação para revisão antes da execução

Implementação da [proposta aprovada](phase_2_1_clustering_optimization_proposal.md). Nenhum job foi submetido, nenhuma matriz real foi executada e nenhum dado TON_IoT foi carregado nesta entrega. Os testes usam fixtures e doubles GPU explicitamente identificados; não são validação CUDA real.

## Arquivos e responsabilidades

| Arquivo | Responsabilidade |
|---|---|
| [Config final](../configs/phase_2_1_clustering_optimization.yaml) | Grade, dados/split pai, parâmetros efetivos e flags congeladas |
| [Contrato comum](../scripts/phase_2_1_common.py) | Config/proveniência, amostras, transformação sem fit, métricas internas, flags, Pareto e validação de decisões |
| [Preparação](../scripts/prepare_phase_2_1_clustering.py) | Auditoria congelada, índices, cache R0, R10/R15 ajustados uma vez, reutilização R20 |
| [Executor A/B](../scripts/run_phase_2_1_clustering.py) | Dois modos isolados, sem rótulos, com a mesma matriz por representação e seeds |
| [Relatório e decisões](../scripts/report_phase_2_1_clustering.py) | Estabilidade, agregações, frente de Pareto e congelamento explícito |
| [Avaliação externa](../scripts/evaluate_phase_2_1_external.py) | Acesso aos rótulos somente depois das decisões A e B verificadas |
| [Testes](../tests/test_phase_2_1_clustering.py) | Contratos, casos inválidos, replay, flags, Pareto e fluxo integrado pequeno |
| [Hashes de fixture](../tests/phase_2_1_sample_hashes.json) | Valores conhecidos e distintos para as duas amostras |

Nenhum arquivo de `src/ids_pipeline` é modificado por esta implementação. As alterações anteriores da correção da Fase 2 são preservadas, assim como resultados e proveniência históricos. A proveniência 2.1 é separada, em `provenance/phase_2_1_source_provenance.json`; não substituir a proveniência do baseline pai.

## Flags finais: clustering_degeneracy/1.0.0

Estes limiares foram definidos antes da execução científica da grade, sem consultar rótulos, AMI/V-measure/purity externos ou resultados da busca. São sinais operacionais descritivos, não limites universais de qualidade. Foram escolhidos para sinalizar domínio de grande parte da massa, maioria de clusters muito pequenos ou forte redução do número efetivo; não impõem balanceamento.

| Flag | Regra exata |
|---|---|
| dominant_cluster | maior fração ≥ 0,90 |
| microclusters_lt10 | pelo menos 50% dos k clusters têm n_j < 10 |
| microclusters_lt100 | pelo menos 50% dos k clusters têm n_j < 100 |
| microclusters_lt1000 | pelo menos 50% dos k clusters têm n_j < 1000 |
| low_entropy | H/ln(k) < 0,50 |
| low_effective_ratio | exp(H)/k < 0,25 |
| top5_concentration | top-5 ≥ 0,95, somente quando k > 5 |
| top10_concentration | top-10 ≥ 0,99, somente quando k > 10 |

Top-10 para k=10 é `not_applicable`, nunca uma flag automaticamente verdadeira. Registrar valor, limiar, operador, estado e motivo por execução. As três flags de microclusters são relacionadas; não somá-las como escore. Limiares absolutos de tamanho devem ser interpretados com n e k. Flags não tornam um run INVALID nem o removem do Pareto. A configuração possui hash científico fixo, que também protege a versão e os valores dos limiares.

Invalidade continua separada: execução com erro; NaN/Inf em matrizes/centros/saídas/métricas; cardinalidade incompatível; GPU errada; artefato inválido; q diferente de k no conjunto de ajuste. Ausência de clusters apenas nos 20k de métricas não viola o contrato de ajuste. Caso a amostra tenha apenas um cluster, silhouette fica null com motivo e o run continua tecnicamente válido; a configuração não possui todos os eixos para a comparação completa e não é promovida silenciosamente. Não trocar a amostra para obter um resultado mais favorável.

## Representações e congelamento

O preprocessing pai é verificado e copiado para a preparação: oito quantitativas com estatísticas pós-log1p do TRAIN e encoder nominal com treze estados. O adaptador segue o mesmo cálculo pandas/CuPy do pipeline, sem aprender novas estatísticas. R0 mantém 21 colunas. R20 usa o redutor pai verificado, sem refit. R10/R15 mantêm os parâmetros do redutor pai salvo n_components e são ajustados uma única vez no TRAIN completo, seed 42. Não há fallback PCA nem refit da representação por seed de KMeans.

Se o bundle pai não puder ser carregado/verificado no ambiente esperado, a preparação falha. Não foi implementada uma substituição silenciosa para R20: uma alternativa ao artefato pai exigiria diagnóstico e revisão separada. O SHA do arquivo é registrado na preparação e conferido posteriormente. Isso garante a imutabilidade a partir da preparação; não inventa um hash histórico do bundle que não existia no manifesto do pai.

Os arquivos `development_indices.npy` e `metric_sample_indices.npy` possuem hashes separados. Também são persistidos offsets no TRAIN/desenvolvimento, posições globais elegíveis, IDs originais e a amostra B da silhouette histórica. Todas as matrizes e modelos têm inventário com checksum, tamanho, dtype/shape quando aplicável e vínculo com o digest da preparação. Stage A/B verificam os artefatos antes de usá-los. O manifesto principal mantém estados e contagens; os JSONs de run são a evidência canônica.

## Pareto e decisões

Unidade de promoção: (representação, k), sempre com as três seeds. O Pareto tem interface fechada de cinco eixos internos: maximizar silhouette média, maximizar CH médio, minimizar DB médio, maximizar AMI mínima entre seeds, minimizar pior fração do maior cluster. Não aceita colunas de alvos ou scores externos. Empates permanecem na frente; flags não são filtro. É exportada também a frente sem o eixo de concentração.

O relatório registra todas as execuções planejadas, inclusive inválidas/ausentes. Uma configuração com seed inválida ou métrica exigida indefinida fica fora do Pareto elegível; nenhuma seed é omitida para favorecer o agregado. Para congelar, todos os runs planejados devem ter registros. O status científico final permanece incompleto se ficaram execuções inválidas de A, ainda que haja finalistas avaliados; isso aparece nas contagens e não é apresentado como matriz integralmente bem-sucedida.

O congelamento exige comando explícito com candidato(s), responsável, justificativa e declaração `--approve-internal-only`. Escreve uma decisão exclusiva, com hashes dos runs e do relatório que foi efetivamente revisado. Não recalcula estabilidade/timers durante a aprovação. Não aceita mais de três pares nem seleção de seed. A decisão B só admite candidatos de A. Relatórios após congelamento não reescrevem a evidência aprovada.

`external_gate` verifica decisões A e B, seus encadeamentos, o snapshot da preparação e os artefatos de todos os runs válidos antes de chamar o carregador de dados ou abrir métricas históricas. Stage A/B só recebem cache de features e índices; não abrem `label/type`. A auditoria oficial de população/split na preparação continua verificando cobertura de classes como invariante congelado, sem usar isso na amostragem ou promoção. Este é um bloqueio verificável do fluxo de execução; não pretende impedir que um usuário com acesso aos arquivos altere código ou leia rótulos fora dele.

Após a decisão B, a avaliação externa produz métricas train/val/test para todas as seeds dos finalistas A e para o modelo histórico salvo, sem novos fits. A referência histórica mantém custos de fit e estabilidade entre seeds como não medidos; não recebe resultados de estabilidade da amostra A. O relatório apresenta tabelas e diferenças descritivas, sem declarar um vencedor por score externo. Nenhum cluster_id definitivo para RF é gerado automaticamente.

## Comandos para revisão, sem submissão automática

Antes do rsync, na origem:

```bash
python scripts/prepare_phase_2_1_clustering.py --validate-config
python scripts/prepare_phase_2_1_clustering.py --write-provenance
```

Estes dois modos não carregam dados nem iniciam CUDA. Transferir também `provenance/`, `tests/` e `docs/`. O snapshot inclui as fontes locais pendentes e o commit de origem; arquivos alterados após a geração exigem nova proveniência antes da preparação. Não reutilizar uma preparação sob novo código/config sem revisão e nova raiz de experimento.

Fluxo dos executores, apresentado para revisão; não executado nesta entrega:

```bash
# Preparação comum, sem KMeans/RF:
python scripts/prepare_phase_2_1_clustering.py

# Cada tarefa A recebe uma das quatro representações; exemplo:
python scripts/run_phase_2_1_clustering.py --stage A --representation R0

# Depois das quatro tarefas A:
python scripts/report_phase_2_1_clustering.py --stage A

# Exemplo de sintaxe, NÃO uma escolha antecipada de vencedor:
python scripts/report_phase_2_1_clustering.py --stage A --freeze \
  --candidate R20:k30 --approved-by NOME --rationale 'JUSTIFICATIVA INTERNA REVISADA' \
  --approve-internal-only

# Somente depois da decisão A, índice de finalista congelado:
python scripts/run_phase_2_1_clustering.py --stage B --finalist-index 0
python scripts/report_phase_2_1_clustering.py --stage B

# Congelamento B usa a mesma sintaxe --freeze, após revisar seus resultados.
# Só então:
python scripts/evaluate_phase_2_1_external.py
```

Esses comandos de execução devem rodar nos jobs alocados, não no nó de login. O comando de congelamento não autoriza sozinho a submissão de outro job. Os candidatos nos exemplos só são aceitos se estiverem na frente do relatório real e forem aprovados pelo usuário.

## Slurm e recursos

| Script | Escopo | Recursos |
|---|---|---|
| [Preparação](../scripts/run_phase_2_1_prepare.slm) | Uma preparação compartilhada | 32 CPUs, 256 GiB, 1 A100, 2 h |
| [Stage A](../scripts/run_phase_2_1_stage_a.slm) | Array 0–3%1; 24 runs por representação | 32 CPUs, 256 GiB, 1 A100, 12 h por tarefa |
| [Stage B](../scripts/run_phase_2_1_stage_b.slm) | Array definido após congelar F finalistas; 3 runs por tarefa | 32 CPUs, 256 GiB, 1 A100, 12 h por tarefa |
| [Externa](../scripts/run_phase_2_1_external.slm) | Predições e relatório depois das decisões | 32 CPUs, 256 GiB, 1 A100, 12 h |

scientific/scientific-qos, `gpu:a100:1`, Python/3.13.5-GCCcore-14.3.0, CUDA/12.9.1, `.venv`, backend GPU. Arrays têm concorrência 1; um lock exclusivo também impede operações concorrentes no mesmo experimento. Stage B exige SLURM_ARRAY_TASK_ID e valida o índice contra a decisão; não inclui um array de três candidatos presumidos no arquivo.

Após revisão e aprovação explícita, a ordem de submissão seria preparação, A, pausa para decisão, B, pausa para decisão, externa. A forma do array B é `--array=0-(F-1)%1`, com limite numérico resolvido a partir do arquivo congelado. Nenhum script chama sbatch ou encadeia automaticamente etapas.

GNU time envolve cada processo quando disponível; scontrol registra a alocação e logs são externos ao Git. Fit, métricas, I/O e total de cada KMeans são separados; `predict_seconds=0` em A/B indica uso de `labels_` do fit para treino e estabilidade, sem chamada independente a predict. Avaliação externa mede suas predições em lotes de 65.536. Timers de setup e estabilidade por grupo ficam separados. Memória observada nas fronteiras não é pico contínuo de VRAM.

## Retomada e integridade

Preparação recusa raiz existente. A/B reutilizam somente runs válidos com mesmos hashes; tentativas inválidas/incompletas exigem `--retry-invalid`, que conserva o registro anterior e grava novos artefatos em outro diretório de tentativa. Essa opção não pode executar depois da decisão da própria etapa. Arquivos corrompidos não são silenciosamente aceitos como cache válido.

Um SIGKILL pode deixar `.phase21.lock`. Não removê-lo automaticamente: primeiro verificar no Slurm que não há processo ativo e revisar o estado. A avaliação externa recusa diretório pré-existente; uma falha externa deve ser preservada/arquivada explicitamente antes da repetição. Nada é apagado automaticamente, nem o baseline pai ou cluster_assignments.csv antigo é lido.

## Estimativa de duração e limites

Referência observada da Fase 2: 314,34 s no estágio de clustering, incluindo métricas/predições/I/O; não é medida de fit isolado. 96 vezes esse estágio equivalem a 8,38 h como referência de escala, não previsão exata. Planejamento provisório: preparação em dezenas de minutos; A aproximadamente 2–12 h agregadas; B aproximadamente 1–3 h; avaliação externa acrescenta dezenas de minutos a horas. Silhouette repetida pode dominar CPU. Rever a previsão com os primeiros runs já previstos na grade, sem alterar a metodologia. As reservas Slurm são maiores e não representam consumo esperado.

O cache R0 ocupa cerca de 1,06 GiB e as quatro matrizes de desenvolvimento cerca de 0,25 GiB. Atribuições amostrais são uint8 (~96 MB em A); não há CSV completo por observação. Conferir quota/storage e recursos novamente no cluster antes de submeter.

## Evidência de validação local

Hashes conhecidos da fixture (TRAIN de 120 posições em ordem inversa, desenvolvimento de 60, métricas de 20):

- development_indices: `sha256:52642ca6e2387bebce0065875920c1c7e903d0fafa6ce8b052b3f0265f0e6e58`.
- metric_sample_indices: `sha256:60d7c4423ad5f42f7424ee0def7076ade16726de0e9083dab877e8dd4b874159`.

Esses são hashes de testes pequenos, não das futuras amostras TON_IoT. Os hashes reais só serão gerados na preparação autorizada no cluster. A fixture integrada usa 260 linhas, 180 de treino, 90 de desenvolvimento e 30 nas métricas, com 12 runs A pequenos e três B; não executa a grade científica de 96 configurações. Doubles de cuML usam sklearn exclusivamente nos testes e não demonstram disponibilidade CUDA.

Validação final: **37 testes específicos passaram** (19,86 s); **326 testes passaram na suíte completa final** (87,34 s), usando `python -m pytest -o addopts= -q -p no:cacheprovider`. A suíte completa inclui as guardas finais de validação do redutor pai. Nenhum teste CUDA real foi executado.

Também passaram: loader/validador da configuração final, equivalência entre YAML da proposta e arquivo final, sintaxe Python, links locais, UTF-8/whitespace e `bash -n` dos quatro Slurm. `git diff --check` passou; como os arquivos da Fase 2.1 ainda são novos/não rastreados, seu whitespace foi verificado diretamente. Foram conferidos 22 hashes de código/configuração contra a proveniência da Fase 2, sem divergência no núcleo congelado.

O `git status` mantém as alterações anteriores da Fase 2 (`docs/phase_2_full_gpu_baseline_proposal.md`, `provenance/phase_2_source_provenance.json`, `scripts/run_phase_2_full_gpu_baseline.py`, `src/ids_pipeline/supervised.py`, `tests/test_phase_2_full_gpu_baseline.py`) e os resultados/testes anteriores. Os arquivos 2.1 aparecem como não rastreados (`??`); nenhum staging ou commit foi feito. O diff rastreado continua exibindo somente as mudanças anteriores da Fase 2 (99 inserções e 14 remoções em cinco arquivos); não confundir esse resumo com o tamanho da nova implementação 2.1.

A implementação permanece aguardando revisão final antes de qualquer sbatch. A proveniência 2.1 é gerada após fechar os arquivos desta entrega e será verificada novamente antes da preparação.
