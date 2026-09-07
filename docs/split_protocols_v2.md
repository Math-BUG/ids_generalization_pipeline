# Protocolos de split — Incremento 3

Entrada: população elegível após a política de qualidade congelada. Nenhuma estatística de modelo ou desempenho participa da construção dos splits. O schema, a quarentena, as features e o preprocessing permanecem intactos.

| Protocolo / versão | Pergunta científica | Contrato |
|---|---|---|
| `group_stratified/2.0.0` | Generalização para combinações não vistas dos atributos de grupo | Grupos inteiros e disjuntos; busca determinística; suporte de classes e proporções validados |
| `temporal/2.0.0` | Treinar no passado e avaliar em períodos globais posteriores | Máximo de treino < mínimo de validação; máximo de validação < mínimo de teste; buckets inteiros |
| `temporal_per_class/2.0.0` | Exemplos anteriores de cada classe precedem seus exemplos posteriores | Contrato temporal dentro de cada classe; análise **auxiliar**, com cronologia global reportada separadamente |

## Configuração e objetivos de tamanho

Preserva-se o significado existente de `test_size`: fração da população total. `val_size` é a fração do restante após reservar teste. Portanto `test_size: 0.25` e `val_size: 0.20` significam **60% / 15% / 25%**, sem alterar os arquivos de configuração existentes. Valores reais sempre constam do manifesto.

Opções adicionais de split são lidas por `PipelineConfig.extra`, mecanismo já existente para campos extras de YAML:

```yaml
group_split_tolerance: 0.02
group_split_candidates: 16
split_min_class_support: 1
split_small_support_threshold: 30
split_export_csv: false
```

A tolerância é absoluta: dois pontos percentuais para cada partição. O mínimo inicial é uma observação de cada valor observado de `label` e `type` em cada partição do protocolo de grupos. O alerta de suporte pequeno (1–29 observações) é descritivo; **não** garante estimativas estáveis. Alterar esses parâmetros é uma decisão explícita e altera o hash do protocolo configurado. Protocolos temporais não exigem cobertura global de todas as classes: isso mascararia o surgimento de classes posteriores.

## Grupos

Todas as `group_cols` devem existir e ser distintas. A identidade é a combinação exata dos valores normalizados, incluindo ausência como um valor distinto. Não se concatenam campos com delimitadores ambíguos e não existe fallback para um grupo por linha. Para TON_IoT: `src_ip`, `dst_ip`, `service`, `proto`. Essa combinação não identifica necessariamente um dispositivo novo.

A busca agrega contagens por grupo e por classe, mantendo todos os registros. Produz candidatos com alocação gulosa determinística, priorizando grupos grandes e usando variações de ordem derivadas da seed. A função de alocação considera tamanho, `label` e `type`, com normalização das classes para incluir as raras. A seleção entre candidatos minimiza, nesta ordem:

1. Número de células classe/partição abaixo do suporte mínimo.
2. Soma do excesso de desvio além da tolerância de tamanho.
3. Desvio absoluto de tamanho somado ao erro médio de distribuição de classes.

O relatório inclui todos os candidatos, o escolhido, tamanhos dos grupos e concentração de cada classe em grupos. Grupos dominantes acima de todos os limites e classes presentes em menos de três grupos são certificados simples de inviabilidade das respectivas restrições. **Busca sem solução não é prova de impossibilidade matemática.** Uma solução fora dos contratos é apenas candidata diagnóstica: não é liberada como split congelado para treinamento. Não se relaxa a tolerância nem se fragmentam grupos.

## Tempo

Timestamps são convertidos para UTC; valores ausentes ou inválidos impedem protocolos temporais. Com `temporal_bucket_freq: 1s`, um segundo inteiro permanece na mesma partição. Com frequência nula, timestamps exatamente iguais ainda permanecem juntos. A interpretação numérica de unidade conserva a configuração existente; a auditoria TON_IoT usa explicitamente segundos Unix.

Os dois cortes são escolhidos entre buckets completos para minimizar a soma dos desvios absolutos das contagens desejadas. O algoritmo considera todos os primeiros cortes e os pontos candidatos ótimos do segundo corte, com desempate determinístico. Todos os três splits precisam ser não vazios. Buckets grandes podem gerar proporções diferentes das desejadas; isso é reportado sem dividir o bucket.

No protocolo por classe, cada valor de `type` (ou `label`, se `type` não existir) precisa ter pelo menos três buckets. Todas as classes inviáveis são listadas e a execução aborta, sem excluir classes ou repartir linhas empatadas. O relatório acusa ausência em **qualquer** partição, inclusive validação. Buckets iguais não se dividem dentro da mesma classe; classes diferentes do mesmo bucket podem ocupar splits diferentes, motivo pelo qual a cronologia global é verificada e pode ser falsa.

Os relatórios distinguem:

- `known_in_train`: classes observadas no treino;
- `first_seen_validation`: classes que aparecem na validação e não no treino;
- `first_seen_test`: classes no teste ausentes tanto no treino quanto na validação;
- `unseen_from_train_in_test`: classes de teste ausentes do treino, mesmo se já apareceram na validação;
- `closed_set_test_classes`: interseção das classes do teste e do treino.

Não há implementação de classificador open-set nesta etapa.

## Artefatos e reutilização

`split_manifest.json` registra versão, pergunta, seed relevante, parâmetros, população, contagem original e elegível, proporções reais, distribuições com zeros explícitos, classes ausentes, suportes pequenos, sobreposição de linhas e grupos, cronologia e resultado final dos contratos. Sobreposição de grupos em protocolos temporais é informação descritiva, não uma falha desses protocolos.

Os índices em `splits/{train,val,test}_indices.npy` são posições `iloc` na população elegível, ordenadas e disjuntas. Não são posições originais dos arquivos. Na auditoria real, `eligible_original_positions.npy` mapeia cada posição elegível para a posição original global; o manifesto da qualidade identifica a ordem dos arquivos, suas contagens e exclusões. Nenhuma deduplicação ocorre.

Hashes SHA256 dos índices usam inteiros little-endian de 64 bits. Hashes de identidade vinculam índices originais à população. O hash completo inclui versão, parâmetros, seed, identidade da população e os três hashes de índices. A população também recebe um hash dos índices e valores ordenados das colunas de split, em blocos de 100 mil linhas, para detectar mudanças na ordem ou em subconjuntos de depuração. Esse hash usa `pandas.util.hash_pandas_object`; registre as versões das dependências para reprodução entre ambientes.

Para reutilização sem regeneração, configure:

```yaml
frozen_splits_dir: reports/increment_3/real/group_stratified
```

`create_splits` passa então a carregar os arrays, recalcular validações e verificar o hash contra a população/configuração fornecida. Alteração de identidade, conteúdo, ordem, parâmetros ou arrays causa erro. `load_frozen_splits` também está disponível diretamente para os futuros seletores. Arquivos congelados com hash diferente não são sobrescritos: use outro diretório para outra versão/configuração. Somente relatórios com `validation_ok: true` podem ser consumidos por esse caminho.

## Auditoria real sem modelos

```powershell
python -B scripts/audit_ton_iot_splits.py
```

O script verifica os SHA256 das 23 fontes Parquet contra o checkpoint congelado, lê em blocos de 100 mil linhas, reaplica a política original de quarentena e normaliza apenas as colunas projetadas necessárias. A auditoria integral anterior continua sendo a evidência de conformidade das demais features. O script mantém apenas sete colunas compactas para construir os splits; não lê `cluster_assignments.csv`, não reconverte fontes, não executa preprocessing, clustering ou treinamento. Os grupos não são features do modelo.

Resultados antigos de splits podem deixar de ser comparáveis: a alocação de grupos muda, cortes temporais passam a respeitar buckets e proporções reais, e a validação estrita elimina antigos fallbacks. Incremento 4 permanece fora do escopo.
