# Organização dos materiais da IC — Generalização

Este repositório contém a continuidade da pesquisa de IDS para IoT: revisão de políticas de atributos, controle de vazamento, separações mais rigorosas, execução GPU e estudo de estabilidade de clusters.

## Materiais

| Conteúdo | Local |
| --- | --- |
| Pipeline modular | `src/ids_pipeline/` |
| Configurações experimentais | `configs/` |
| Executores e submissões Slurm | `scripts/` |
| Testes | `tests/` |
| Métodos, auditorias e propostas | `docs/` |
| Evidências das execuções consolidadas | `resultados_fase_2/` |
| Proveniência | `provenance/` |
| Figuras selecionadas e resumos para consulta | `figuras/`, `resultados/` (futuros) |

O `README.md` existente tem instruções operacionais e deve continuar funcionando. Guardar dados brutos e outputs volumosos fora do Git, conforme a política atual de `.gitignore`.

## Status científico

- A avaliação consolidada da fase 2 usa a população TON_IoT elegível e sua própria política de atributos. Não é diretamente comparável ao recorte publicado no SBCUP apenas como um efeito de escala.
- Propostas e scripts da fase 2.1 **não significam** que toda a grade experimental foi executada.
- Não declarar validação externa entre datasets diferentes nem implementação operacional em gateways enquanto não existirem evidências.

## Trabalhos relacionados

- [ERAD-SE 2026 — CPU/GPU](https://github.com/Math-BUG/experimento_erad)
- [SBCUP 2026 — agrupamento + classificação](https://github.com/Math-BUG/iot-fog-ids-two-phase)

O arquivo `Fontes_projeto_IDS/artigo/Artigo_SBCUP_2026.pdf` é uma cópia de referência já presente neste repositório; a versão canônica do artigo deve ficar no repositório SBCUP. Remoção desta cópia deve ser feita apenas após conferir e transferir o arquivo, sem perda de conteúdo.
