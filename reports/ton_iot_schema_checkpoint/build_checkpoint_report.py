"""Build a report only from completed, saved audit aggregates; no dataset reads."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent


def build():
    parquet = json.loads((ROOT / "parquet/audit.json").read_text())
    csv = json.loads((ROOT / "csv_complete/audit.json").read_text())
    assert parquet["complete"] and csv["complete"]
    assert parquet["file_count"] == csv["file_count"] == 23
    assert parquet["rows"] == csv["rows"]
    rows = parquet["rows"]
    bad = {name: item for name, item in parquet["columns"].items() if item["invalid_cells"]}
    invalid = sum(item["invalid_cells"] for item in bad.values())
    token_verification = json.loads((ROOT / "invalid_token_verification.json").read_text())
    assert token_verification["complete_for_affected_files"]
    assert sum(token_verification["token_counts"].values()) == bad["src_bytes"]["invalid_cells"]
    bad_csv = {name: item for name, item in csv["columns"].items() if item["invalid_cells"]}
    counts_match = {name: item["invalid_cells"] for name, item in bad.items()} == {
        name: item["invalid_cells"] for name, item in bad_csv.items()}
    assert all(item["source_unchanged_size_mtime"] for a in (csv, parquet) for item in a["files"])
    file_rows = []
    for a, b in zip(csv["files"], parquet["files"]):
        assert a["file"].replace(".csv", "") == b["file"].replace(".parquet", "")
        assert a["rows"] == b["rows"]
        file_rows.append({"partition": int(a["file"].split("_")[-1].split(".")[0]),
                          "rows": a["rows"], "csv_cols": len(a["columns_present"]),
                          "parquet_cols": len(b["columns_present"]),
                          "csv_missing": ", ".join(a["columns_absent"]) or "nenhuma",
                          "csv_invalid": sum(v["invalid_cells"] for v in a["columns"].values()),
                          "parquet_invalid": sum(v["invalid_cells"] for v in b["columns"].values())})
    col_rows = []
    dtype_rows = []
    for name, item in parquet["columns"].items():
        original = csv["columns"][name]
        col_rows.append({"column": name, "csv_nulls": original["native_nulls"],
                         "csv_sentinels": original["registered_sentinels"],
                         "parquet_nulls": item["native_nulls"],
                         "parquet_sentinels": item["registered_sentinels"],
                         "invalid_csv": original["invalid_cells"], "invalid_parquet": item["invalid_cells"]})
        dtype_rows.append({"column": name, "parquet_dtype": ", ".join(item["physical_dtypes"]),
                           "pandas_dtype": ", ".join(item["pandas_dtypes"]),
                           "csv_absent_rows": original["absent_rows"]})
    issue_rows = []
    for name, item in bad.items():
        for kind, issue in item["issues"].items():
            if issue["count"]:
                issue_rows.append({"column": name, "rule": kind, "count": issue["count"],
                                   "examples": "; ".join(e["value"] for e in issue["examples"])})
    checks = []
    fields = {"conversion_failure": "Falha de conversão", "infinite": "Infinitos",
              "negative_nonnegative_domain": "Negativos proibidos", "nonintegral": "Não integrais em domínio integral",
              "ports_out_of_range": "Portas fora de [0, 65535]", "invalid_label": "Label inválido",
              "unexpected_boolean": "Booleanos inesperados", "unsafe_integer_precision": "Precisão inteira insegura",
              "missing_required": "Alvo nulo/ausente", "domain_violation": "Violação de domínio (união por coluna)"}
    for field, label in fields.items():
        values = {}
        for fmt, audit in (("csv", csv), ("parquet", parquet)):
            values[fmt] = sum(item.get(field, item["issues"].get(field, {}).get("count", 0))
                              for item in audit["columns"].values())
        checks.append({"check": label, **values})
    verdict = "NÃO está compatível" if not (parquet["compatible"] and csv["compatible"]) else "Está compatível"
    summary = {"rows_per_representation": rows, "logical_partitions": 23, "physical_files_read": 46,
               "invalid_cells_parquet": invalid, "invalid_rate_parquet": invalid / rows,
               "invalid_counts_csv_parquet_match": counts_match,
               "csv_compatible": csv["compatible"], "parquet_compatible": parquet["compatible"],
               "checks": checks, "file_rows": file_rows, "issue_rows": issue_rows}
    (ROOT / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    title = "Checkpoint do esquema TON_IoT"
    now = datetime.now(timezone.utc).isoformat()
    sources = [{"id": "audit_csv", "label": "Auditoria dos 23 CSVs Network_dataset_1..23",
                "path": "csv_complete/audit.json"},
               {"id": "audit_parquet", "label": "Auditoria dos 23 Parquets Network_dataset_1..23",
                "path": "parquet/audit.json"},
               {"id": "comparison", "label": "Comparação agregada das duas representações",
                "path": "comparison.json"},
               {"id": "tokens", "label": "Contagem exata dos tokens inválidos nas partições 1, 22 e 23",
                "path": "invalid_token_verification.json"}]
    blocks, tables, datasets, charts = [], [], {}, []
    database = sqlite3.connect(":memory:")
    database.row_factory = sqlite3.Row
    queries = []

    def sql_dataset(id, data, sort):
        # The portable reader requires SQL provenance for tables/charts. This
        # staging database contains ONLY completed audit aggregates, never flows.
        table_name = "report_" + id
        fields = list(data[0])
        declarations = ", ".join('"' + field + '" ' + ("TEXT" if isinstance(data[0][field], str) else "INTEGER") for field in fields)
        database.execute(f'CREATE TABLE "{table_name}" ({declarations})')
        database.executemany(f'INSERT INTO "{table_name}" VALUES ({",".join("?" for _ in fields)})',
                             [tuple(row[field] for field in fields) for row in data])
        query = f'SELECT * FROM "{table_name}" ORDER BY "{sort}" ASC;'
        datasets[id] = [dict(row) for row in database.execute(query)]
        queries.append(query)
        sources.append({"id": id + "_query", "label": "Agregados de auditoria: " + id,
                        "path": "report_queries.sql", "query": {"engine": "sqlite", "language": "sql",
                        "sql": query, "executed_at": now, "tables_used": [table_name],
                        "description": "Consulta de apresentação em report_tables.sqlite. A tabela foi preenchida por build_checkpoint_report.py a partir de parquet/audit.json e csv_complete/audit.json; a auditoria original foi executada em Python e permanece documentada nesses arquivos. Nenhum registro bruto é carregado nesta etapa."}})
        return id + "_query"

    def paragraph(id, body, source=None):
        blocks.append({"id": id, "type": "markdown", "body": body, **({"sourceId": source} if source else {})})

    def table(id, title, data, labels, sort):
        query_source = sql_dataset(id, data, sort)
        tables.append({"id": id, "title": title, "dataset": id, "sourceId": query_source,
                       "defaultSort": {"field": sort, "direction": "asc"},
                       "columns": [{"field": field, "label": label,
                                    **({"type": "text"} if isinstance(data[0][field], str) else {"format": "number"})}
                                   for field, label in labels]})
        blocks.append({"id": id + "_block", "type": "table", "tableId": id})

    paragraph("title", "# " + title)
    paragraph("summary", f"## {verdict} com ton_iot_network/1.0.0\n\n"
              f"Foram examinadas integralmente **23 partições e {rows:,} registros por representação**, "
              f"em CSV e Parquet. Os Parquets apresentam **{invalid:,} células inválidas** "
              f"({100 * invalid / rows:.6f}% do número de registros). "
              f"As contagens de erros por coluna entre as duas representações "
              f"{'coincidem' if counts_match else 'diferem'}. A normalização de produção interrompe a leitura diante dessas violações. "
              "O checkpoint não autoriza iniciar experimentos; primeiro é necessário resolver os valores rejeitados.", "comparison")
    paragraph("findings", "## As violações têm diagnóstico preservado\n\n"
              "A tabela identifica a coluna e a regra que rejeitou cada conjunto de valores. Exemplos são limitados; "
              "a contagem inclui todos os blocos. Uma célula pode violar mais de uma regra, portanto não se devem somar regras sobrepostas.")
    if issue_rows:
        table("issues", "Violações nos Parquets", issue_rows,
              [("column", "Coluna"), ("rule", "Regra"), ("count", "Quantidade"), ("examples", "Tokens exemplificados")], "column")
    paragraph("exact_tokens", "### O mesmo token já existe nos CSVs originais\n\n"
              "Uma verificação complementar da coluna src_bytes nas três partições afetadas contou "
              "exatamente **869 ocorrências de 0.0.0.0**: 175 na partição 1, 690 na 22 e 4 na 23. "
              "Não há outro token entre as falhas de conversão dessa coluna nesses arquivos. "
              "O problema antecede a conversão Parquet; a causa de origem não foi estabelecida.", "tokens")
    paragraph("concentration", "## As falhas estão concentradas em três partições\n\n"
              "A partição 22 reúne 690 das 869 ocorrências. O gráfico mostra somente as três partições afetadas; "
              "as outras 20 tiveram zero falhas. As barras são contagens, não taxas: a partição 23 tem 339.021 linhas, "
              "enquanto as partições 1 e 22 têm um milhão cada.", "comparison")
    chart_source = sql_dataset("affected_partitions", [{"partition": str(row["partition"]), "invalid": row["parquet_invalid"],
                                        "rows": row["rows"]} for row in file_rows if row["parquet_invalid"]], "partition")
    charts.append({"id": "invalid_by_partition", "title": "Valores inválidos por partição afetada",
                   "type": "bar", "dataset": "affected_partitions", "sourceId": chart_source,
                   "valueFormat": "number", "encodings": {
                       "x": {"field": "partition", "type": "nominal", "label": "Partição"},
                       "y": {"field": "invalid", "type": "quantitative", "label": "Valores inválidos", "format": "number"}}})
    blocks.append({"id": "concentration_chart", "type": "chart", "chartId": "invalid_by_partition"})
    paragraph("controls", "## Verificações de domínio e conversão\n\n"
              "Zeros indicam que nenhuma ocorrência foi encontrada na varredura completa. "
              "Ausências permitidas não são classificadas como erro. Integralidade é verificada em contagens, códigos, portas e label.")
    table("checks", "Verificações nas duas representações", checks,
          [("check", "Verificação"), ("csv", "CSV"), ("parquet", "Parquet")], "check")
    paragraph("partitions", "## Cobertura por partição\n\n"
              "As linhas de CSV e Parquet foram confrontadas por partição. A diferença de estrutura esperada é uid: "
              "existe no CSV da partição 6 e foi adicionada como nulo nas demais partições Parquet. "
              "As listas completas de colunas presentes, ausentes e adicionais estão nos JSONs por arquivo.")
    table("files", "Contagens por arquivo", file_rows,
          [("partition", "Partição"), ("rows", "Linhas"), ("csv_cols", "Colunas CSV"),
           ("parquet_cols", "Colunas Parquet"), ("csv_missing", "Ausentes CSV"),
           ("csv_invalid", "Erros CSV"), ("parquet_invalid", "Erros Parquet")], "partition")
    paragraph("missing", "## Ausência reconhecida é separada de erro\n\n"
              "Nulos são valores ausentes nativos; sentinelas são somente os tokens exatos vazio e hífen. "
              "As contagens abaixo são por célula. Colunas ausentes de um arquivo não entram como nulos no CSV; "
              "a ausência estrutural aparece na tabela de tipos e nos arquivos detalhados. "
              "O cruzamento completo arquivo × coluna está em by_file_column.csv de cada representação.")
    table("columns", "Nulos, sentinelas e erros por coluna", col_rows,
          [("column", "Coluna"), ("csv_nulls", "Nulos CSV"), ("csv_sentinels", "Sentinelas CSV"),
           ("parquet_nulls", "Nulos Parquet"), ("parquet_sentinels", "Sentinelas Parquet"),
           ("invalid_csv", "Erros CSV"), ("invalid_parquet", "Erros Parquet")], "column")
    paragraph("types", "## Tipos de armazenamento não definem semântica\n\n"
              "CSV armazena texto, sem dtype intrínseco; a auditoria preservou seus tokens como string. "
              "A tabela mostra os tipos Arrow declarados nos Parquets e os tipos recebidos pelo pandas. "
              "Uma coluna quantitativa armazenada como string pode ser compatível após conversão; "
              "tokens inválidos continuam sendo rejeitados. Não foi usada inferência física para escolher tratamento de modelo.")
    table("dtypes", "Tipos físicos observados", dtype_rows,
          [("column", "Coluna"), ("parquet_dtype", "Arrow Parquet"), ("pandas_dtype", "Pandas Parquet"),
           ("csv_absent_rows", "Linhas sem a coluna no CSV")], "column")
    paragraph("method", "## Método reproduzível e fontes preservadas\n\n"
              "Cada leitor processou no máximo 100.000 registros por bloco, reutilizando o normalizador de produção. "
              "A auditoria capturou SchemaValidationError e acumulou seu relatório, sem devolver dados inválidos ao pipeline. "
              "O detalhamento de negativos, integralidade e portas foi calculado à parte apenas para o relatório. "
              "Tamanho e mtime de todos os arquivos coincidiram antes/depois; a contagem lida dos Parquets coincidiu com seus metadados. "
              "Não houve escrita nas fontes, treinamento de IDS, clustering ou alteração das regras de produção. "
              "Os hashes dos códigos e os caminhos exatos de origem estão nos audit.json.")
    paragraph("limits", "## Limites do parecer\n\n"
              "Este é um censo de conformidade com o esquema, não uma prova da veracidade dos dados. "
              "Os exemplos não enumeram todos os tokens distintos. Contagens iguais entre CSV e Parquet não provam igualdade célula a célula. "
              "A auditoria não examinou a correção de splits, vazamento, ontologias abertas ou consistência entre alvos. "
              "Tamanho/mtime preservados não equivalem a hashes integrais de conteúdo.")
    paragraph("next", "## Decisões necessárias antes de continuar\n\n"
              "Revisar a política de tratamento dos tokens rejeitados nas colunas indicadas. "
              "Para src_bytes, manter a semântica quantitativa e a exigência de número finito, integral e não negativo. "
              "O token 0.0.0.0 não é número nem sentinela cadastrada: sua proveniência precisa ser confirmada antes de qualquer "
              "decisão explícita de reparo ou ausência. Não transformá-lo em zero, não ampliar o vocabulário de ausência por heurística "
              "e não voltar a tratar bytes como categoria. Não há evidência nesta auditoria para relaxar outras regras que passaram. "
              "A pergunta pendente é o significado desses registros na preparação original do TON_IoT. "
              "O Incremento 2 permanece fora deste checkpoint.")
    artifact = {"surface": "report", "manifest": {"version": 1, "surface": "report", "title": title,
                "generatedAt": now, "sources": sources, "blocks": blocks, "tables": tables, "charts": charts},
                "snapshot": {"version": 1, "generatedAt": now, "status": "ready", "datasets": datasets}}
    (ROOT / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "report_queries.sql").write_text("\n".join(queries) + "\n", encoding="utf-8")
    database.commit()
    with sqlite3.connect(ROOT / "report_tables.sqlite") as saved:
        database.backup(saved)
    database.close()
    # The comparison source includes all reviewed table rows used in the report.
    summary["report_tables"] = datasets
    (ROOT / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"report_tables", "file_rows"}}, indent=2))


if __name__ == "__main__":
    build()
