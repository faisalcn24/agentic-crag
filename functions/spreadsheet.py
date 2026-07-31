from __future__ import annotations

import re
from typing import Any

import duckdb


OPERATIONS = {
    "select",
    "filter",
    "sort",
    "count",
    "sum",
    "average",
    "minimum",
    "maximum",
    "difference",
    "comparison",
    "group",
}
AGGREGATES = {"none", "count", "sum", "average", "minimum", "maximum"}
FILTER_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "contains", "in"}

SPREADSHEET_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"type": "string", "enum": sorted(OPERATIONS)},
        "value_column": {"type": "string"},
        "select_columns": {"type": "array", "items": {"type": "string"}},
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "operator": {"type": "string", "enum": sorted(FILTER_OPERATORS)},
                    "value": {"type": "string"},
                    "values": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["column", "operator", "value", "values"],
                "additionalProperties": False,
            },
        },
        "sort_column": {"type": "string"},
        "sort_direction": {"type": "string", "enum": ["asc", "desc"]},
        "group_by": {"type": "string"},
        "aggregate": {"type": "string", "enum": sorted(AGGREGATES)},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
    },
    "required": [
        "operation",
        "value_column",
        "select_columns",
        "filters",
        "sort_column",
        "sort_direction",
        "group_by",
        "aggregate",
        "limit",
    ],
    "additionalProperties": False,
}


def is_spreadsheet_analysis_question(
    question: str, sources: list[dict[str, Any]]
) -> bool:
    if not any(source.get("type") == "xlsx" for source in sources):
        return False
    lowered = question.casefold()
    if re.search(r"\b(?:count|how many)\b", lowered):
        question_tokens = set(re.findall(r"[a-z0-9]+", lowered))
        column_tokens = {
            token
            for column in spreadsheet_columns(sources)
            for token in re.findall(r"[a-z0-9]+", _column_label(column))
        }
        return bool(question_tokens & column_tokens)
    patterns = (
        r"\b(?:highest|lowest|slowest|fastest|maximum|minimum|max|min)\b",
        r"\b(?:total|sum|average|mean|difference|faster|slower|compare|comparison)\b",
        r"\b(?:sort|sorted|order|group)\b",
        r"\bwhich\b.+\bstatus\b",
        r"\bopen risks?\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def spreadsheet_columns(sources: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for source in sources:
        for row in _source_rows(source):
            for column in row:
                if column not in columns:
                    columns.append(column)
    return columns


def answer_spreadsheet_lookup(
    question: str, sources: list[dict[str, Any]]
) -> str | None:
    """Answer an unambiguous lookup from one spreadsheet row."""
    fields = _requested_lookup_fields(question)
    if not fields:
        return None
    question_tokens = _lookup_tokens(question)
    candidates: list[tuple[int, str, dict[str, str]]] = []
    for source in sources:
        for row in _source_rows(source):
            if not all(field in row for field in fields):
                continue
            searchable_values = " ".join(
                value for key, value in row.items() if key != "Notes"
            )
            candidates.append(
                (
                    len(question_tokens & _lookup_tokens(searchable_values)),
                    source.get("filename", "unknown"),
                    row,
                )
            )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if candidates[0][0] == 0 or (
        len(candidates) > 1 and candidates[0][0] == candidates[1][0]
    ):
        return None
    _, filename, row = candidates[0]
    if "exceed" in question.casefold() and {
        "Budget_USD",
        "Actual_USD",
    } <= row.keys():
        budget = _number(row["Budget_USD"])
        actual = _number(row["Actual_USD"])
        if budget is not None and actual is not None and actual <= budget:
            return (
                f"It did not exceed the budget; actual spend was "
                f"{_format_value('Actual_USD', row['Actual_USD'])} against a "
                f"{_format_value('Budget_USD', row['Budget_USD'])} budget. "
                f"[{filename}]"
            )
    values = [_format_value(field, row[field]) for field in fields]
    if len(values) == 1:
        return f"{values[0]} [{filename}]"
    labels = {"Budget_USD": "Budget", "Actual_USD": "Actual spend"}
    parts = [
        f"{labels.get(field, field.replace('_', ' '))}: {value}"
        for field, value in zip(fields, values, strict=True)
    ]
    return f"{'; '.join(parts)} [{filename}]"


def spreadsheet_plan_prompt(
    question: str, sources: list[dict[str, Any]]
) -> str:
    columns = spreadsheet_columns(sources)
    samples = []
    for source in sources:
        if source.get("type") != "xlsx":
            continue
        rows = _source_rows(source)[:3]
        samples.append(f"{source.get('filename', 'unknown')}: {rows}")
    return (
        "Create a spreadsheet query plan using only the listed columns. Do not write SQL. "
        "Use empty strings or arrays for fields that do not apply. For minimum or maximum, "
        "put the measured column in value_column. For a difference, filter the two named rows. "
        "Return the schema-constrained object only.\n"
        f"Question: {question}\nColumns: {columns}\nSamples:\n" + "\n".join(samples)
    )


def validate_spreadsheet_plan(
    plan: Any, sources: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if not isinstance(plan, dict) or plan.get("operation") not in OPERATIONS:
        return None
    columns = spreadsheet_columns(sources)
    if not columns:
        return None

    normalized = {
        "operation": plan.get("operation"),
        "value_column": "",
        "select_columns": [],
        "filters": [],
        "sort_column": "",
        "sort_direction": plan.get("sort_direction", "asc"),
        "group_by": "",
        "aggregate": plan.get("aggregate", "none"),
        "limit": plan.get("limit", 100),
    }
    if normalized["sort_direction"] not in {"asc", "desc"}:
        return None
    if normalized["aggregate"] not in AGGREGATES:
        return None
    if not isinstance(normalized["limit"], int) or not 1 <= normalized["limit"] <= 100:
        return None

    for key in ("value_column", "sort_column", "group_by"):
        raw_column = plan.get(key, "")
        if not isinstance(raw_column, str):
            return None
        if raw_column:
            resolved = _resolve_column(raw_column, columns)
            if resolved is None:
                return None
            normalized[key] = resolved

    select_columns = plan.get("select_columns", [])
    if not isinstance(select_columns, list):
        return None
    for raw_column in select_columns:
        if not isinstance(raw_column, str):
            return None
        resolved = _resolve_column(raw_column, columns)
        if resolved is None:
            return None
        if resolved not in normalized["select_columns"]:
            normalized["select_columns"].append(resolved)

    filters = plan.get("filters", [])
    if not isinstance(filters, list):
        return None
    for item in filters:
        if not isinstance(item, dict):
            return None
        column = _resolve_column(str(item.get("column", "")), columns)
        operator = item.get("operator")
        value = item.get("value", "")
        values = item.get("values", [])
        if (
            column is None
            or operator not in FILTER_OPERATORS
            or not isinstance(value, str)
            or not isinstance(values, list)
            or any(not isinstance(entry, str) for entry in values)
            or (operator == "in" and not values)
        ):
            return None
        normalized["filters"].append(
            {
                "column": column,
                "operator": operator,
                "value": value,
                "values": values,
            }
        )

    operation = normalized["operation"]
    if operation in {"sum", "average", "minimum", "maximum", "difference", "comparison"} and not normalized["value_column"]:
        return None
    if operation == "sort" and not normalized["sort_column"]:
        return None
    if operation == "group" and (
        not normalized["group_by"] or normalized["aggregate"] == "none"
    ):
        return None
    if operation == "group" and normalized["aggregate"] != "count" and not normalized["value_column"]:
        return None
    return normalized


def execute_spreadsheet_plan(
    sources: list[dict[str, Any]], plan: dict[str, Any]
) -> dict[str, Any] | None:
    normalized = validate_spreadsheet_plan(plan, sources)
    if normalized is None:
        return None
    table = _select_table(sources, normalized)
    if table is None:
        return None
    source, rows = table
    columns = list(rows[0])
    connection = duckdb.connect(":memory:")
    try:
        definitions = ", ".join(f"{_identifier(column)} VARCHAR" for column in columns)
        connection.execute(f"CREATE TABLE data ({definitions})")
        placeholders = ", ".join("?" for _ in columns)
        connection.executemany(
            f"INSERT INTO data VALUES ({placeholders})",
            [[row.get(column, "") for column in columns] for row in rows],
        )
        result = _execute(connection, normalized, columns)
    finally:
        connection.close()
    if result is None:
        return None
    result["source"] = source.get("filename", "unknown")
    evidence_rows = result.get("rows") or rows
    result["evidence"] = {
        "filename": source.get("filename", "unknown"),
        "type": "xlsx",
        "score": source.get("score"),
        "text": "\n".join(_row_text(row) for row in evidence_rows),
    }
    result["plan"] = normalized
    return result


def fallback_spreadsheet_plan(
    question: str, sources: list[dict[str, Any]]
) -> dict[str, Any] | None:
    columns = spreadsheet_columns(sources)
    if not columns:
        return None
    lowered = question.casefold()
    value_column = _question_value_column(lowered, columns)
    label_column, mentioned_values = _mentioned_row_values(lowered, sources)
    if value_column and not mentioned_values:
        label_column = _label_for_column(value_column, sources)
    result: dict[str, Any] = {
        "operation": "select",
        "value_column": value_column or "",
        "select_columns": [],
        "filters": [],
        "sort_column": "",
        "sort_direction": "asc",
        "group_by": "",
        "aggregate": "none",
        "limit": 100,
    }

    if re.search(r"\b(?:difference|faster|slower)\b", lowered):
        if not value_column or not label_column or len(mentioned_values) < 2:
            return None
        result.update(
            operation="difference",
            select_columns=[label_column, value_column],
            filters=[
                {
                    "column": label_column,
                    "operator": "in",
                    "value": "",
                    "values": mentioned_values[:2],
                }
            ],
        )
    elif re.search(r"\b(?:compare|comparison|versus|vs\.?|contrast)\b", lowered):
        if not value_column or not label_column or len(mentioned_values) < 2:
            return None
        result.update(
            operation="comparison",
            select_columns=[label_column, value_column],
            filters=[
                {
                    "column": label_column,
                    "operator": "in",
                    "value": "",
                    "values": mentioned_values[:2],
                }
            ],
        )
    elif re.search(r"\b(?:highest|slowest|maximum|max)\b", lowered):
        if not value_column:
            return None
        result.update(
            operation="maximum",
            select_columns=[column for column in (label_column, value_column) if column],
        )
    elif re.search(r"\b(?:lowest|fastest|minimum|min)\b", lowered):
        if not value_column:
            return None
        result.update(
            operation="minimum",
            select_columns=[column for column in (label_column, value_column) if column],
        )
    elif re.search(r"\b(?:average|mean)\b", lowered):
        result["operation"] = "average"
    elif re.search(r"\b(?:total|sum)\b", lowered):
        result["operation"] = "sum"
    elif re.search(r"\b(?:count|how many)\b", lowered):
        result["operation"] = "count"
    elif re.search(r"\bwhich\b.+\bstatus\b|\bopen risks?\b", lowered):
        status_column = _resolve_column("status", columns)
        if status_column is None:
            return None
        status_value = (
            "Open"
            if "open risk" in lowered
            else _mentioned_value_for_column(lowered, status_column, sources)
        )
        if status_value is None:
            return None
        label_column = _label_for_column(status_column, sources)
        result.update(
            operation="filter",
            select_columns=[label_column] if label_column else [],
            filters=[
                {
                    "column": status_column,
                    "operator": "eq",
                    "value": status_value,
                    "values": [],
                }
            ],
        )
    else:
        return None
    return validate_spreadsheet_plan(result, sources)


def format_spreadsheet_answer(question: str, result: dict[str, Any]) -> str:
    plan = result["plan"]
    operation = plan["operation"]
    source = result["source"]
    rows = result.get("rows", [])
    value_column = plan.get("value_column", "")
    if operation in {"minimum", "maximum"} and rows:
        row = rows[0]
        label_column = _label_column(list(row), exclude={value_column})
        label = row.get(label_column, "The matching row") if label_column else "The matching row"
        if label_column == "Risk_ID" and row.get("Risk"):
            label = f"{label} ({row['Risk']})"
        adjective = "highest" if operation == "maximum" else "lowest"
        return (
            f"{label} has the {adjective} {_column_label(value_column)}: "
            f"{_format_value(value_column, row[value_column])}. [{source}]"
        )
    if operation == "difference" and len(rows) >= 2:
        label_column = _label_column(list(rows[0]), exclude={value_column})
        first_label = rows[0].get(label_column, "First row")
        second_label = rows[1].get(label_column, "Second row")
        return (
            f"The difference is {_format_value(value_column, result['value'])}: "
            f"{first_label} is {_format_value(value_column, rows[0][value_column])} and "
            f"{second_label} is {_format_value(value_column, rows[1][value_column])}. [{source}]"
        )
    if operation in {"sum", "average", "count"}:
        label = "row count" if operation == "count" else f"{operation} {_column_label(value_column)}"
        column = value_column if operation != "count" else ""
        return f"The {label} is {_format_value(column, result['value'])}. [{source}]"
    if operation in {"filter", "select", "sort", "comparison", "group"} and rows:
        if all(len(row) == 1 for row in rows):
            values = [str(next(iter(row.values()))) for row in rows]
            return f"{_join_values(values)}. [{source}]"
        rendered = "; ".join(
            ", ".join(f"{_column_label(key)}: {_format_value(key, value)}" for key, value in row.items())
            for row in rows
        )
        return f"{rendered}. [{source}]"
    return "The answer is not present in the provided documents."


def _execute(connection, plan: dict[str, Any], columns: list[str]) -> dict[str, Any] | None:
    where_sql, params = _where_clause(plan["filters"])
    operation = plan["operation"]
    limit = plan["limit"]
    if operation == "count":
        value = connection.execute(f"SELECT COUNT(*) FROM data{where_sql}", params).fetchone()[0]
        return {"columns": ["count"], "rows": [], "value": value}
    if operation in {"sum", "average"}:
        function = "SUM" if operation == "sum" else "AVG"
        expression = _numeric_expression(plan["value_column"])
        value = connection.execute(
            f"SELECT {function}({expression}) FROM data{where_sql}", params
        ).fetchone()[0]
        return {"columns": [plan["value_column"]], "rows": [], "value": value}
    if operation in {"minimum", "maximum"}:
        value_column = plan["value_column"]
        expression = _numeric_expression(value_column)
        direction = "ASC" if operation == "minimum" else "DESC"
        extra = " AND " if where_sql else " WHERE "
        sql = (
            f"SELECT * FROM data{where_sql}{extra}{expression} IS NOT NULL "
            f"ORDER BY {expression} {direction} LIMIT 1"
        )
        return _row_result(connection, sql, params)
    if operation == "group":
        group_by = plan["group_by"]
        aggregate = plan["aggregate"]
        if aggregate == "count":
            expression = "COUNT(*)"
        else:
            functions = {
                "sum": "SUM",
                "average": "AVG",
                "minimum": "MIN",
                "maximum": "MAX",
            }
            expression = f"{functions[aggregate]}({_numeric_expression(plan['value_column'])})"
        alias = f"{aggregate}_{plan['value_column']}".rstrip("_")
        sql = (
            f"SELECT {_identifier(group_by)}, {expression} AS {_identifier(alias)} "
            f"FROM data{where_sql} GROUP BY {_identifier(group_by)} "
            f"ORDER BY {_identifier(group_by)} LIMIT {limit}"
        )
        return _row_result(connection, sql, params)

    selected = plan["select_columns"] or columns
    if operation in {"difference", "comparison"} and plan["value_column"] not in selected:
        selected.append(plan["value_column"])
    select_sql = ", ".join(_identifier(column) for column in selected)
    order_sql = ""
    if operation == "sort":
        direction = plan["sort_direction"].upper()
        expression = _numeric_expression(plan["sort_column"])
        order_sql = f" ORDER BY {expression} {direction} NULLS LAST"
    sql = f"SELECT {select_sql} FROM data{where_sql}{order_sql} LIMIT {limit}"
    result = _row_result(connection, sql, params)
    if result is not None and operation == "difference":
        values = [
            number
            for row in result["rows"]
            if (number := _number(row.get(plan["value_column"]))) is not None
        ]
        if len(values) != 2:
            return None
        result["value"] = abs(values[0] - values[1])
    return result


def _where_clause(filters: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for item in filters:
        column = _identifier(item["column"])
        operator = item["operator"]
        if operator == "in":
            placeholders = ", ".join("?" for _ in item["values"])
            clauses.append(f"LOWER(TRIM(COALESCE({column}, ''))) IN ({placeholders})")
            params.extend(value.casefold() for value in item["values"])
        elif operator == "contains":
            clauses.append(f"LOWER(COALESCE({column}, '')) LIKE ?")
            params.append(f"%{item['value'].casefold()}%")
        elif operator in {"gt", "gte", "lt", "lte"}:
            symbols = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
            clauses.append(f"{_numeric_expression(item['column'])} {symbols[operator]} ?")
            numeric = _number(item["value"])
            if numeric is None:
                return " WHERE FALSE", []
            params.append(numeric)
        else:
            symbol = "=" if operator == "eq" else "<>"
            clauses.append(f"LOWER(TRIM(COALESCE({column}, ''))) {symbol} ?")
            params.append(item["value"].casefold())
    return (" WHERE " + " AND ".join(clauses), params) if clauses else ("", params)


def _row_result(connection, sql: str, params: list[Any]) -> dict[str, Any]:
    cursor = connection.execute(sql, params)
    columns = [description[0] for description in cursor.description]
    rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    return {"columns": columns, "rows": rows}


def _select_table(
    sources: list[dict[str, Any]], plan: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]] | None:
    referenced = {
        column
        for column in [
            plan["value_column"],
            plan["sort_column"],
            plan["group_by"],
            *plan["select_columns"],
            *(item["column"] for item in plan["filters"]),
        ]
        if column
    }
    for source in sources:
        rows = _source_rows(source)
        if rows and referenced <= set(rows[0]):
            return source, rows
    return None


def _source_rows(source: dict[str, Any]) -> list[dict[str, str]]:
    if source.get("type") != "xlsx":
        return []
    rows = []
    seen = set()
    for line in source.get("text", "").splitlines():
        row = parse_spreadsheet_row(line)
        marker = tuple(row.items())
        if row and marker not in seen:
            rows.append(row)
            seen.add(marker)
    return rows


def parse_spreadsheet_row(line: str) -> dict[str, str]:
    if " | " not in line:
        return {}
    cells = [cell.split(": ", 1) for cell in line.split(" | ")]
    if any(len(cell) != 2 for cell in cells):
        return {}
    return {key.strip(): value.strip() for key, value in cells}


def _requested_lookup_fields(question: str) -> list[str]:
    lowered = question.casefold()
    fields = []
    if "budget" in lowered:
        fields.append("Budget_USD")
    if any(term in lowered for term in ("actual", "spent", "spend", "spending")):
        fields.append("Actual_USD")
    if "retrieval latency" in lowered or "retrieval time" in lowered:
        fields.append("Retrieval_Time_ms")
    if "index time" in lowered or (
        "indexing benchmark" in lowered and "how long" in lowered
    ):
        fields.append("Index_Time_Seconds")
    if "probability" in lowered:
        fields.append("Probability")
    if "what caused" in lowered or "root cause" in lowered:
        fields.append("Root_Cause")
    if "what date" in lowered or "on what date" in lowered:
        fields.append("Date")
    if "what fix" in lowered or "fix resolved" in lowered:
        fields.append("Fix")
    return fields


def _lookup_tokens(text: str) -> set[str]:
    stop_words = {
        "a",
        "and",
        "at",
        "did",
        "for",
        "how",
        "is",
        "on",
        "the",
        "to",
        "was",
        "what",
        "which",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if token not in stop_words
    }


def _resolve_column(requested: str, columns: list[str]) -> str | None:
    exact = [column for column in columns if column.casefold() == requested.casefold()]
    if len(exact) == 1:
        return exact[0]
    key = re.sub(r"[^a-z0-9]", "", requested.casefold())
    normalized = [
        column
        for column in columns
        if re.sub(r"[^a-z0-9]", "", column.casefold()) == key
    ]
    return normalized[0] if len(normalized) == 1 else None


def _question_value_column(question: str, columns: list[str]) -> str | None:
    aliases = (
        (("retrieval time", "retrieval latency", "retrieval"), "Retrieval_Time_ms"),
        (("index time", "indexing time"), "Index_Time_Seconds"),
        (("probability",), "Probability"),
        (("actual spend", "actually spent", "actual"), "Actual_USD"),
        (("budget",), "Budget_USD"),
        (("score",), "Score"),
        (("cost",), "Cost"),
    )
    for phrases, candidate in aliases:
        if any(phrase in question for phrase in phrases):
            resolved = _resolve_column(candidate, columns)
            if resolved:
                return resolved
    mentioned = [column for column in columns if _column_label(column) in question]
    return mentioned[0] if len(mentioned) == 1 else None


def _mentioned_row_values(
    question: str, sources: list[dict[str, Any]]
) -> tuple[str | None, list[str]]:
    columns = spreadsheet_columns(sources)
    preferred = [
        "Risk_ID",
        "Incident_ID",
        "Document_Set",
        "Name",
        "Quarter",
        "Category",
    ]
    ordered = [column for column in preferred if column in columns] + [
        column for column in columns if column not in preferred
    ]
    best_column = None
    best_values: list[str] = []
    for column in ordered:
        values = []
        for source in sources:
            for row in _source_rows(source):
                value = row.get(column, "")
                if value and not _is_numeric(value) and value.casefold() in question:
                    if value not in values:
                        values.append(value)
        if len(values) > len(best_values):
            best_column, best_values = column, values
    if best_column is None:
        best_column = _label_column(columns)
    return best_column, best_values


def _mentioned_value_for_column(
    question: str, column: str, sources: list[dict[str, Any]]
) -> str | None:
    for source in sources:
        for row in _source_rows(source):
            value = row.get(column, "")
            if value and value.casefold() in question:
                return value
    return None


def _label_for_column(
    value_column: str, sources: list[dict[str, Any]]
) -> str | None:
    for source in sources:
        rows = _source_rows(source)
        if rows and value_column in rows[0]:
            return _label_column(list(rows[0]), exclude={value_column})
    return None


def _label_column(columns: list[str], exclude: set[str] | None = None) -> str | None:
    exclude = exclude or set()
    preferred = ("Risk_ID", "Incident_ID", "Document_Set", "Name", "Quarter", "Category")
    for column in preferred:
        if column in columns and column not in exclude:
            return column
    return next((column for column in columns if column not in exclude), None)


def _numeric_expression(column: str) -> str:
    identifier = _identifier(column)
    return f"TRY_CAST(REPLACE(REPLACE(NULLIF(TRIM({identifier}), ''), '$', ''), ',', '') AS DOUBLE)"


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace("$", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def _is_numeric(value: Any) -> bool:
    return _number(value) is not None


def _column_label(column: str) -> str:
    return column.replace("_USD", "").replace("_ms", "").replace("_Seconds", "").replace("_", " ").casefold()


def _format_value(column: str, value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if column.endswith("_USD"):
        return f"${value}"
    if column.endswith("_ms"):
        return f"{value} ms"
    if column.endswith("_Seconds"):
        return f"{value} seconds"
    return str(value)


def _join_values(values: list[str]) -> str:
    if len(values) < 2:
        return values[0] if values else ""
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _row_text(row: dict[str, Any]) -> str:
    return " | ".join(f"{key}: {value}" for key, value in row.items())
