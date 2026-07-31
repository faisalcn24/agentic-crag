from __future__ import annotations

import pytest

from functions.spreadsheet import (
    execute_spreadsheet_plan,
    fallback_spreadsheet_plan,
    is_spreadsheet_analysis_question,
    spreadsheet_columns,
    validate_spreadsheet_plan,
)


SOURCE = {
    "filename": "benchmarks.xlsx-Data",
    "type": "xlsx",
    "text": (
        "Name: alpha | Team: A | Score: 10 | Cost: 2\n"
        "Name: beta | Team: A | Score: 25 | Cost: 5\n"
        "Name: gamma | Team: B | Score: 15 | Cost: \n"
        "Name: broken | Team: B | Score: N/A | Cost: 3"
    ),
}


def plan(operation: str, **changes):
    value = {
        "operation": operation,
        "value_column": "",
        "select_columns": [],
        "filters": [],
        "sort_column": "",
        "sort_direction": "asc",
        "group_by": "",
        "aggregate": "none",
        "limit": 100,
    }
    value.update(changes)
    return value


def test_select_and_filter_rows():
    result = execute_spreadsheet_plan(
        [SOURCE],
        plan(
            "filter",
            select_columns=["Name", "Score"],
            filters=[
                {"column": "Team", "operator": "eq", "value": "A", "values": []}
            ],
        ),
    )

    assert result["rows"] == [
        {"Name": "alpha", "Score": "10"},
        {"Name": "beta", "Score": "25"},
    ]


def test_sort_rows_numerically_when_values_are_text():
    result = execute_spreadsheet_plan(
        [SOURCE],
        plan(
            "sort",
            select_columns=["Name", "Score"],
            sort_column="Score",
            sort_direction="desc",
        ),
    )

    assert [row["Name"] for row in result["rows"][:3]] == ["beta", "gamma", "alpha"]


def test_count_rows():
    result = execute_spreadsheet_plan(
        [SOURCE],
        plan(
            "count",
            filters=[
                {"column": "Team", "operator": "eq", "value": "B", "values": []}
            ],
        ),
    )

    assert result["value"] == 2


@pytest.mark.parametrize(
    ("operation", "expected"), [("sum", 50.0), ("average", 50.0 / 3)]
)
def test_sum_and_average_ignore_blanks_and_mixed_text(operation, expected):
    result = execute_spreadsheet_plan(
        [SOURCE], plan(operation, value_column="Score")
    )

    assert result["value"] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("operation", "name", "score"),
    [("minimum", "alpha", "10"), ("maximum", "beta", "25")],
)
def test_minimum_and_maximum_return_the_matching_row(operation, name, score):
    result = execute_spreadsheet_plan(
        [SOURCE], plan(operation, value_column="Score")
    )

    assert result["rows"][0]["Name"] == name
    assert result["rows"][0]["Score"] == score


def test_difference_returns_both_rows_and_exact_value():
    result = execute_spreadsheet_plan(
        [SOURCE],
        plan(
            "difference",
            value_column="Score",
            select_columns=["Name", "Score"],
            filters=[
                {
                    "column": "Name",
                    "operator": "in",
                    "value": "",
                    "values": ["alpha", "beta"],
                }
            ],
        ),
    )

    assert result["value"] == 15.0
    assert [row["Name"] for row in result["rows"]] == ["alpha", "beta"]


def test_comparison_preserves_requested_rows():
    result = execute_spreadsheet_plan(
        [SOURCE],
        plan(
            "comparison",
            value_column="Cost",
            select_columns=["Name", "Cost"],
            filters=[
                {
                    "column": "Name",
                    "operator": "in",
                    "value": "",
                    "values": ["alpha", "beta"],
                }
            ],
        ),
    )

    assert result["rows"] == [
        {"Name": "alpha", "Cost": "2"},
        {"Name": "beta", "Cost": "5"},
    ]


def test_simple_grouping_uses_requested_aggregate():
    result = execute_spreadsheet_plan(
        [SOURCE],
        plan(
            "group",
            value_column="Score",
            group_by="Team",
            aggregate="average",
        ),
    )

    assert result["rows"] == [
        {"Team": "A", "average_Score": 17.5},
        {"Team": "B", "average_Score": 15.0},
    ]


def test_validation_rejects_ambiguous_or_missing_column_names():
    sources = [
        {
            "filename": "ambiguous.xlsx-Sheet",
            "type": "xlsx",
            "text": "Total Cost: 5 | Total_Cost: 6",
        }
    ]

    assert validate_spreadsheet_plan(plan("sum", value_column="total-cost"), sources) is None
    assert validate_spreadsheet_plan(plan("sum", value_column="missing"), sources) is None


def test_fallback_plan_handles_highest_filter_and_difference_questions():
    source = {
        "filename": "benchmarks.xlsx-Indexing Benchmarks",
        "type": "xlsx",
        "text": (
            "Document_Set: tiny-smoke | Retrieval_Time_ms: 180 | Status: Open\n"
            "Document_Set: large-policy-pack | Retrieval_Time_ms: 410 | Status: Closed"
        ),
    }

    highest = fallback_spreadsheet_plan("Which benchmark had the slowest retrieval time?", [source])
    filtered = fallback_spreadsheet_plan("Which benchmarks have status Open?", [source])
    difference = fallback_spreadsheet_plan(
        "How much faster was tiny-smoke retrieval than large-policy-pack retrieval?",
        [source],
    )

    assert spreadsheet_columns([source]) == ["Document_Set", "Retrieval_Time_ms", "Status"]
    assert highest["operation"] == "maximum"
    assert highest["value_column"] == "Retrieval_Time_ms"
    assert filtered["operation"] == "filter"
    assert filtered["select_columns"] == ["Document_Set"]
    assert filtered["filters"][0]["value"] == "Open"
    assert difference["operation"] == "difference"
    assert difference["filters"][0]["values"] == ["tiny-smoke", "large-policy-pack"]


def test_fallback_chooses_label_from_the_sheet_with_the_value_column():
    sources = [
        {
            "filename": "risks.xlsx-Risks",
            "type": "xlsx",
            "text": "Risk_ID: R-1 | Probability: 0.5 | Status: Open",
        },
        {
            "filename": "benchmarks.xlsx-Benchmarks",
            "type": "xlsx",
            "text": (
                "Document_Set: tiny | Retrieval_Time_ms: 10\n"
                "Document_Set: large | Retrieval_Time_ms: 30"
            ),
        },
    ]

    plan_value = fallback_spreadsheet_plan(
        "Which benchmark had the slowest retrieval time?", sources
    )
    result = execute_spreadsheet_plan(sources, plan_value)

    assert plan_value["select_columns"] == ["Document_Set", "Retrieval_Time_ms"]
    assert result["source"] == "benchmarks.xlsx-Benchmarks"
    assert result["rows"][0]["Document_Set"] == "large"


def test_status_filter_returns_row_identifiers_not_the_status_value():
    source = {
        "filename": "risks.xlsx-Risk Register",
        "type": "xlsx",
        "text": (
            "Risk_ID: R-005 | Risk: Retrieval miss | Status: Open\n"
            "Risk_ID: R-008 | Risk: No authentication | Status: Open"
        ),
    }

    plan_value = fallback_spreadsheet_plan("Which risks have status Open?", [source])
    result = execute_spreadsheet_plan([source], plan_value)

    assert plan_value["select_columns"] == ["Risk_ID"]
    assert result["rows"] == [{"Risk_ID": "R-005"}, {"Risk_ID": "R-008"}]


def test_open_risks_phrase_routes_to_status_filter():
    source = {
        "filename": "risks.xlsx-Risk Register",
        "type": "xlsx",
        "text": (
            "Risk_ID: R-005 | Status: Open\n"
            "Risk_ID: R-008 | Status: Open\n"
            "Risk_ID: R-006 | Status: Monitoring"
        ),
    }

    plan_value = fallback_spreadsheet_plan("List the open risks.", [source])
    result = execute_spreadsheet_plan([source], plan_value)

    assert plan_value["operation"] == "filter"
    assert result["rows"] == [{"Risk_ID": "R-005"}, {"Risk_ID": "R-008"}]


def test_unrelated_how_many_question_does_not_route_to_spreadsheet_count():
    source = {
        "filename": "budget.xlsx-Budget",
        "type": "xlsx",
        "text": "Quarter: Q1 | Category: Hosting | Budget_USD: 18",
    }

    assert not is_spreadsheet_analysis_question(
        "How many active users used the demo last week?", [source]
    )
