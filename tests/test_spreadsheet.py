from __future__ import annotations

import pytest

from functions.spreadsheet import (
    execute_spreadsheet_plan,
    is_spreadsheet_analysis_question,
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
            filters=[{"column": "Team", "operator": "eq", "value": "A", "values": []}],
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
            filters=[{"column": "Team", "operator": "eq", "value": "B", "values": []}],
        ),
    )

    assert result["value"] == 2


@pytest.mark.parametrize(
    ("operation", "expected"), [("sum", 50.0), ("average", 50.0 / 3)]
)
def test_sum_and_average_ignore_blanks_and_mixed_text(operation, expected):
    result = execute_spreadsheet_plan([SOURCE], plan(operation, value_column="Score"))

    assert result["value"] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("operation", "name", "score"),
    [("minimum", "alpha", "10"), ("maximum", "beta", "25")],
)
def test_minimum_and_maximum_return_the_matching_row(operation, name, score):
    result = execute_spreadsheet_plan([SOURCE], plan(operation, value_column="Score"))

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

    assert (
        validate_spreadsheet_plan(plan("sum", value_column="total-cost"), sources)
        is None
    )
    assert (
        validate_spreadsheet_plan(plan("sum", value_column="missing"), sources) is None
    )


def test_unrelated_how_many_question_does_not_route_to_spreadsheet_count():
    source = {
        "filename": "budget.xlsx-Budget",
        "type": "xlsx",
        "text": "Quarter: Q1 | Category: Hosting | Budget_USD: 18",
    }

    assert not is_spreadsheet_analysis_question(
        "How many active users used the demo last week?", [source]
    )
