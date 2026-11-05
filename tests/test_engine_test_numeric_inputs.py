from __future__ import annotations

from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path


class NumericInputParser(HTMLParser):
    """Collect numeric inputs grouped by their enclosing form."""

    def __init__(self) -> None:
        super().__init__()
        self._form_id: str | None = None
        self.inputs: dict[str, dict[str, dict[str, str]]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "form":
            self._form_id = attributes.get("id")
        elif tag == "input" and self._form_id and attributes.get("type") == "number":
            name = attributes.get("name")
            if name:
                self.inputs.setdefault(self._form_id, {})[name] = {
                    key: value if value is not None else ""
                    for key, value in attributes.items()
                }

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self._form_id = None


def numeric_inputs() -> dict[str, dict[str, dict[str, str]]]:
    """Read the solver and recovery numeric input contracts from the HTML."""
    html_path = Path(__file__).parents[1] / "web" / "engine_test.html"
    parser = NumericInputParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    return parser.inputs


def is_on_step_grid(attributes: dict[str, str], value: str) -> bool:
    """Return whether a decimal value satisfies an HTML number input grid."""
    minimum = Decimal(attributes.get("min", "0"))
    step = Decimal(attributes.get("step", "1"))
    candidate = Decimal(value)
    maximum = attributes.get("max")
    return (
        candidate >= minimum
        and (maximum is None or candidate <= Decimal(maximum))
        and (candidate - minimum) % step == 0
    )


def test_solver_and_recovery_numeric_inputs_accept_defaults_and_recovery_edges() -> None:
    inputs = numeric_inputs()

    recovery = inputs["recovery-form"]
    assert recovery["length_m"] == {
        "name": "length_m",
        "type": "number",
        "value": "0.8",
        "min": "0.001",
        "step": "0.001",
        "required": "",
    }
    assert recovery["duration_s"] == {
        "name": "duration_s",
        "type": "number",
        "value": "15",
        "min": "0.01",
        "max": "300",
        "step": "0.01",
        "required": "",
    }
    assert recovery["gravity_m_s2"]["min"] == "0.001"
    assert recovery["gravity_m_s2"]["step"] == "0.00001"

    for value in ("0.80", "0.791", "0.801"):
        assert is_on_step_grid(recovery["length_m"], value)
    for value in ("15", "15.01", "14.91"):
        assert is_on_step_grid(recovery["duration_s"], value)
    assert is_on_step_grid(recovery["gravity_m_s2"], "9.81")

    solver = inputs["solver-form"]
    assert solver["damping_s_inv"]["max"] == "10"
    assert solver["gravity_m_s2"]["max"] == "100"
    for fields in (solver, recovery):
        for attributes in fields.values():
            assert is_on_step_grid(attributes, attributes["value"])


def test_decimal_precision_is_available_across_engine_test_tabs() -> None:
    inputs = numeric_inputs()

    solver = inputs["solver-form"]
    recovery = inputs["recovery-form"]
    prediction = inputs["prediction-form"]
    selection = inputs["selection-form"]

    for fields in (solver, recovery, prediction):
        damping = fields["damping_s_inv"]
        assert damping["min"] == "0"
        assert damping["max"] == "10"
        assert damping["step"] == "0.001"
        assert is_on_step_grid(damping, "0.082")

    assert recovery["noise_sigma_deg"]["step"] == "0.001"
    assert is_on_step_grid(recovery["noise_sigma_deg"], "0.125")
    assert selection["noise_sigma_deg"]["step"] == "0.001"
    assert is_on_step_grid(selection["noise_sigma_deg"], "0.125")

    assert is_on_step_grid(solver["length_m"], "1.234")
    assert is_on_step_grid(recovery["duration_s"], "14.37")
    assert is_on_step_grid(recovery["gravity_m_s2"], "9.80665")
    assert is_on_step_grid(recovery["initial_velocity_rad_s"], "0.01")
