from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_without_exceptions():
    app_path = Path(__file__).resolve().parents[1] / 'app.py'
    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not app.exception
    assert any(button.label == 'Run Backtest' for button in app.button)
    assert any(box.label == 'Run MA parameter optimizer' for box in app.checkbox)

    optimizer_toggle = next(box for box in app.checkbox if box.label == 'Run MA parameter optimizer')
    optimizer_toggle.set_value(True)
    app.run(timeout=30)
    assert not app.exception
    assert any(select.label == 'Optimizer validation' for select in app.selectbox)
