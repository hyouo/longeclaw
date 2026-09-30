import pytest
from longevityclaw.codex_suite.io import SuiteError, read_json, table, measurements

@pytest.mark.parametrize("text", ['1e999', '-1e999', '{"x":[1e400]}',
                                  '{"x":NaN}', '{"x":1,"x":2}'])
def test_json_rejects_nonfinite_and_ambiguous_input(text):
    with pytest.raises(SuiteError):
        read_json(text)


def test_json_finite_exponents_and_integers_are_preserved():
    assert read_json('{"x":1e300,"n":1234567890123456789}') == {
        "x": 1e300, "n": 1234567890123456789}

@pytest.mark.parametrize("text", ['feature_id,value\nG1,"1',
                                  '"feature_id,value\nG1,1',
                                  'feature_id,value\nG1,"1"garbage'])
def test_malformed_quoting_has_actionable_error(tmp_path, text):
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(SuiteError, match="Malformed table"):
        table(path)


def test_quoted_fields_bom_and_crlf_are_supported(tmp_path):
    path = tmp_path / "valid.csv"
    path.write_bytes(b'\xef\xbb\xbffeature_id,value\r\n"GENE,1","2"\r\n')
    assert measurements(path, "long") == {"sample_1": {"GENE,1": 2.0}}
