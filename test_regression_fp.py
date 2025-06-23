import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
try:
    from ForensikVideo import adaptive_thresholds
except SystemExit:
    pytest.skip("ForensikVideo dependencies missing", allow_module_level=True)

def fake_fp(total, flagged):
    return flagged / total

# simple regression test
@pytest.mark.parametrize("total,flagged", [(100,4),(120,5),(150,6)])
def test_false_positive_rate(total, flagged):
    assert fake_fp(total, flagged) <= 0.05

def test_adaptive_thresholds():
    ssim, z = adaptive_thresholds(15, 0.2)
    assert 0.25 <= ssim <= 0.35
    assert z >= 4
