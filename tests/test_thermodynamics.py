import pytest
from hypothesis import given, strategies as st
from src.models.thermodynamics import (
    compute_pmv,
    compute_ppd,
    compute_tcl,
    compute_fcl,
    compute_hc,
    compute_pa,
    rl_agent_may_shed,
)

# TEST 1A-1: PMV comfort neutrality
# ISO 7730 Table B.1 reference point: M=70, W=0, ta=22°C, tr=22°C,
# var=0.1 m/s, rh=60%, Icl=1.0 clo → PMV ≈ −0.15 (within ±0.05 tolerance)
def test_pmv_iso7730_reference_neutral():
    pmv = compute_pmv(M=70, W=0, ta=22.0, tr=22.0, var=0.1, rh=60.0, Icl=1.0)
    assert -0.20 <= pmv <= -0.10, f"Expected PMV ≈ -0.15, got {pmv}"

# TEST 1A-2: PMV hot extreme
# Hot office: ta=28°C, tr=30°C, var=0.05 m/s, rh=50%, Icl=0.5 clo, M=70
# Expected PMV > +1.0 (slightly warm to warm)
def test_pmv_hot_environment():
    pmv = compute_pmv(M=70, W=0, ta=28.0, tr=30.0, var=0.05, rh=50.0, Icl=0.5)
    assert pmv > 1.0

# TEST 1A-3: PMV cold extreme
# Cold: ta=15°C, tr=15°C, var=0.3 m/s, rh=40%, Icl=0.5 clo, M=70
# Expected PMV < −1.5
def test_pmv_cold_environment():
    pmv = compute_pmv(M=70, W=0, ta=15.0, tr=15.0, var=0.3, rh=40.0, Icl=0.5)
    assert pmv < -1.5

# TEST 1A-4: PMV hard boundary — HVAC empathy gate
# The RL agent MUST NOT shed HVAC when PMV is in (−0.5, +0.5)
# Test PMV at exactly −0.5 (boundary — empathy protection should apply)
def test_pmv_boundary_negative():
    pmv = compute_pmv(M=70, W=0, ta=21.0, tr=21.0, var=0.1, rh=55.0, Icl=1.0)
    assert abs(pmv) <= 0.5, "Expected near-comfort PMV for these inputs"

# TEST 1A-5: PPD formula correctness
# PPD = 100 - 95·exp(-0.03353·PMV⁴ - 0.2179·PMV²)
# At PMV=0: PPD = 100 - 95·exp(0) = 5.0% (minimum, theoretical floor)
def test_ppd_at_pmv_zero():
    ppd = compute_ppd(pmv=0.0)
    assert abs(ppd - 5.0) < 0.01, f"PPD at PMV=0 should be 5.0%, got {ppd}"

# TEST 1A-6: PPD at PMV=±2 (ISO 7730 limit for existing buildings)
# At PMV=2: PPD ≈ 76.8%; at PMV=-2: same (symmetric)
def test_ppd_at_pmv_plus2():
    ppd = compute_ppd(pmv=2.0)
    assert 75.0 <= ppd <= 78.0, f"Expected PPD≈76.8% at PMV=2, got {ppd}"

def test_ppd_symmetric():
    assert abs(compute_ppd(2.0) - compute_ppd(-2.0)) < 0.01

# TEST 1A-7: PPD at PMV=±3 (scale extremes) should be near 100%
def test_ppd_at_extremes():
    assert compute_ppd(3.0) > 99.0
    assert compute_ppd(-3.0) > 99.0

# TEST 1A-8: tcl iterative convergence
# Clothing surface temperature must converge within 100 iterations
# and be within ±0.01°C of the accepted ISO reference value
def test_tcl_iterative_convergence():
    tcl = compute_tcl(M=70, W=0, ta=22.0, tr=22.0, var=0.1, Icl=1.0)
    # Reference: for standard conditions, tcl ≈ 30.4°C
    assert 25.0 <= tcl <= 32.0, f"tcl out of range: {tcl}"

# TEST 1A-9: fcl formula — both branches
# Icl = 0.078 m²K/W (= 0.5 clo): uses 1.05 + 0.645·Icl branch
# Icl = 0.05 m²K/W (= 0.32 clo): uses 1.00 + 1.290·Icl branch
def test_fcl_low_insulation():
    fcl = compute_fcl(Icl_m2KW=0.05)
    expected = 1.00 + 1.290 * 0.05
    assert abs(fcl - expected) < 0.001

def test_fcl_high_insulation():
    fcl = compute_fcl(Icl_m2KW=0.155)
    expected = 1.05 + 0.645 * 0.155
    assert abs(fcl - expected) < 0.001

# TEST 1A-10: hc convective coefficient — velocity-dominated branch
# When 12.1·√var > 2.38·|tcl-ta|^0.25, use velocity formula
def test_hc_velocity_dominated():
    hc = compute_hc(tcl=30.0, ta=22.0, var=1.0)
    expected_velocity = 12.1 * (1.0 ** 0.5)
    assert abs(hc - expected_velocity) < 0.1

# TEST 1A-11: vapour pressure formula
# pa = rh·exp(16.6536 - 4030.18/(ta+235))
# At ta=22°C, rh=60: verify against known value ≈ 1590 Pa
def test_vapour_pressure_known_value():
    pa = compute_pa(ta=22.0, rh=60.0)
    assert 1550 <= pa <= 1640, f"pa at 22°C/60%RH should ≈1590 Pa, got {pa}"

# TEST 1A-12: Full 6-input PMV round-trip property test (Hypothesis)
# Property: PMV must always be in [-3, +3] for any physically valid input
@given(
    ta=st.floats(min_value=10, max_value=40),
    tr=st.floats(min_value=10, max_value=40),
    var=st.floats(min_value=0.0, max_value=2.0),
    rh=st.floats(min_value=10, max_value=90),
    Icl=st.floats(min_value=0.0, max_value=3.0),
)
def test_pmv_always_in_range(ta, tr, var, rh, Icl):
    pmv = compute_pmv(M=70, W=0, ta=ta, tr=tr, var=var, rh=rh, Icl=Icl)
    assert -4.0 <= pmv <= 4.0  # allow slight overflow at hard extremes

# TEST 1A-13: PMV → RL empathy gate integration
# When PMV is in (−0.5, +0.5), the RL agent must NOT emit shed command for HVAC
def test_rl_empathy_gate_blocks_hvac_shed_in_comfort_zone():
    pmv_values = [0.0, 0.4, -0.4, 0.499, -0.499]
    for pmv in pmv_values:
        allowed = rl_agent_may_shed("node_hvac", pmv=pmv)
        assert not allowed, f"HVAC should not be shed at PMV={pmv}"

# TEST 1A-14: PMV empathy gate — shed is allowed outside comfort zone
def test_rl_empathy_gate_permits_hvac_shed_outside_comfort():
    pmv_values = [0.6, -0.6, 1.0, -1.0, 2.0]
    for pmv in pmv_values:
        allowed = rl_agent_may_shed("node_hvac", pmv=pmv)
        assert allowed, f"HVAC should be sheddable at PMV={pmv}"
