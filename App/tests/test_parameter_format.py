from App.ui.parameter_format import format_parameter_map


def test_parameter_display_uses_engineering_symbols_and_units():
    value = format_parameter_map(
        {
            "E_prime_gpa": 40.0,
            "C_L_m_sqrt_s": 0.1,
            "mu_pa_s": 0.12,
            "sigma_min_mpa": 88.0,
        }
    )

    assert "E_prime_gpa" not in value
    assert "<i>E</i><sup>&prime;</sup>" in value
    assert "<i>C</i><sub>L</sub>" in value
    assert "GPa" in value
    assert "MPa" in value
