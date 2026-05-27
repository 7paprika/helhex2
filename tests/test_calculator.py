from calculator import apply_loaded_data, calculate_design, find_optimal_geometry, format_datasheet_markdown, merged_inputs


def test_default_design_calculates_successfully():
    data = merged_inputs()
    result = calculate_design(data)
    assert result.is_valid, result.errors
    assert result.values["Q_kW"] > 0
    assert result.values["U_calc"] > 0
    assert result.values["Area_design"] >= result.values["Area_req"] > 0
    opt = find_optimal_geometry(data, result.values)
    assert opt["opt_best_Dc"] is not None


def test_negative_flow_is_rejected():
    result = calculate_design(merged_inputs({"m_hot": -1.0}))
    assert not result.is_valid
    assert any("m_hot" in message for message in result.errors)


def test_nonphysical_wall_thickness_is_rejected():
    result = calculate_design(merged_inputs({"d_o": 25.4, "t_thick": 13.0}))
    assert not result.is_valid
    assert any("resulting ID must stay positive" in message for message in result.errors)


def test_inconsistent_heating_scenario_is_rejected():
    result = calculate_design(
        merged_inputs({
            "T_hot_in": 30.0,
            "T_hot_out": 80.0,
            "T_cold_in": 120.0,
            "T_cold_out": 130.0,
        })
    )
    assert not result.is_valid
    assert any("Heating mode requires" in message for message in result.errors)


def test_temperature_cross_is_flagged():
    result = calculate_design(
        merged_inputs({
            "T_hot_in": 80.0,
            "T_hot_out": 40.0,
            "T_cold_in": 60.0,
            "T_cold_out": 90.0,
        })
    )
    assert not result.is_valid
    assert any("Temperature cross" in message for message in result.errors)


def test_apply_loaded_data_whitelists_known_keys_only():
    parsed = apply_loaded_data('{"tag_no": "HE-999", "junk": 1}')
    assert parsed == {"tag_no": "HE-999"}


def test_datasheet_uses_selected_tube_material():
    data = merged_inputs({"tube_material": "Titanium (k=22)", "tube_k_wall": 22.0})
    result = calculate_design(data)
    markdown = format_datasheet_markdown(data, result.values)
    assert "Titanium (k=22)" in markdown
