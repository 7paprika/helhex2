from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import copy
import datetime as dt
import html
import json
import math


INIT_STATE: dict[str, Any] = {
    "tag_no": "HE-101",
    "tube_fluid_name": "Process Slurry",
    "shell_fluid_name": "Hot Water / Steam",
    "fluid_type": "Liquid (뉴턴 유체 - 물, 오일 등)",
    "t_rho": 998.0,
    "t_cp": 4180.0,
    "t_k": 0.6,
    "t_mu": 1.0,
    "s_rho": 998.0,
    "s_mu": 1.0,
    "s_cp": 4180.0,
    "s_k": 0.6,
    "rheology_model": "Power-law (멱법칙)",
    "tau_y": 5.0,
    "plastic_visc": 0.05,
    "consistency_k": 0.1,
    "flow_index_n": 0.8,
    "m_hot": 5000.0,
    "m_cold": 8000.0,
    "T_hot_in": 30.0,
    "T_hot_out": 80.0,
    "T_cold_in": 120.0,
    "T_cold_out": 90.0,
    "allowable_dp_tube": 1.5,
    "allowable_dp_shell": 0.5,
    "N_p": 3,
    "d_o": 25.4,
    "t_thick": 2.11,
    "D_c": 400.0,
    "pitch": 50.0,
    "D_s": 500.0,
    "shell_thick": 10.0,
    "D_mandrel": 350.0,
    "tube_material": "Stainless Steel 316 (k=16)",
    "tube_k_wall": 16.0,
    "R_fi": 0.000176,
    "R_fo": 0.000176,
    "overdesign_pct": 10.0,
    "design_p_shell": 10.0,
    "allow_s_shell": 137.9,
    "joint_e": 0.85,
    "ca_shell": 3.0,
    "orientation": "Vertical (수직형)",
}


@dataclass
class CalculationResult:
    values: dict[str, Any]
    errors: list[str]
    warnings: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.errors


NUMERIC_POSITIVE_KEYS = (
    "t_rho",
    "t_cp",
    "t_k",
    "s_rho",
    "s_cp",
    "s_k",
    "m_hot",
    "m_cold",
    "allowable_dp_tube",
    "allowable_dp_shell",
    "d_o",
    "D_c",
    "pitch",
    "D_s",
    "D_mandrel",
    "tube_k_wall",
    "allow_s_shell",
)


def merged_inputs(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    data = copy.deepcopy(INIT_STATE)
    if overrides:
        data.update(overrides)
    return data


def apply_loaded_data(json_str: str) -> dict[str, Any]:
    parsed = json.loads(json_str)
    if not isinstance(parsed, dict):
        raise ValueError("JSON root must be an object.")
    return {k: parsed[k] for k in INIT_STATE if k in parsed}


def _safe_log_mean_temperature_difference(d1: float, d2: float) -> tuple[float, bool]:
    if d1 == d2 and d1 > 0:
        return d1, False
    if d1 > 0 and d2 > 0:
        return (d1 - d2) / math.log(d1 / d2), False
    return 1.0, True


def _add_basic_validation(data: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    for key in NUMERIC_POSITIVE_KEYS:
        if float(data[key]) <= 0:
            errors.append(f"{key} must be greater than zero.")

    if int(data["N_p"]) < 1:
        errors.append("N_p must be at least 1.")
    if float(data["design_p_shell"]) < 0:
        errors.append("design_p_shell cannot be negative.")
    if not (0 < float(data["joint_e"]) <= 1.0):
        errors.append("joint_e must be between 0 and 1.")
    if float(data["ca_shell"]) < 0:
        errors.append("ca_shell cannot be negative.")
    if float(data["R_fi"]) < 0 or float(data["R_fo"]) < 0:
        errors.append("Fouling factors cannot be negative.")
    if float(data["overdesign_pct"]) < 0:
        errors.append("overdesign_pct cannot be negative.")
    if float(data["t_thick"]) <= 0:
        errors.append("t_thick must be greater than zero.")
    if float(data["d_o"]) <= 2.0 * float(data["t_thick"]):
        errors.append("Tube wall thickness is too large for the selected OD; resulting ID must stay positive.")
    if float(data["pitch"]) < float(data["d_o"]):
        errors.append("Coil pitch must be at least the tube OD to avoid overlap.")
    if float(data["D_c"]) <= float(data["d_o"]):
        errors.append("Coil center diameter must be larger than tube OD.")
    if float(data["D_mandrel"]) > (float(data["D_c"]) - float(data["d_o"])):
        errors.append("Mandrel OD is larger than the available inner coil diameter.")
    if float(data["D_s"]) < (float(data["D_c"]) + float(data["d_o"])):
        errors.append("Shell ID is smaller than coil OD envelope.")

    tube_heating = float(data["T_hot_out"]) > float(data["T_hot_in"])
    if tube_heating and float(data["T_cold_out"]) >= float(data["T_cold_in"]):
        errors.append("Heating mode requires the shell-side hot stream to cool down across the exchanger.")
    if not tube_heating and float(data["T_cold_out"]) <= float(data["T_cold_in"]):
        errors.append("Cooling mode requires the shell-side cold stream to warm up across the exchanger.")

    if "Liquid" not in str(data["fluid_type"]):
        if "Power" in str(data["rheology_model"]):
            if float(data["consistency_k"]) <= 0:
                errors.append("consistency_k must be greater than zero for the power-law model.")
            if float(data["flow_index_n"]) <= 0:
                errors.append("flow_index_n must be greater than zero for the power-law model.")
            if float(data["flow_index_n"]) > 1.5:
                warnings.append("flow_index_n is unusually high for most shear-thinning slurries.")
        else:
            if float(data["plastic_visc"]) <= 0:
                errors.append("plastic_visc must be greater than zero for the Bingham model.")
            if float(data["tau_y"]) < 0:
                errors.append("tau_y cannot be negative for the Bingham model.")


def calculate_design(raw_inputs: dict[str, Any]) -> CalculationResult:
    data = merged_inputs(raw_inputs)
    errors: list[str] = []
    warnings: list[str] = []
    _add_basic_validation(data, errors, warnings)

    values: dict[str, Any] = {"generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    d_i = float(data["d_o"]) - 2.0 * float(data["t_thick"])
    values["d_i"] = d_i

    tube_heating = float(data["T_hot_out"]) > float(data["T_hot_in"])
    values["is_tube_heating"] = tube_heating
    values["op_mode"] = "🔥 Heater Mode" if tube_heating else "❄️ Cooler Mode"

    q_kw = (float(data["m_hot"]) / 3600.0) * (float(data["t_cp"]) / 1000.0) * abs(float(data["T_hot_in"]) - float(data["T_hot_out"]))
    values["Q_kW"] = q_kw
    s_cp_kj = float(data["s_cp"]) / 1000.0
    delta_t_shell = abs(float(data["T_cold_in"]) - float(data["T_cold_out"]))
    values["est_m_cold"] = (q_kw * 3600.0) / (s_cp_kj * max(0.1, delta_t_shell)) if delta_t_shell > 0 else 0.0

    if tube_heating:
        values["est_T_cold_out"] = float(data["T_cold_in"]) - ((q_kw * 3600.0) / (max(0.1, float(data["m_cold"])) * s_cp_kj))
        d_t1 = float(data["T_cold_in"]) - float(data["T_hot_out"])
        d_t2 = float(data["T_cold_out"]) - float(data["T_hot_in"])
    else:
        values["est_T_cold_out"] = float(data["T_cold_in"]) + ((q_kw * 3600.0) / (max(0.1, float(data["m_cold"])) * s_cp_kj))
        d_t1 = float(data["T_hot_in"]) - float(data["T_cold_out"])
        d_t2 = float(data["T_hot_out"]) - float(data["T_cold_in"])

    lmtd, lmtd_error = _safe_log_mean_temperature_difference(d_t1, d_t2)
    values["LMTD"] = lmtd
    values["lmtd_error"] = lmtd_error
    if lmtd_error:
        errors.append("Temperature cross detected; the specified inlet/outlet temperatures do not support feasible heat transfer.")

    if errors:
        return CalculationResult(values=values, errors=errors, warnings=warnings)

    t_mu_pa = float(data.get("t_mu", 1.0)) / 1000.0
    s_mu_pa = float(data.get("s_mu", 1.0)) / 1000.0
    curvature_ratio = d_i / float(data["D_c"])
    values["curvature_ratio"] = curvature_ratio

    n_p = max(1, int(data["N_p"]))
    m_hot_per_tube = (float(data["m_hot"]) / 3600.0) / n_p
    area_cross = math.pi * ((d_i / 1000.0) ** 2) / 4.0
    v_tube = m_hot_per_tube / (float(data["t_rho"]) * area_cross)
    values["v_tube"] = v_tube

    if "Liquid" in str(data["fluid_type"]):
        re = (float(data["t_rho"]) * v_tube * (d_i / 1000.0)) / max(1e-6, t_mu_pa)
        pr = (float(data["t_cp"]) * t_mu_pa) / max(1e-6, float(data["t_k"]))
    else:
        n_val = float(data["flow_index_n"]) if "Power" in str(data["rheology_model"]) else 1.0
        k_val = float(data["consistency_k"]) if "Power" in str(data["rheology_model"]) else float(data["plastic_visc"])
        d_m_tube = d_i / 1000.0
        term1 = float(data["t_rho"]) * (v_tube ** (2.0 - n_val)) * (d_m_tube ** n_val)
        term2 = (8.0 ** (n_val - 1.0)) * max(k_val, 1e-4) * (((3.0 * n_val + 1.0) / (4.0 * n_val)) ** n_val)
        re = term1 / term2 if term2 > 0 else 0.0
        mu_app = term1 / (re * v_tube) if (re * v_tube) > 0 else 0.001
        pr = (float(data["t_cp"]) * mu_app) / max(1e-6, float(data["t_k"]))
        values["mu_app"] = mu_app

    de = re * math.sqrt(max(0.0, curvature_ratio))
    re_crit = 2100.0 * (1.0 + 12.0 * math.sqrt(max(0.0, curvature_ratio)))
    f_c = (64.0 / max(re, 1.0) * (1.0 + 0.033 * (math.log10(max(de, 1.0))) ** 4.0)) if re < re_crit else (0.304 / (max(re, 1.0) ** 0.25) + 0.029 * math.sqrt(max(0.0, curvature_ratio)))
    nu_straight = 4.36 if re < re_crit else 0.023 * (max(re, 1.0) ** 0.8) * (pr ** 0.4)
    nu_calc = nu_straight * (1.0 + 3.5 * curvature_ratio)
    h_i = (nu_calc * float(data["t_k"])) / (d_i / 1000.0)

    values.update({
        "Re": re,
        "Pr": pr,
        "De": de,
        "Re_crit": re_crit,
        "f_c": f_c,
        "Nu_straight": nu_straight,
        "Nu_calc": nu_calc,
        "h_i": h_i,
    })

    m_cold_kg_s = float(data["m_cold"]) / 3600.0
    d_s_m = float(data["D_s"]) / 1000.0
    d_man_m = float(data["D_mandrel"]) / 1000.0
    d_o_m = float(data["d_o"]) / 1000.0
    pitch_m = float(data["pitch"]) / 1000.0
    d_c_m = float(data["D_c"]) / 1000.0
    lead_m = pitch_m * n_p
    length_per_turn = math.sqrt((math.pi * d_c_m) ** 2 + lead_m ** 2)

    a_annulus = (math.pi / 4.0) * (d_s_m ** 2 - d_man_m ** 2)
    a_tube_cross = (math.pi / 4.0) * (d_o_m ** 2)
    a_blocked = n_p * a_tube_cross * (length_per_turn / max(1e-6, lead_m))
    a_free_flow = max(a_annulus * 0.1, a_annulus - a_blocked)
    v_shell = m_cold_kg_s / (float(data["s_rho"]) * a_free_flow)
    d_e_shell = d_s_m - d_man_m
    re_shell = (float(data["s_rho"]) * v_shell * d_e_shell) / max(1e-6, s_mu_pa)
    pr_shell = (float(data["s_cp"]) * s_mu_pa) / max(1e-6, float(data["s_k"]))
    nu_shell = 0.33 * (max(re_shell, 1.0) ** 0.6) * (pr_shell ** 0.33)
    h_o = (nu_shell * float(data["s_k"])) / max(1e-6, d_o_m)

    pitch_ratio = float(data["pitch"]) / max(1e-6, float(data["d_o"]))
    penalty_factor = max(0.5, 1.0 - 2.0 * (1.25 - pitch_ratio)) if pitch_ratio < 1.25 else 1.0
    h_o *= penalty_factor

    r_wall = (d_o_m * math.log(float(data["d_o"]) / d_i)) / (2.0 * max(1e-6, float(data["tube_k_wall"])))
    u_calc = 1.0 / ((1.0 / max(h_o, 0.1)) + float(data["R_fo"]) + r_wall + float(data["R_fi"]) * (float(data["d_o"]) / d_i) + (float(data["d_o"]) / d_i) * (1.0 / max(h_i, 0.1)))
    area_req = (q_kw * 1000.0) / (u_calc * lmtd)
    area_design = area_req * (1.0 + float(data["overdesign_pct"]) / 100.0)
    total_tube_length = area_design / (math.pi * d_o_m)
    length_per_tube = total_tube_length / n_p
    turns_per_tube = length_per_tube / length_per_turn
    dp_tube_bar = (f_c * (length_per_tube / (d_i / 1000.0)) * (float(data["t_rho"]) * (v_tube ** 2) / 2.0)) / 100000.0
    l_shell_m = turns_per_tube * lead_m
    l_shell_mm = l_shell_m * 1000.0
    f_s = 0.316 / (max(re_shell, 1.0) ** 0.25)
    dp_shell_bar = (f_s * (l_shell_m / max(1e-6, d_e_shell)) * (float(data["s_rho"]) * (v_shell ** 2) / 2.0)) / 100000.0

    inner_clearance_rad = ((float(data["D_c"]) - float(data["d_o"])) - float(data["D_mandrel"])) / 2.0
    outer_clearance_rad = (float(data["D_s"]) - (float(data["D_c"]) + float(data["d_o"]))) / 2.0

    p_mpa = float(data["design_p_shell"]) / 10.0
    r_mm = float(data["D_s"]) / 2.0
    denom = float(data["allow_s_shell"]) * float(data["joint_e"]) - 0.6 * p_mpa
    if denom <= 0:
        errors.append("Mechanical thickness denominator is non-positive; check shell pressure, allowable stress, and weld efficiency.")
        return CalculationResult(values=values, errors=errors, warnings=warnings)
    t_req = (p_mpa * r_mm) / denom + float(data["ca_shell"])
    t_final = max(6.0, math.ceil(t_req))
    shell_od = float(data["D_s"]) + 2.0 * t_final
    shell_tt_length_m = l_shell_m + (2.0 * d_s_m)
    shell_tt_length_mm = shell_tt_length_m * 1000.0
    footprint_area = (math.pi / 4.0) * ((shell_od / 1000.0) ** 2) if "Vertical" in str(data["orientation"]) else (shell_od / 1000.0) * shell_tt_length_m

    if penalty_factor < 1.0:
        warnings.append(f"Shell-side heat transfer coefficient was reduced by {(1.0 - penalty_factor) * 100:.0f}% because pitch is below 1.25×OD.")
    if dp_tube_bar > float(data["allowable_dp_tube"]):
        warnings.append("Tube-side pressure drop exceeds the allowable limit.")
    if dp_shell_bar > float(data["allowable_dp_shell"]):
        warnings.append("Shell-side pressure drop exceeds the allowable limit.")
    if v_tube < 1.0:
        warnings.append("Tube velocity is below the recommended liquid-service range and may increase fouling risk.")
    if v_tube > 3.0:
        warnings.append("Tube velocity exceeds the recommended liquid-service range and may increase erosion risk.")
    if v_shell < 0.2:
        warnings.append("Shell velocity is below the recommended range and may cause thermal dead zones.")
    if v_shell > 1.5:
        warnings.append("Shell velocity exceeds the recommended range and may increase vibration risk.")
    if shell_tt_length_m > 10.0:
        warnings.append("Estimated shell T/T length exceeds 10 m and may be difficult to fabricate or lay out.")

    values.update(
        {
            "N_p_val": n_p,
            "A_c": area_cross,
            "m_hot_per_tube": m_hot_per_tube,
            "m_cold_kg_s": m_cold_kg_s,
            "D_s_m": d_s_m,
            "D_man_m": d_man_m,
            "d_o_m": d_o_m,
            "p_m": pitch_m,
            "D_c_m": d_c_m,
            "Lead_m": lead_m,
            "Length_per_Turn": length_per_turn,
            "A_annulus": a_annulus,
            "A_blocked": a_blocked,
            "A_free_flow": a_free_flow,
            "v_shell": v_shell,
            "D_e_shell": d_e_shell,
            "Re_shell": re_shell,
            "Pr_shell": pr_shell,
            "Nu_shell": nu_shell,
            "h_o": h_o,
            "pitch_ratio": pitch_ratio,
            "penalty_factor": penalty_factor,
            "R_wall": r_wall,
            "U_calc": u_calc,
            "Area_req": area_req,
            "Area_design": area_design,
            "Total_Tube_Length": total_tube_length,
            "Length_per_Tube": length_per_tube,
            "Turns_per_Tube": turns_per_tube,
            "dp_tube_bar": dp_tube_bar,
            "L_shell_m": l_shell_m,
            "L_shell_mm": l_shell_mm,
            "f_s": f_s,
            "dp_shell_bar": dp_shell_bar,
            "inner_clearance_rad": inner_clearance_rad,
            "outer_clearance_rad": outer_clearance_rad,
            "t_req": t_req,
            "t_final": t_final,
            "shell_od": shell_od,
            "Shell_TT_Length_m": shell_tt_length_m,
            "Shell_TT_Length_mm": shell_tt_length_mm,
            "Footprint_Area": footprint_area,
        }
    )
    return CalculationResult(values=values, errors=errors, warnings=warnings)


def find_optimal_geometry(data: dict[str, Any], values: dict[str, Any]) -> dict[str, float | None]:
    if values.get("lmtd_error") or values.get("d_i", 0) <= 0:
        return {"opt_best_Dc": None, "opt_min_LTT": None, "opt_best_Dm": None, "opt_best_Ds": None}

    opt_best_dc = None
    opt_min_ltt = float("inf")
    opt_best_dm = None
    opt_best_ds = None
    n_p = int(values["N_p_val"])
    opt_p_m = (float(data["d_o"]) * 1.25) / 1000.0
    opt_lead_m = opt_p_m * n_p
    d_i = float(values["d_i"])
    d_o_m = float(values["d_o_m"])
    pr_shell = float(values["Pr_shell"])
    r_wall = float(values["R_wall"])
    lmtd = float(values["LMTD"])
    q_kw = float(values["Q_kW"])
    nu_straight = float(values["Nu_straight"])
    m_cold_kg_s = float(values["m_cold_kg_s"])
    s_mu_pa = float(data["s_mu"]) / 1000.0

    for t_dc in range(int(float(data["d_o"]) * 10.0), 3000, 10):
        t_dc_m = t_dc / 1000.0
        t_dm = max(10.0, t_dc - float(data["d_o"]) - 10.0)
        t_dm_m = t_dm / 1000.0
        t_ds = t_dc + float(data["d_o"]) + 40.0
        t_ds_m = t_ds / 1000.0
        t_a_annulus = (math.pi / 4.0) * (t_ds_m ** 2 - t_dm_m ** 2)
        t_length_per_turn = math.sqrt((math.pi * t_dc_m) ** 2 + opt_lead_m ** 2)
        t_a_blocked = n_p * ((math.pi / 4.0) * (d_o_m ** 2)) * (t_length_per_turn / max(1e-6, opt_lead_m))
        t_a_free = max(t_a_annulus * 0.1, t_a_annulus - t_a_blocked)
        t_v_shell = m_cold_kg_s / (float(data["s_rho"]) * t_a_free)
        t_re_shell = (float(data["s_rho"]) * t_v_shell * (t_ds_m - t_dm_m)) / max(1e-6, s_mu_pa)
        t_ho = ((0.33 * (max(t_re_shell, 1.0) ** 0.6) * (pr_shell ** 0.33)) * float(data["s_k"])) / max(1e-6, d_o_m)
        t_cr = d_i / t_dc
        t_hi = ((nu_straight * (1.0 + 3.5 * t_cr)) * float(data["t_k"])) / (d_i / 1000.0)
        t_u = 1.0 / ((1.0 / max(t_ho, 0.1)) + float(data["R_fo"]) + r_wall + float(data["R_fi"]) * (float(data["d_o"]) / d_i) + (float(data["d_o"]) / d_i) * (1.0 / max(t_hi, 0.1)))
        t_area_req = (q_kw * 1000.0) / (t_u * lmtd)
        t_area_design = t_area_req * (1.0 + float(data["overdesign_pct"]) / 100.0)
        t_turns = (t_area_design / (math.pi * d_o_m * n_p)) / t_length_per_turn
        t_l_shell_m = t_turns * opt_lead_m
        t_l_tt = t_l_shell_m + (2.0 * t_ds_m)
        if t_dc_m < t_l_tt and t_l_tt < opt_min_ltt:
            opt_min_ltt = t_l_tt
            opt_best_dc = float(t_dc)
            opt_best_dm = float(t_dm)
            opt_best_ds = float(t_ds)

    return {
        "opt_best_Dc": opt_best_dc,
        "opt_min_LTT": None if opt_best_dc is None else opt_min_ltt,
        "opt_best_Dm": opt_best_dm,
        "opt_best_Ds": opt_best_ds,
    }


def format_datasheet_markdown(data: dict[str, Any], values: dict[str, Any]) -> str:
    return f"""
| **Item Tag No.** | **{data['tag_no']}** | **Type** | Helical Coil Heat Exchanger |
| :--- | :--- | :--- | :--- |
| **Performance Data** | | | |
| Heat Duty (kW) | {values['Q_kW']:,.2f} | Overall U-value (W/m²K) | {values['U_calc']:,.1f} |
| Req. Area / Design Area | {values['Area_req']:,.2f} m² / **{values['Area_design']:,.2f} m²** (+{data['overdesign_pct']}%) | LMTD (°C) | {values['LMTD']:,.1f} |
| **Process Conditions** | **Tube Side (Inner)** | **Shell Side (Outer)** | |
| Fluid Name | **{data['tube_fluid_name']}** | **{data['shell_fluid_name']}** | |
| Total Flow Rate (kg/h) | {data['m_hot']:,.0f} | {data['m_cold']:,.0f} | |
| Temp. In / Out (°C) | {data['T_hot_in']} / {data['T_hot_out']} | {data['T_cold_in']} / {data['T_cold_out']} | |
| Velocity (m/s) | **{values['v_tube']:.2f}** | **{values['v_shell']:.2f}** | |
| Calc. Press. Drop (bar) | **{values['dp_tube_bar']:.3f}** (Allow: {data['allowable_dp_tube']}) | **{values['dp_shell_bar']:.3f}** (Allow: {data['allowable_dp_shell']}) | |
| **Mechanical Design** | | | |
| **[Tube]** OD x Thick. (mm) | {data['d_o']} x {data['t_thick']} | **[Tube]** Material | {data['tube_material']} |
| **[Tube]** Parallel Coils (N_p) | **{data['N_p']} ea** | **[Tube]** Length per Tube | {values['Length_per_Tube']:,.1f} m |
| **[Coil]** Center Dia. (D_c) | {data['D_c']} mm | **[Coil]** Pitch (Gap) | {data['pitch']} mm |
| **[Coil]** Turns per Tube | {values['Turns_per_Tube']:,.1f} turns | **[Install]** Orientation | {data['orientation']} |
| **[Shell]** ID / Mandrel OD | {data['D_s']} mm / {data['D_mandrel']} mm | **[Shell]** OD x Thick. (mm) | **{values['shell_od']:.1f} x {values['t_final']:.0f}** |
| **[Shell]** T/T Length (mm) | **{values['Shell_TT_Length_mm']:,.0f} mm** | | |
"""


def format_html_report(data: dict[str, Any], values: dict[str, Any], warnings: list[str]) -> str:
    tag = html.escape(str(data["tag_no"]))
    tube_name = html.escape(str(data["tube_fluid_name"]))
    shell_name = html.escape(str(data["shell_fluid_name"]))
    tube_mat = html.escape(str(data["tube_material"]))
    warning_html = "".join(f"<li>{html.escape(item)}</li>" for item in warnings) or "<li>No active design warnings.</li>"
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset=\"UTF-8\">
    <title>{tag} - Heat Exchanger Datasheet</title>
    <style>
        body {{ font-family: Arial, sans-serif; color: #333; line-height: 1.5; margin: 20px; }}
        .header {{ text-align: center; border-bottom: 3px solid #004488; padding-bottom: 10px; margin-bottom: 24px; }}
        .meta-info {{ font-size: 12px; color: #666; text-align: right; margin-bottom: 12px; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 12px; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
        th {{ background-color: #f4f7f6; font-weight: bold; }}
        .section-title {{ background-color: #004488; color: white; padding: 6px 12px; font-size: 14px; font-weight: bold; }}
        .notice {{ background: #fff3cd; border: 1px solid #ffeeba; padding: 10px; margin-bottom: 16px; }}
    </style>
</head>
<body>
    <div class=\"notice\">Use browser Print → Save as PDF for clean export.</div>
    <div class=\"header\">
        <h2>COMMERCIAL DATASHEET</h2>
        <p>Helical Coil Heat Exchanger</p>
    </div>
    <div class=\"meta-info\">Generated on: {html.escape(str(values['generated_at']))}</div>
    <table>
        <tr><td class=\"section-title\" colspan=\"4\">1. General Information</td></tr>
        <tr><th>Item Tag No.</th><td><b>{tag}</b></td><th>Overall U-value</th><td>{values['U_calc']:,.1f} W/m²K</td></tr>
        <tr><th>Heat Duty</th><td>{values['Q_kW']:,.2f} kW</td><th>Req. / Design Area</th><td>{values['Area_req']:,.2f} / <b>{values['Area_design']:,.2f} m²</b> (+{data['overdesign_pct']}%)</td></tr>
        <tr><th>LMTD</th><td>{values['LMTD']:,.1f} °C</td><th>Operation Mode</th><td>{html.escape(str(values['op_mode']))}</td></tr>
        <tr><td class=\"section-title\" colspan=\"4\">2. Process Conditions</td></tr>
        <tr><th>Parameter</th><th>Tube Side (Inner)</th><th colspan=\"2\">Shell Side (Outer)</th></tr>
        <tr><td>Fluid Name</td><td>{tube_name}</td><td colspan=\"2\">{shell_name}</td></tr>
        <tr><td>Flow Rate (kg/h)</td><td>{data['m_hot']:,.0f}</td><td colspan=\"2\">{data['m_cold']:,.0f}</td></tr>
        <tr><td>Temp. In / Out (°C)</td><td>{data['T_hot_in']} / {data['T_hot_out']}</td><td colspan=\"2\">{data['T_cold_in']} / {data['T_cold_out']}</td></tr>
        <tr><td>Velocity (m/s)</td><td>{values['v_tube']:.2f}</td><td colspan=\"2\">{values['v_shell']:.2f}</td></tr>
        <tr><td>Pressure Drop (bar)</td><td>{values['dp_tube_bar']:.3f}</td><td colspan=\"2\">{values['dp_shell_bar']:.3f}</td></tr>
        <tr><td class=\"section-title\" colspan=\"4\">3. Mechanical Design</td></tr>
        <tr><th>[Tube] OD x Thick. (mm)</th><td>{data['d_o']} x {data['t_thick']}</td><th>[Tube] Material</th><td>{tube_mat}</td></tr>
        <tr><th>[Tube] Parallel Coils</th><td>{data['N_p']} ea</td><th>[Tube] Length per Tube</th><td>{values['Length_per_Tube']:,.1f} m</td></tr>
        <tr><th>[Shell] ID / Mandrel OD</th><td>{data['D_s']} mm / {data['D_mandrel']} mm</td><th>[Shell] OD x Thick. (mm)</th><td>{values['shell_od']:.1f} x {values['t_final']:.0f}</td></tr>
        <tr><th>[Shell] T/T Length</th><td colspan=\"3\"><b>{values['Shell_TT_Length_mm']:,.0f} mm</b></td></tr>
    </table>
    <h3>Active warnings</h3>
    <ul>{warning_html}</ul>
</body>
</html>"""
