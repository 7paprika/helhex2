import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import datetime

st.set_page_config(page_title="Helical Tube Heat Exchanger Designer", layout="wide")
st.title("플랜트 공정 설계: Helical Tube Heat Exchanger 최적화")
st.markdown("---")

# =========================================================
# [Constants] 명명 상수 (Named Constants) — 매직 넘버 제거
# =========================================================
# 마찰계수 상관식 계수 (Ito correlation for helical coils)
FRICTION_LAMINAR_LOG_COEFF = 0.033       # 층류 헬리컬 보정 계수
FRICTION_TURBULENT_A = 0.304             # 난류 마찰 계수 A
FRICTION_TURBULENT_B = 0.029             # 난류 마찰 계수 B (곡률 의존)
FRICTION_TURBULENT_EXP = 0.25            # 난류 Re 지수

# Nusselt 상관식 계수 (Dittus-Boelter / Hausen)
NU_LAMINAR_CONST = 4.36                  # 완전발달 층류 Nu (일정 열유속)
NU_TURBULENT_COEFF = 0.023               # Dittus-Boelter 계수
NU_TURBULENT_RE_EXP = 0.8               # Dittus-Boelter Re 지수
NU_TURBULENT_PR_EXP = 0.4               # Dittus-Boelter Pr 지수 (가열 시)
CURVATURE_NU_FACTOR = 3.5               # 코일 곡률 Nu 증강 계수

# Shell 측 Nu 상관식 (external crossflow over helical coil bank)
SHELL_NU_LAMINAR_COEFF = 0.60            # 층류 계수 (Zukauskas)
SHELL_NU_LAMINAR_RE_EXP = 0.50           # 층류 Re 지수
SHELL_NU_TURBULENT_COEFF = 0.33          # 난류 계수
SHELL_NU_TURBULENT_RE_EXP = 0.60         # 난류 Re 지수
SHELL_NU_PR_EXP = 0.33                   # Pr 지수 (공통)
SHELL_RE_TRANSITION = 1000.0             # Shell 측 층류/난류 전이 Re

# 임계 Re 관련
RE_CRIT_BASE = 2100                      # 직관 임계 Re
RE_CRIT_CURVATURE_COEFF = 12.0           # 곡률에 의한 임계 Re 증가 계수

# Dean Effect (Tube 측 추가 압력 손실 보정)
DEAN_FRICTION_FACTOR = 0.37             # Mishra-Gupta 상관식 Dean 보정 계수
DEAN_FRICTION_EXP = 0.36                # Dean 보정 지수

# 기하학적 여유 기준
MIN_SHELL_RADIAL_GAP_MM = 40.0           # Shell ↔ Coil 최소 여유 (열팽창/조립)
MANDREL_ASSEMBLY_GAP_MM = 10.0           # Mandrel ↔ Coil 조립/용접 여유
PITCH_RATIO_REF = 1.25                   # Pitch/OD 참조 비율 (TEMA)
PITCH_PENALTY_THRESHOLD = 1.25           # 페널티 적용 임계 Pitch Ratio
PITCH_PENALTY_SLOPE = 2.0               # 페널티 기울기
PITCH_PENALTY_MIN = 0.5                  # 페널티 최소값

# ASME Sec. VIII
MIN_SHELL_THICKNESS_MM = 6.0             # 최소 Shell 두께

# 유속 경고 기준 (m/s)
V_TUBE_LOW = 1.0
V_TUBE_HIGH = 3.0
V_SHELL_LOW = 0.2
V_SHELL_HIGH = 1.5

# 장비 길이 한계 (m)
MAX_EQUIPMENT_LENGTH_M = 10.0

# 최적화 탐색 범위
OPT_DC_STEP_MM = 10.0
OPT_DC_MAX_MM = 3000.0

# safe_div 유틸리티 최소값
EPSILON = 1e-6

# =========================================================
# 유틸리티 함수
# =========================================================
def safe_div(numerator, denominator, min_denom=EPSILON):
    """Division by zero 방어 유틸리티"""
    return numerator / max(min_denom, abs(denominator))


def calc_tube_side(Re, Pr, curvature_ratio, d_i_m, t_k, v_tube, rho, L_tube, d_o, d_i):
    """Tube 측 열전달 계수(h_i), 마찰계수(f_c), 압력손실(dp_bar) 통합 계산
    
    Returns:
        dict: h_i, Nu, f_c, dp_bar, Re_crit, De, flow_regime
    """
    De = Re * np.sqrt(max(0, curvature_ratio))
    Re_crit = RE_CRIT_BASE * (1.0 + RE_CRIT_CURVATURE_COEFF * np.sqrt(max(0, curvature_ratio)))
    
    is_laminar = Re < Re_crit
    
    if is_laminar:
        # Ito correlation (laminar helical)
        f_straight = 64.0 / max(Re, 1.0)
        f_c = f_straight * (1.0 + FRICTION_LAMINAR_LOG_COEFF * (np.log10(max(De, 1.0)))**4.0)
        Nu_straight = NU_LAMINAR_CONST
    else:
        # Mishra-Gupta correlation (turbulent helical)
        f_straight = FRICTION_TURBULENT_A / (max(Re, 1.0) ** FRICTION_TURBULENT_EXP)
        f_c = f_straight + FRICTION_TURBULENT_B * np.sqrt(max(0, curvature_ratio))
        Nu_straight = NU_TURBULENT_COEFF * (max(Re, 1.0) ** NU_TURBULENT_RE_EXP) * (Pr ** NU_TURBULENT_PR_EXP)
    
    # Coil curvature Nu enhancement
    Nu = Nu_straight * (1.0 + CURVATURE_NU_FACTOR * curvature_ratio)
    h_i = (Nu * t_k) / max(d_i_m, EPSILON)
    
    # Dean Effect factor for KPI
    dean_factor = f_c / f_straight if f_straight > 0 else 1.0
    
    # Pressure drop (Helical friction factor 적용)
    dp_coil = f_c * (L_tube / max(d_i_m, EPSILON)) * (rho * v_tube**2 / 2.0)
    dp_bar = dp_coil / 100000.0
    
    return {
        'h_i': h_i, 'Nu': Nu, 'f_c': f_c, 'dp_bar': dp_bar,
        'Re_crit': Re_crit, 'De': De,
        'flow_regime': 'Laminar' if is_laminar else 'Turbulent',
        'dean_factor': dean_factor
    }


def calc_shell_side(Re_shell, Pr_shell, s_k, d_o_m, pitch_ratio):
    """Shell 측 열전달 계수(h_o) 계산 — 층류/난류 분기 포함
    
    Returns:
        dict: h_o, Nu_shell, flow_regime, penalty_factor
    """
    # 층류/난류 분기 (Zukauskas correlation for crossflow over tube banks)
    if Re_shell < SHELL_RE_TRANSITION:
        Nu_shell = SHELL_NU_LAMINAR_COEFF * (max(Re_shell, 1.0) ** SHELL_NU_LAMINAR_RE_EXP) * (Pr_shell ** SHELL_NU_PR_EXP)
        flow_regime = 'Laminar'
    else:
        Nu_shell = SHELL_NU_TURBULENT_COEFF * (max(Re_shell, 1.0) ** SHELL_NU_TURBULENT_RE_EXP) * (Pr_shell ** SHELL_NU_PR_EXP)
        flow_regime = 'Turbulent'
    
    h_o = (Nu_shell * s_k) / max(d_o_m, EPSILON)
    
    # 코일 밀착 페널티
    penalty_factor = 1.0
    if pitch_ratio < PITCH_PENALTY_THRESHOLD:
        penalty_factor = max(PITCH_PENALTY_MIN, 1.0 - PITCH_PENALTY_SLOPE * (PITCH_PENALTY_THRESHOLD - pitch_ratio))
    h_o *= penalty_factor
    
    return {
        'h_o': h_o, 'Nu_shell': Nu_shell, 
        'flow_regime': flow_regime, 'penalty_factor': penalty_factor
    }


def calc_overall_U(h_i, h_o, R_fi, R_fo, R_wall, d_o, d_i):
    """총괄 열전달 계수(U) 및 개별 저항 계산"""
    ratio = d_o / max(d_i, EPSILON)
    _R_o = 1.0 / max(h_o, 0.1)
    _R_fo = R_fo
    _R_w = R_wall
    _R_fi = R_fi * ratio
    _R_i = ratio * (1.0 / max(h_i, 0.1))
    _R_tot = _R_o + _R_fo + _R_w + _R_fi + _R_i
    U = 1.0 / max(_R_tot, EPSILON)
    return {
        'U': U, 'R_tot': _R_tot,
        'R_o': _R_o, 'R_fo': _R_fo, 'R_w': _R_w, 'R_fi': _R_fi, 'R_i': _R_i
    }


# =========================================================
# [A] 글로벌 상태(Session State) 초기화 (이전 코드 유지)
# =========================================================
init_state = {
    'tag_no': 'HE-101', 
    'tube_fluid_name': 'Process Slurry',
    'shell_fluid_name': 'Hot Water / Steam',
    'fluid_type': "Liquid (뉴턴 유체 - 물, 오일 등)",
    't_rho': 998.0, 't_cp': 4180.0, 't_k': 0.6, 't_mu': 1.0, 
    's_rho': 998.0, 's_mu': 1.0, 's_cp': 4180.0, 's_k': 0.6,
    'rheology_model': "Power-law (멱법칙)",
    'tau_y': 5.0, 'plastic_visc': 0.05,
    'consistency_k': 0.1, 'flow_index_n': 0.8,
    'm_hot': 5000.0, 'm_cold': 8000.0,
    'T_hot_in': 30.0, 'T_hot_out': 80.0,
    'T_cold_in': 120.0, 'T_cold_out': 90.0,
    'allowable_dp_tube': 1.5, 'allowable_dp_shell': 0.5,
    'N_p': 3, 
    'd_o': 25.4, 't_thick': 2.11, 'D_c': 400.0, 'pitch': 50.0, 'D_s': 500.0,
    'shell_thick': 10.0,
    'D_mandrel': 350.0, 
    'tube_material': 'Stainless Steel 316 (k=16)', 'tube_k_wall': 16.0,
    'R_fi': 0.000176, 'R_fo': 0.000176,
    'overdesign_pct': 10.0,
    'design_p_shell': 10.0, 'allow_s_shell': 137.9, 'joint_e': 0.85, 'ca_shell': 3.0,
    'orientation': 'Vertical (수직형)'
}

for k, v in init_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

def apply_json():
    json_str = st.session_state['json_input_text']
    if not json_str.strip():
        st.warning("JSON 데이터를 입력하십시오.")
        return
    try:
        parsed_data = json.loads(json_str)
        for k in init_state.keys():
            if k in parsed_data:
                st.session_state[k] = parsed_data[k]
        st.success(f"✅ 설계 데이터(Tag: {st.session_state.get('tag_no')}) 로드 완료.")
    except Exception as e:
        st.error(f"🚨 데이터 로드 실패: {e}")

# =========================================================
# [B] 환경설정 및 사이드바 (이전 코드 유지)
# =========================================================
with st.sidebar:
    st.header("📋 Document Control")
    st.text_input("Item Tag No.", key='tag_no')
    st.markdown("---")
    st.header("💾 설계 시나리오 (Save/Load)")
    current_tag = st.session_state.get('tag_no', 'HE-101')
    current_data = {k: st.session_state[k] for k in init_state.keys()}
    filename = f"{current_tag}_design.json"
    st.download_button(f"📥 '{filename}' 다운로드", json.dumps(current_data, indent=4), file_name=filename, mime="application/json")
    st.text_area("JSON Load:", value="", key='json_input_text', height=150, help="여기에 JSON 텍스트를 붙여넣고 아래 버튼을 누르십시오.")
    st.button("시나리오 적용 (Load)", on_click=apply_json, use_container_width=True)

st.subheader(f"🏷️ Equipment Tag: **{st.session_state['tag_no']}**")

# =========================================================
# [C] 1. 유체 식별 및 물성치 (이전 코드 유지)
# =========================================================
st.subheader("1. 유체 식별 및 물성치")
st.radio("Tube 유체 상(Phase) 선택", ["Liquid (뉴턴 유체 - 물, 오일 등)", "Slurry (비뉴턴 유체 - 고농도 혼합물)"], key='fluid_type', horizontal=True)

col_tube, col_shell = st.columns(2)
with col_tube:
    st.markdown("#### **Tube-side (Inner)**")
    st.text_input("유체 명칭", key='tube_fluid_name')
    st.number_input("혼합 밀도 (kg/m³)", key='t_rho', help="일반 액체: 700 - 1000, 슬러리: 1100 - 1800 이상")
    st.number_input("비열 (J/kg·K)", key='t_cp', help="물: 4180, 일반 오일류: 1800 - 2400")
    st.number_input("열전도도 (W/m·K)", key='t_k', help="물: 0.6, 일반 오일류: 0.1 - 0.2")
    if "Liquid" in st.session_state['fluid_type']:
        st.number_input("점도 (cP)", format="%.2f", key='t_mu', help="물(20°C): 1.0 cP, 경질유: 2.0 - 10.0")
    else:
        st.selectbox("유변학 모델", ["Power-law (멱법칙)", "Bingham Plastic (빙햄 가소성)"], key='rheology_model')
        if "Bingham" in st.session_state['rheology_model']:
            st.number_input("항복 응력 (Pa)", key='tau_y', help="펄프/고농도 슬러리: 5 - 50 Pa")
            st.number_input("가소성 점도 (Pa·s)", format="%.4f", key='plastic_visc')
        else:
            st.number_input("점조도 지수 K (Pa·sⁿ)", format="%.4f", key='consistency_k')
            st.number_input("유동 지수 n", step=0.1, key='flow_index_n')

with col_shell:
    st.markdown("#### **Shell-side (Outer)**")
    st.text_input("유체 명칭", key='shell_fluid_name')
    st.number_input("밀도 (kg/m³)", key='s_rho')
    st.number_input("비열 (J/kg·K)", key='s_cp')
    st.number_input("열전도도 (W/m·K)", key='s_k')
    st.number_input("점도 (cP)", format="%.2f", key='s_mu')

st.markdown("---")

# =========================================================
# [D] 2. 공정 운전 조건 (Energy Balance) (이전 코드 유지)
# =========================================================
st.subheader("2. 공정 운전 조건 (Energy Balance)")

t_in = st.session_state['T_hot_in']
t_out = st.session_state['T_hot_out']
s_in = st.session_state['T_cold_in']
s_out = st.session_state['T_cold_out']
m_t = st.session_state['m_hot']
m_s = st.session_state['m_cold']

Q_kW = (m_t / 3600.0) * (st.session_state['t_cp'] / 1000.0) * abs(t_in - t_out)
s_cp_kJ = st.session_state['s_cp'] / 1000.0

is_tube_heating = (t_out > t_in)
delta_T_shell = abs(s_in - s_out)
est_m_cold = safe_div(Q_kW * 3600.0, s_cp_kJ * max(0.1, delta_T_shell)) if delta_T_shell > 0 else 0.0

if is_tube_heating:
    est_T_cold_out = s_in - safe_div(Q_kW * 3600.0, max(0.1, m_s) * s_cp_kJ)
    op_mode = "🔥 Heater Mode"
else:
    est_T_cold_out = s_in + safe_div(Q_kW * 3600.0, max(0.1, m_s) * s_cp_kJ)
    op_mode = "❄️ Cooler Mode"

col_pc1, col_pc2, col_pc3, col_pc4 = st.columns(4)
with col_pc1:
    st.number_input("Tube 유량 (kg/h)", step=100.0, key='m_hot')
    st.number_input("Shell 유량 (kg/h)", step=100.0, key='m_cold')
    st.caption(f"💡 필요 Shell 유량 추정치: **{est_m_cold:,.0f} kg/h**")
with col_pc2:
    st.number_input("Tube 입구 온도 (°C)", key='T_hot_in')
    st.number_input("Tube 목표 출구 온도 (°C)", key='T_hot_out')
    st.caption(f"**운전 모드: {op_mode}** (Q: {Q_kW:,.1f} kW)")
with col_pc3:
    st.number_input("Shell 입구 온도 (°C)", key='T_cold_in')
    st.number_input("Shell 목표 출구 온도 (°C)", key='T_cold_out')
    st.caption(f"💡 예상 Shell 출구 온도: **{est_T_cold_out:,.1f} °C**")
with col_pc4:
    st.number_input("Tube 허용 ΔP (bar)", 0.1, 10.0, step=0.1, key='allowable_dp_tube', help="TEMA 가이드: 0.5 - 0.7 bar 권장")
    st.number_input("Shell 허용 ΔP (bar)", 0.1, 10.0, step=0.1, key='allowable_dp_shell', help="Coil 외부 유동 특성상 0.3 - 0.5 bar 이내 설계 요망")

if is_tube_heating:
    dT1 = s_in - t_out  
    dT2 = s_out - t_in  
else:
    dT1 = t_in - s_out  
    dT2 = t_out - s_in  

lmtd_error = False
if dT1 == dT2 and dT1 > 0:
    LMTD = dT1
elif dT1 > 0 and dT2 > 0:
    LMTD = (dT1 - dT2) / np.log(dT1 / dT2)
else:
    LMTD = 1.0  
    lmtd_error = True

if lmtd_error:
    st.error("🚨 **열역학 에러 (Temperature Cross):** 열전달이 불가능한 온도 역전 현상이 발생했습니다.")

st.markdown("---")

# =========================================================
# [E] 3. 기하학적 설계 (Geometry Design) - 예외 처리 로직 추가 (이전 코드 유지)
# =========================================================
st.subheader("3. 기하학적 설계 (Geometry Design)")

st.radio("설치 방향 (Installation Orientation)", ["Vertical (수직형)", "Horizontal (수평형)"], key='orientation', horizontal=True, help="Footprint(바닥 면적) 계산을 위해 장비의 설치 방향을 선택하십시오.")
bbox_placeholder = st.empty()
st.markdown("<br>", unsafe_allow_html=True)

col_g1, col_g2, col_g3, col_g4 = st.columns(4)

with col_g1:
    st.number_input("Parallel Tubes (N_p, 가닥)", 1, 50, step=1, key='N_p', help="유량을 N_p개로 분산시킵니다. N_p가 커지면 Tube 상승각(Lead)이 가팔라집니다.")
    
    do_options = {'3/8" (9.53 mm)': 9.53, '1/2" (12.7 mm)': 12.7, '3/4" (19.05 mm)': 19.05, '1" (25.4 mm)': 25.4, 'Custom (직접 입력)': -1}
    do_keys = list(do_options.keys())
    do_vals = list(do_options.values())
    curr_do = st.session_state.get('d_o', 25.4)
    try: do_idx = do_vals.index(curr_do)
    except ValueError: do_idx = len(do_keys) - 1

    selected_do = st.selectbox("Tube OD (외경)", do_keys, index=do_idx, help="표준: 19.05 mm (3/4\"), 슬러리/고점도: 25.4 mm (1\") 이상 권장")
    if "Custom" in selected_do:
        safe_do = max(5.0, min(100.0, float(curr_do)))
        st.session_state['d_o'] = st.number_input("Tube OD 직접 입력 (mm)", 5.0, 100.0, value=safe_do, step=0.1)
    else:
        st.session_state['d_o'] = do_options[selected_do]

    with st.expander("💡 Tube OD(외경) 가이드"):
        st.markdown("| 규격 (inch) | 외경 (mm) | 추천 적용 분야 |\n|:---|:---|:---|\n| **3/8\"** | 9.53 | 초소형 장비용 |\n| **1/2\"** | 12.7 | 일반 컴팩트 설계 |\n| **3/4\"** | 19.05 | 압력 손실과 제작 편의성 균형 (표준) |\n| **1\"** | 25.4 | 슬러리 적용 시 플러깅 방지 권장 |")

    bwg_options = {'BWG 10 (3.40 mm)': 3.40, 'BWG 12 (2.77 mm)': 2.77, 'BWG 14 (2.11 mm)': 2.11, 'BWG 16 (1.65 mm)': 1.65, 'BWG 18 (1.24 mm)': 1.24, 'BWG 20 (0.89 mm)': 0.89, 'BWG 22 (0.71 mm)': 0.71, 'Custom (직접 입력)': -1}
    bwg_keys = list(bwg_options.keys())
    bwg_vals = list(bwg_options.values())
    curr_t = st.session_state.get('t_thick', 2.11)
    try: bwg_idx = bwg_vals.index(curr_t)
    except ValueError: bwg_idx = len(bwg_keys) - 1

    selected_bwg = st.selectbox("Tube Thickness (BWG)", bwg_keys, index=bwg_idx, help="일반적인 산업용 표준은 BWG 14 (2.11 mm) 또는 BWG 16 (1.65 mm) 입니다.")
    if "Custom" in selected_bwg:
        safe_t = max(0.5, min(10.0, float(curr_t)))
        st.session_state['t_thick'] = st.number_input("Tube 두께 직접 입력 (mm)", 0.5, 10.0, value=safe_t, step=0.1)
    else:
        st.session_state['t_thick'] = bwg_options[selected_bwg]

    with st.expander("💡 Tube Thickness(BWG) 가이드"):
        st.markdown("| BWG | mm | 특징 |\n|:---|:---|:---|\n| **10** | 3.40 | 고압/부식성 유체, 좁은 밴딩 |\n| **14** | 2.11 | 산업용 열교환기 표준 두께 |\n| **16** | 1.65 | 범용 표준 (유량/내압 균형) |\n| **20** | 0.89 | 계측기 또는 소구경 튜브용 |")

    d_i = st.session_state['d_o'] - 2 * st.session_state['t_thick']
    if d_i <= 0:
        st.error("🚨 Tube 두께 에러")
    else:
        st.caption(f"✓ 유효 내경 (Tube ID): **{d_i:.2f} mm**")

with col_g2:
    st.number_input("Coil Center Dia. (D_c, mm)", step=10.0, key='D_c', help="Coil 벤딩 시 파열을 막기 위해 Tube OD의 최소 10배 이상 권장")
    st.caption(f"💡 추천 최소값: **{st.session_state['d_o'] * 10.0:.1f} mm**")
    
    min_pitch = st.session_state['d_o']
    st.number_input("Coil Pitch (p, mm)", min_value=float(min_pitch), step=1.0, key='pitch', help="상하로 인접한 서로 다른 Tube 중심 간 수직 거리. p=OD일 경우 코일이 딱 붙습니다.")
    st.caption(f"💡 밀착 제작: **{min_pitch:.1f} mm** / TEMA 여유: **{min_pitch*PITCH_RATIO_REF:.1f} mm**")
    
    mat_dict = {'Stainless Steel 316 (k=16)': 16.0, 'Titanium (k=22)': 22.0, 'Custom (직접 입력)': -1}
    mat_keys = list(mat_dict.keys())
    mat_vals = list(mat_dict.values())
    curr_k = st.session_state.get('tube_k_wall', 16.0)
    try: mat_idx = mat_vals.index(curr_k)
    except ValueError: mat_idx = len(mat_keys) - 1

    selected_mat = st.selectbox("Tube Material", mat_keys, index=mat_idx)
    if "Custom" in selected_mat:
        st.session_state['tube_k_wall'] = st.number_input("열전도도 입력", value=float(curr_k))
    else:
        st.session_state['tube_k_wall'] = mat_dict[selected_mat]

with col_g3:
    st.number_input("Mandrel OD (mm)", step=5.0, key='D_mandrel', help="Coil 내측 공간을 채워 Shell 유체의 바이패스를 막는 코어 기둥입니다.")
    rec_mandrel = max(10.0, st.session_state['D_c'] - st.session_state['d_o'] - MANDREL_ASSEMBLY_GAP_MM)
    st.caption(f"💡 추천 최적값: **{rec_mandrel:.1f} mm** (Coil 내측 직경에서 조립/용접 여유 {MANDREL_ASSEMBLY_GAP_MM:.0f}mm 제외)")
    
    inner_clearance_rad = ((st.session_state['D_c'] - st.session_state['d_o']) - st.session_state['D_mandrel']) / 2.0
    if inner_clearance_rad < 0:
        st.error(f"🚨 간섭! Mandrel이 Coil을 파고듭니다 ({-inner_clearance_rad:.1f} mm)")
    
    ds_options = {
        'NPS 8" Pipe (ID: 202.7 mm)': 202.7,
        'NPS 10" Pipe (ID: 254.5 mm)': 254.5,
        'NPS 12" Pipe (ID: 304.8 mm)': 304.8,
        'NPS 14" Pipe (ID: 336.6 mm)': 336.6,
        'NPS 16" Pipe (ID: 387.4 mm)': 387.4,
        'NPS 18" Pipe (ID: 438.2 mm)': 438.2,
        'NPS 20" Pipe (ID: 489.0 mm)': 489.0,
        'NPS 24" Pipe (ID: 590.6 mm)': 590.6,
        'Custom (Rolled Plate, 직접입력)': -1
    }
    ds_keys = list(ds_options.keys())
    ds_vals = list(ds_options.values())
    curr_ds = st.session_state.get('D_s', 500.0)
    try: ds_idx = ds_vals.index(curr_ds)
    except ValueError: ds_idx = len(ds_keys) - 1

    selected_ds = st.selectbox("Shell ID (mm)", ds_keys, index=ds_idx, help="NPS 24인치 이하 중소형은 표준 파이프 사용이 원가에 유리하며, 대형은 50mm 단위로 압연 제작합니다.")
    if "Custom" in selected_ds:
        safe_ds = max(200.0, min(5000.0, float(curr_ds)))
        st.session_state['D_s'] = st.number_input("Shell ID 직접 입력 (50mm 단위 권장)", 200.0, 5000.0, value=safe_ds, step=50.0)
    else:
        st.session_state['D_s'] = ds_options[selected_ds]

    rec_Ds = st.session_state['D_c'] + st.session_state['d_o'] + MIN_SHELL_RADIAL_GAP_MM
    st.caption(f"💡 추천 최소값: **{rec_Ds:.1f} mm** (Coil 외경에서 열팽창/조립 여유 {MIN_SHELL_RADIAL_GAP_MM:.0f}mm 확보)")

    outer_clearance_rad = (st.session_state['D_s'] - (st.session_state['D_c'] + st.session_state['d_o'])) / 2.0
    if outer_clearance_rad < 0:
        st.error(f"🚨 간섭! Coil이 Shell을 뚫고 나갑니다 ({-outer_clearance_rad:.1f} mm)")

    with st.expander("💡 상업용 Shell 규격 가이드"):
        st.markdown("""
        **1. 중소형 쉘 (NPS 24인치 이하)**
        원가 절감을 위해 시중에서 유통되는 **Seamless / ERW Pipe**를 절단하여 쉘로 사용합니다. 위 드롭다운의 규격(Sch 40 기준 내경)을 선택하십시오.
        
        **2. 대형 쉘 (Custom Rolled Plate)**
        24인치를 초과하는 거대한 쉘은 철판(Plate)을 롤러로 말아서 용접 제작합니다. 철판 로스(Loss)를 최소화하기 위해 내경을 **50 mm 단위 (예: 800, 850, 900)**로 설정하는 것이 관례입니다.
        """)

with col_g4:
    st.markdown("#### 🛡️ 오염계수 및 여유율")
    st.number_input("Tube Fouling Factor (R_fi)", 0.0, 0.02, format="%.6f", key='R_fi', help="Tube 내부 유체의 스케일 저항값")
    st.number_input("Shell Fouling Factor (R_fo)", 0.0, 0.02, format="%.6f", key='R_fo', help="Tube 외부 스케일 저항값. 세척이 어려워 보수적으로 적용.")
    st.number_input("Overdesign (%)", 0.0, 100.0, step=1.0, key='overdesign_pct', help="계산된 필요 면적에 추가할 설계 안전 여유율 (통상 10~20%)")
    
    with st.expander("💡 TEMA Fouling 레퍼런스"):
        st.markdown("| 유체 | 오염계수 (m²·K/W) |\n|:---|:---|\n| 청정수 | 0.00018 |\n| 냉각수 | 0.00035 |\n| 공정 슬러리 | 0.00150+ |")

st.markdown("---")

# =========================================================
# [F] 4. 기계적 설계 (Mechanical Design) (이전 코드 유지)
# =========================================================
st.subheader("4. 기계적 설계 (Mechanical Design - ASME Sec.VIII)")

with st.expander("💡 ASME 기계 설계 가이드 (S, E, C.A.)"):
    st.markdown("""
    **1. 주요 재질별 허용 응력 (S)**
    | 재질 (Material) | ASME 규격 | 허용 응력 (MPa) |
    | :--- | :--- | :--- |
    | **일반 탄소강** | SA-516 Gr.70 | 137.9 |
    | **오스테나이트 스텐레스** | SA-240 304/316 | 115.0 - 137.0 |
    
    **2. 용접 조인트 효율 (E)**
    | RT 검사 수준 | 효율 (E) | 적용 기준 |
    | :--- | :--- | :--- |
    | **Full RT (전면 검사)** | 1.00 | 고압, 맹독성 유체 |
    | **Spot RT (국부 검사)** | 0.85 | 일반적인 Shell 표준 |
    
    **3. 부식 여유 (C.A.)**
    | 재질 및 환경 | C.A. (mm) | 비고 |
    | :--- | :--- | :--- |
    | **스텐레스강** | 0.0 - 1.5 | 부식 없음 가정 |
    | **탄소강 (일반)** | 3.0 | 일반 표준 (1/8 인치) |
    | **탄소강 (슬러리)** | 6.0 | 부식/마모 극심 |
    """)
    
cc1, cc2, cc3, cc4 = st.columns(4)
with cc1:
    st.number_input("Shell 설계 압력 (bar)", step=1.0, key='design_p_shell', help="운전 압력의 110% 또는 +1.5 bar")
with cc2:
    st.number_input("허용 응력 (S, MPa)", step=1.0, key='allow_s_shell', help="재질에 따른 ASME 허용 응력")
with cc3:
    st.number_input("용접 효율 (E)", max_value=1.0, key='joint_e', help="RT 검사 범위 (1.0 또는 0.85)")
with cc4:
    st.number_input("부식 여유 (C.A., mm)", step=0.5, key='ca_shell', help="탄소강 기본 3.0mm")

P_mpa = st.session_state['design_p_shell'] / 10.0
R_mm = st.session_state['D_s'] / 2.0
S_mpa = st.session_state['allow_s_shell']
E_eff = st.session_state['joint_e']
C_A = st.session_state['ca_shell']

t_req = (P_mpa * R_mm) / (S_mpa * E_eff - 0.6 * P_mpa) + C_A
t_final = max(MIN_SHELL_THICKNESS_MM, np.ceil(t_req))
st.session_state['shell_thick'] = t_final
shell_od = st.session_state['D_s'] + 2.0 * t_final

st.info(f"✓ 상업용 Shell Thickness: **{t_final:.0f} mm** 확정 (ASME 이론 두께: {t_req:.2f} mm)")

# =========================================================
# [G] 백그라운드 수력학/열역학 코어 연산 — 리팩토링된 함수 호출
# =========================================================
t_mu_pa = st.session_state.get('t_mu', 1.0) / 1000.0
s_mu_pa = st.session_state.get('s_mu', 1.0) / 1000.0
curvature_ratio = d_i / st.session_state['D_c'] if st.session_state['D_c'] > 0 else 0

m_hot_per_tube = (m_t / 3600.0) / max(1, st.session_state['N_p'])
A_c = np.pi * ((d_i / 1000.0) ** 2) / 4.0 if d_i > 0 else EPSILON
v_tube = m_hot_per_tube / (st.session_state['t_rho'] * A_c)

if "Liquid" in st.session_state['fluid_type']:
    Re = (st.session_state['t_rho'] * v_tube * (max(EPSILON, d_i) / 1000.0)) / max(EPSILON, t_mu_pa)
    Pr = (st.session_state['t_cp'] * t_mu_pa) / max(EPSILON, st.session_state['t_k'])
else:
    n_val = st.session_state['flow_index_n'] if "Power" in st.session_state['rheology_model'] else 1.0
    K_val = st.session_state['consistency_k'] if "Power" in st.session_state['rheology_model'] else st.session_state['plastic_visc']
    D_m_tube = max(EPSILON, d_i) / 1000.0
    term1 = st.session_state['t_rho'] * (v_tube ** (2.0 - n_val)) * (D_m_tube ** n_val)
    term2 = (8.0 ** (n_val - 1.0)) * max(K_val, 0.0001) * (((3.0 * n_val + 1.0) / (4.0 * n_val)) ** n_val)
    Re = safe_div(term1, term2) if term2 > 0 else 0.0
    mu_app = safe_div(term1, Re * v_tube) if (Re * v_tube) > 0 else 0.001
    Pr = (st.session_state['t_cp'] * mu_app) / max(EPSILON, st.session_state['t_k'])

m_cold_kg_s = m_s / 3600.0
D_s_m = st.session_state['D_s'] / 1000.0
D_man_m = st.session_state['D_mandrel'] / 1000.0
d_o_m = st.session_state['d_o'] / 1000.0

N_p_val = max(1, st.session_state['N_p'])
p_m = st.session_state['pitch'] / 1000.0
D_c_m = st.session_state['D_c'] / 1000.0
Lead_m = p_m * N_p_val
Length_per_Turn = np.sqrt((np.pi * D_c_m)**2 + Lead_m**2) if D_c_m > 0 else 1.0

A_annulus = (np.pi / 4.0) * (D_s_m**2 - D_man_m**2)
A_tube_cross = (np.pi / 4.0) * (d_o_m**2)
A_blocked = N_p_val * A_tube_cross * safe_div(Length_per_Turn, Lead_m)
A_free_flow = max(A_annulus * 0.1, A_annulus - A_blocked)

v_shell = safe_div(m_cold_kg_s, st.session_state['s_rho'] * A_free_flow) if A_free_flow > 0 else 0.0

D_e_shell = D_s_m - D_man_m
Re_shell = (st.session_state['s_rho'] * v_shell * D_e_shell) / max(EPSILON, s_mu_pa)
Pr_shell = (st.session_state['s_cp'] * s_mu_pa) / max(EPSILON, st.session_state['s_k'])

# --- 리팩토링된 Tube 측 계산 함수 호출 ---
d_i_m = max(EPSILON, d_i) / 1000.0
# 임시로 Area/Length 추정 (반복 계산 위해 L_tube 필요)
# 먼저 shell 측을 구해서 U→Area→Length 유도
pitch_ratio = st.session_state['pitch'] / max(EPSILON, st.session_state['d_o'])
shell_result = calc_shell_side(Re_shell, Pr_shell, st.session_state['s_k'], d_o_m, pitch_ratio)
h_o = shell_result['h_o']
penalty_factor = shell_result['penalty_factor']

R_wall = (d_o_m * np.log(st.session_state['d_o'] / max(EPSILON, d_i))) / (2.0 * max(EPSILON, st.session_state['tube_k_wall'])) if d_i > 0 else 0

# Tube 측 h_i (Dean factor는 dp 전용이므로 우선 Nu 계산)
tube_result_pre = calc_tube_side(Re, Pr, curvature_ratio, d_i_m, st.session_state['t_k'], v_tube, st.session_state['t_rho'], 1.0, st.session_state['d_o'], d_i)
h_i = tube_result_pre['h_i']

# 총괄 U 계산
u_result = calc_overall_U(h_i, h_o, st.session_state['R_fi'], st.session_state['R_fo'], R_wall, st.session_state['d_o'], d_i)
U_calc = u_result['U']

Area_req = (Q_kW * 1000.0) / (U_calc * LMTD) if not lmtd_error else 0.0
Area_design = Area_req * (1.0 + st.session_state['overdesign_pct'] / 100.0)

Total_Tube_Length = safe_div(Area_design, np.pi * d_o_m) if d_o_m > 0 else 0.0
Length_per_Tube = Total_Tube_Length / N_p_val
Turns_per_Tube = Length_per_Tube / Length_per_Turn

# 정확한 L_tube로 Tube 측 dp 재계산
tube_result = calc_tube_side(Re, Pr, curvature_ratio, d_i_m, st.session_state['t_k'], v_tube, st.session_state['t_rho'], Length_per_Tube, st.session_state['d_o'], d_i)
dp_tube_bar = tube_result['dp_bar']

L_shell_m = Turns_per_Tube * Lead_m
L_shell_mm = L_shell_m * 1000.0
# Shell ΔP: Zukauskas tube bank crossflow correlation
# Eu (Euler number per row) ≈ C × Re^n, then ΔP = Eu × N_rows × ρv²/2
_N_rows_shell = max(1, Turns_per_Tube * N_p_val)  # 유효 tube row 수
if Re_shell < SHELL_RE_TRANSITION:
    _Eu_per_row = 10.0 / max(Re_shell, 1.0)**0.5  # 층류: Eu ∝ Re^-0.5
else:
    _Eu_per_row = 1.0 / max(Re_shell, 1.0)**0.2   # 난류: Eu ∝ Re^-0.2
dp_shell_bar = (_Eu_per_row * _N_rows_shell * st.session_state['s_rho'] * v_shell**2 / 2.0) / 100000.0

# =========================================================
# [H] AI 최적화 제안 (Optimizer) — 리팩토링 (함수 재사용)
# =========================================================
opt_best_Dc = None
opt_min_LTT = float('inf')
opt_best_Dm = None
opt_best_Ds = None
opt_p_m = (st.session_state['d_o'] * PITCH_RATIO_REF) / 1000.0
opt_Lead_m = opt_p_m * N_p_val

for t_Dc in np.arange(st.session_state['d_o'] * 10.0, OPT_DC_MAX_MM, OPT_DC_STEP_MM):
    t_Dc_m = t_Dc / 1000.0
    t_Dm = max(10.0, t_Dc - st.session_state['d_o'] - MANDREL_ASSEMBLY_GAP_MM)
    t_Dm_m = t_Dm / 1000.0
    t_Ds = t_Dc + st.session_state['d_o'] + MIN_SHELL_RADIAL_GAP_MM
    t_Ds_m = t_Ds / 1000.0

    t_A_annulus = (np.pi / 4.0) * (t_Ds_m**2 - t_Dm_m**2)
    t_Length_per_Turn = np.sqrt((np.pi * t_Dc_m)**2 + opt_Lead_m**2) if t_Dc_m > 0 else 1.0
    t_A_blocked = N_p_val * ((np.pi / 4.0) * (d_o_m**2)) * safe_div(t_Length_per_Turn, opt_Lead_m)
    t_A_free = max(t_A_annulus * 0.1, t_A_annulus - t_A_blocked)

    t_v_shell = safe_div(m_cold_kg_s, st.session_state['s_rho'] * t_A_free) if t_A_free > 0 else 0.0
    t_Re_shell = (st.session_state['s_rho'] * t_v_shell * (t_Ds_m - t_Dm_m)) / max(EPSILON, s_mu_pa)
    
    # 리팩토링된 함수 사용 (Optimizer는 pitch = OD × 1.25 고정이므로 pitch_ratio = 1.25)
    t_shell_res = calc_shell_side(t_Re_shell, Pr_shell, st.session_state['s_k'], d_o_m, 
                                  PITCH_RATIO_REF)
    t_ho = t_shell_res['h_o']
    
    t_cr = d_i / t_Dc if t_Dc > 0 else 0
    t_Nu = (NU_LAMINAR_CONST if Re < RE_CRIT_BASE * (1.0 + RE_CRIT_CURVATURE_COEFF * np.sqrt(max(0, t_cr))) 
            else NU_TURBULENT_COEFF * (max(Re, 1.0) ** NU_TURBULENT_RE_EXP) * (Pr ** NU_TURBULENT_PR_EXP))
    t_hi = ((t_Nu * (1.0 + CURVATURE_NU_FACTOR * t_cr)) * st.session_state['t_k']) / max(d_i_m, EPSILON)
    
    t_u_res = calc_overall_U(t_hi, t_ho, st.session_state['R_fi'], st.session_state['R_fo'], R_wall, st.session_state['d_o'], d_i)
    t_U = t_u_res['U']
    t_Area_req = (Q_kW * 1000.0) / (t_U * LMTD) if not lmtd_error else 0.0
    t_Area_design = t_Area_req * (1.0 + st.session_state['overdesign_pct'] / 100.0)
    
    t_Turns = safe_div(t_Area_design, np.pi * d_o_m * N_p_val) / (np.sqrt((np.pi * t_Dc_m)**2 + opt_Lead_m**2) if t_Dc_m > 0 else 1.0)
    t_L_shell_m = t_Turns * opt_Lead_m
    t_L_TT = t_L_shell_m + (t_Ds_m / 2.0)
    
    if t_Dc_m < t_L_TT: 
        if t_L_TT < opt_min_LTT:
            opt_min_LTT = t_L_TT
            opt_best_Dc = t_Dc
            opt_best_Dm = t_Dm
            opt_best_Ds = t_Ds

# =========================================================
# [I] 실시간 Bounding Box 렌더링 (이전 코드 유지)
# =========================================================
Shell_TT_Length_m = L_shell_m + (D_s_m / 2.0)  # 2:1 Ellip. Head depth = D_s/4 × 2
Shell_TT_Length_mm = Shell_TT_Length_m * 1000.0
shell_od_m = shell_od / 1000.0

if "Vertical" in st.session_state['orientation']:
    Footprint_Area = (np.pi / 4.0) * (shell_od_m ** 2)
else:
    Footprint_Area = shell_od_m * Shell_TT_Length_m

with bbox_placeholder.container():
    st.markdown("#### 📐 실시간 장비 예상 규격 (Estimated Bounding Box)")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Shell OD (외경)", f"{shell_od:,.1f} mm")
    b2.metric("Coiled Section Height", f"{L_shell_mm:,.0f} mm")
    if Shell_TT_Length_m > MAX_EQUIPMENT_LENGTH_M:
        b3.metric("🚨 Shell Length (T/T)", f"{Shell_TT_Length_mm:,.0f} mm", delta="과도한 길이! 배관 불가", delta_color="inverse")
    else:
        b3.metric("Shell Length (T/T)", f"{Shell_TT_Length_mm:,.0f} mm", delta="안정적 구조", delta_color="normal")
    b4.metric("장비 바닥 면적 (Footprint)", f"{Footprint_Area:,.2f} m²", help="설치 방향(Vertical/Horizontal)에 따라 달라집니다.")
    
    if opt_best_Dc:
        st.success(f"💡 **[AI 최적화 제안]** N_p={N_p_val}가닥 기준, 최소 Shell 길이({opt_min_LTT*1000:,.0f} mm)를 달성하는 최적 콤보: **Coil D_c = {opt_best_Dc:,.0f} mm** (이때 D_m={opt_best_Dm:,.0f}, D_s={opt_best_Ds:,.0f}, Pitch={st.session_state['d_o']*PITCH_RATIO_REF:.1f})")
    st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# [J] 5. 상업용 데이터시트 검증 (Datasheet) — 색상 코딩 강화
# =========================================================
st.markdown("---")
st.subheader("5. 열전달 및 수력학 검증 (Datasheet & Report)")
st.caption(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with st.expander("💡 설계 유속(Velocity) 가이드라인 및 판정 기준"):
    st.markdown("""
    | 유체 경로 | 권장 유속 범위 | 초과/미달 시 발생 문제 |
    | :--- | :--- | :--- |
    | **Tube 측 (액체)** | 1.0 ~ 2.5 m/s | **< 1.0:** 침전물/오염 유발 <br> **> 3.0:** Tube 침식(Erosion) 및 파열 |
    | **Shell 측 (액체)** | 0.3 ~ 1.0 m/s | **< 0.2:** 열전달 사각지대 발생 <br> **> 1.5:** 유체 유발 진동(FIV)으로 코일 파손 |
    """)

if penalty_factor < 1.0:
    st.warning(f"⚠️ **코일 밀착 페널티 적용됨:** Pitch가 기준치({PITCH_RATIO_REF}*OD)보다 작아 코일 틈새 사각지대 현상을 반영했습니다. Shell 측 열전달 계수(h_o)가 {(1.0-penalty_factor)*100:.0f}% 삭감되었습니다.")

# --- 색상 코딩된 Metric 카드 (Priority 4) ---
st.markdown("#### 📊 핵심 성능 지표 (Key Performance Indicators)")

def _kpi_color(val, low, high, reverse=False):
    """범위 내 = 녹색, 약간 벗어남 = 주황, 크게 벗어남 = 적색"""
    if reverse:
        if val <= high: return "✅", "normal"
        elif val <= high * 1.5: return "⚠️", "off"
        else: return "🚨", "inverse"
    else:
        if low <= val <= high: return "✅", "normal"
        elif val < low: return "⚠️", "inverse"
        else: return "⚠️", "inverse"

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

# Tube Velocity KPI
_tv_icon, _tv_dc = _kpi_color(v_tube, V_TUBE_LOW, V_TUBE_HIGH)
kpi1.metric(f"{_tv_icon} Tube 유속", f"{v_tube:.2f} m/s", 
            delta=f"{'적정' if V_TUBE_LOW<=v_tube<=V_TUBE_HIGH else '주의'}", 
            delta_color=_tv_dc)

# Shell Velocity KPI
_sv_icon, _sv_dc = _kpi_color(v_shell, V_SHELL_LOW, V_SHELL_HIGH)
kpi2.metric(f"{_sv_icon} Shell 유속", f"{v_shell:.2f} m/s",
            delta=f"{'적정' if V_SHELL_LOW<=v_shell<=V_SHELL_HIGH else '주의'}",
            delta_color=_sv_dc)

# Tube ΔP KPI
_tdp_icon, _tdp_dc = _kpi_color(dp_tube_bar, 0, st.session_state['allowable_dp_tube'], reverse=True)
kpi3.metric(f"{_tdp_icon} Tube ΔP", f"{dp_tube_bar:.3f} bar",
            delta=f"허용: {st.session_state['allowable_dp_tube']} bar",
            delta_color=_tdp_dc)

# Shell ΔP KPI
_sdp_icon, _sdp_dc = _kpi_color(dp_shell_bar, 0, st.session_state['allowable_dp_shell'], reverse=True)
kpi4.metric(f"{_sdp_icon} Shell ΔP", f"{dp_shell_bar:.3f} bar",
            delta=f"허용: {st.session_state['allowable_dp_shell']} bar",
            delta_color=_sdp_dc)

# U-value
kpi5.metric("🔬 총괄 U", f"{U_calc:,.1f} W/m²K")

# Dean Factor
kpi6.metric("🌀 Dean 보정", f"×{tube_result['dean_factor']:.3f}",
            delta=f"Tube {tube_result['flow_regime']}",
            help="헬리컬 곡률에 의한 추가 압력 손실 보정 계수 (Mishra-Gupta)")

# Shell 흐름 영역 표시
st.caption(f"📌 **유동 영역:** Tube = {tube_result['flow_regime']} (Re={Re:,.0f}, De={tube_result['De']:,.0f}, Re_crit={tube_result['Re_crit']:,.0f}) | Shell = {shell_result['flow_regime']} (Re={Re_shell:,.0f})")

# --- 기존 Datasheet 테이블 ---
datasheet_md = f"""
| **Item Tag No.** | **{st.session_state['tag_no']}** | **Type** | Helical Coil Heat Exchanger |
| :--- | :--- | :--- | :--- |
| **Performance Data** | | | |
| Heat Duty (kW) | {Q_kW:,.2f} | Overall U-value (W/m²K) | {U_calc:,.1f} |
| Req. Area / Design Area | {Area_req:,.2f} m² / **{Area_design:,.2f} m²** (+{st.session_state['overdesign_pct']}%) | LMTD (°C) | {LMTD:,.1f} |
| **Process Conditions** | **Tube Side (Inner)** | **Shell Side (Outer)** | |
| Fluid Name | **{st.session_state['tube_fluid_name']}** | **{st.session_state['shell_fluid_name']}** | |
| Total Flow Rate (kg/h) | {st.session_state['m_hot']:,.0f} | {st.session_state['m_cold']:,.0f} | |
| Temp. In / Out (°C) | {st.session_state['T_hot_in']} / {st.session_state['T_hot_out']} | {st.session_state['T_cold_in']} / {st.session_state['T_cold_out']} | |
| Velocity (m/s) | **{v_tube:.2f}** | **{v_shell:.2f}** | |
| Reynolds Number | {Re:,.0f} ({tube_result['flow_regime']}) | {Re_shell:,.0f} ({shell_result['flow_regime']}) | |
| Dean Number (De) | {tube_result['De']:,.0f} | — | |
| Dean ΔP Factor | ×{tube_result['dean_factor']:.3f} | — | |
| Calc. Press. Drop (bar)| **{dp_tube_bar:.3f}** (Allow: {st.session_state['allowable_dp_tube']}) | **{dp_shell_bar:.3f}** (Allow: {st.session_state['allowable_dp_shell']}) | |
| **Mechanical Design** | | | |
| **[Tube]** OD x Thick. (mm) | {st.session_state['d_o']} x {st.session_state['t_thick']} | **[Tube]** Material | {st.session_state['tube_material']} |
| **[Tube]** Parallel Coils (N_p)| **{st.session_state['N_p']} ea** | **[Tube]** Length per Tube | {Length_per_Tube:,.1f} m |
| **[Coil]** Center Dia. (D_c) | {st.session_state['D_c']} mm | **[Coil]** Pitch (Gap) | {st.session_state['pitch']} mm |
| **[Coil]** Turns per Tube | {Turns_per_Tube:,.1f} turns | **[Install]** Orientation | {st.session_state['orientation']} |
| **[Shell]** ID / Mandrel OD | {st.session_state['D_s']} mm / {st.session_state['D_mandrel']} mm | **[Shell]** OD x Thick. (mm) | **{shell_od:.1f} x {st.session_state['shell_thick']:.0f}** |
| **[Shell]** T/T Length (mm) | **{Shell_TT_Length_mm:,.0f} mm** | | |
"""
st.markdown(datasheet_md)

html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{st.session_state['tag_no']} - Heat Exchanger Datasheet</title>
    <style>
        body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333; line-height: 1.6; margin: 20px; }}
        .header {{ text-align: center; border-bottom: 3px solid #004488; padding-bottom: 10px; margin-bottom: 30px; }}
        h2 {{ margin: 0; color: #004488; font-size: 24px; }}
        .meta-info {{ font-size: 12px; color: #666; text-align: right; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 12px; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
        th {{ background-color: #f4f7f6; font-weight: bold; color: #333; }}
        .section-title {{ background-color: #004488; color: white; padding: 6px 12px; font-size: 14px; font-weight: bold; }}
        @media print {{
            body {{ margin: 0; padding: 20px; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="no-print" style="background-color: #fff3cd; padding: 10px; border: 1px solid #ffeeba; margin-bottom: 20px; font-size: 14px;">
        💡 <b>엔지니어 가이드:</b> 완벽한 PDF를 얻으려면 <code>Ctrl + P</code> (인쇄)를 누른 뒤, 대상을 <b>'PDF로 저장'</b>으로 변경하십시오.
    </div>
    
    <div class="header">
        <h2>COMMERCIAL DATASHEET</h2>
        <p style="margin:5px 0; font-weight:bold;">Helical Coil Heat Exchanger</p>
    </div>
    
    <div class="meta-info">Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    
    <table>
        <tr><td class="section-title" colspan="4">1. General Information</td></tr>
        <tr><th>Item Tag No.</th><td><b>{st.session_state['tag_no']}</b></td><th>Overall U-value</th><td>{U_calc:,.1f} W/m²K</td></tr>
        <tr><th>Heat Duty</th><td>{Q_kW:,.2f} kW</td><th>Req. / Design Area</th><td>{Area_req:,.2f} / <b>{Area_design:,.2f} m²</b> (+{st.session_state['overdesign_pct']}%)</td></tr>
        <tr><th>LMTD</th><td>{LMTD:,.1f} &deg;C</td><th>Operation Mode</th><td>{op_mode}</td></tr>
        
        <tr><td class="section-title" colspan="4">2. Process Conditions</td></tr>
        <tr><th>Parameter</th><th colspan="1">Tube Side (Inner)</th><th colspan="2">Shell Side (Outer)</th></tr>
        <tr><td>Fluid Name</td><td colspan="1">{st.session_state['tube_fluid_name']}</td><td colspan="2">{st.session_state['shell_fluid_name']}</td></tr>
        <tr><td>Flow Rate (kg/h)</td><td colspan="1">{st.session_state['m_hot']:,.0f}</td><td colspan="2">{st.session_state['m_cold']:,.0f}</td></tr>
        <tr><td>Temp. In / Out (&deg;C)</td><td colspan="1">{st.session_state['T_hot_in']} / {st.session_state['T_hot_out']}</td><td colspan="2">{st.session_state['T_cold_in']} / {st.session_state['T_cold_out']}</td></tr>
        <tr><td>Velocity (m/s)</td><td colspan="1">{v_tube:.2f}</td><td colspan="2">{v_shell:.2f}</td></tr>
        <tr><td>Reynolds Number</td><td colspan="1">{Re:,.0f} ({tube_result['flow_regime']})</td><td colspan="2">{Re_shell:,.0f} ({shell_result['flow_regime']})</td></tr>
        <tr><td>Dean Number (De)</td><td colspan="1">{tube_result['De']:,.0f}</td><td colspan="2">&mdash;</td></tr>
        <tr><td>Dean &Delta;P Factor</td><td colspan="1">&times;{tube_result['dean_factor']:.3f}</td><td colspan="2">&mdash;</td></tr>
        <tr><td>Pressure Drop (bar)</td><td colspan="1"><b>{dp_tube_bar:.3f}</b> (Allow: {st.session_state['allowable_dp_tube']})</td><td colspan="2"><b>{dp_shell_bar:.3f}</b> (Allow: {st.session_state['allowable_dp_shell']})</td></tr>
        <tr><td>Fouling Factor</td><td colspan="1">{st.session_state['R_fi']:.6f}</td><td colspan="2">{st.session_state['R_fo']:.6f}</td></tr>
        
        <tr><td class="section-title" colspan="4">3. Mechanical Design (ASME Sec.VIII)</td></tr>
        <tr><th>[Tube] OD x Thick. (mm)</th><td>{st.session_state['d_o']} x {st.session_state['t_thick']}</td><th>[Tube] Material</th><td>{st.session_state['tube_material']}</td></tr>
        <tr><th>[Tube] Parallel Coils (N_p)</th><td>{st.session_state['N_p']} ea</td><th>[Tube] Length per Tube</th><td>{Length_per_Tube:,.1f} m</td></tr>
        <tr><th>[Coil] Center Dia. (D_c)</th><td>{st.session_state['D_c']} mm</td><th>[Coil] Pitch (Gap)</th><td>{st.session_state['pitch']} mm</td></tr>
        <tr><th>[Coil] Turns per Tube</th><td>{Turns_per_Tube:,.1f} turns</td><th>[Install] Orientation</th><td>{st.session_state['orientation']}</td></tr>
        <tr><th>[Shell] ID / Mandrel OD</th><td>{st.session_state['D_s']} mm / {st.session_state['D_mandrel']} mm</td><th>[Shell] OD x Thick. (mm)</th><td>{shell_od:.1f} x {st.session_state['shell_thick']:.0f}</td></tr>
        <tr><th>[Shell] T/T Length (mm)</th><td colspan="3" style="font-size:16px;"><b>{Shell_TT_Length_mm:,.0f} mm</b></td></tr>
    </table>
</body>
</html>
"""

col_dl1, col_dl2 = st.columns([1, 2])
with col_dl1:
    st.download_button(label="📄 Datasheet 다운로드 (HTML/PDF용)", data=html_report, file_name=f"{st.session_state['tag_no']}_Datasheet.html", mime="text/html")
with col_dl2:
    st.info("💡 폰트 에러 없는 PDF 출력을 위해 HTML로 내보냅니다. 브라우저 인쇄(Ctrl+P) 기능을 활용하세요.")

err_msg = []
if lmtd_error: err_msg.append("Temperature Cross (온도 역전) 발생")
if inner_clearance_rad < 0: err_msg.append("Mandrel - Coil 내측 간섭 발생")
if outer_clearance_rad < 0: err_msg.append("Shell - Coil 외측 간섭 발생")
if dp_tube_bar > st.session_state['allowable_dp_tube']: err_msg.append(f"Tube 측 ΔP 초과")
if dp_shell_bar > st.session_state['allowable_dp_shell']: err_msg.append(f"Shell 측 ΔP 초과")
if Shell_TT_Length_m > MAX_EQUIPMENT_LENGTH_M: err_msg.append(f"장비 총 길이 {MAX_EQUIPMENT_LENGTH_M:.0f}m 초과 (레이아웃 한계)")
if d_i <= 0: err_msg.append("내경(ID) 계산 불가")

if v_tube < V_TUBE_LOW: err_msg.append("Tube 유속 저하 (오염/침전 위험)")
if v_tube > V_TUBE_HIGH: err_msg.append("Tube 유속 초과 (침식 위험)")
if v_shell < V_SHELL_LOW: err_msg.append("Shell 유속 저하 (열전달 사각지대 위험)")
if v_shell > V_SHELL_HIGH: err_msg.append("Shell 유속 초과 (진동/파손 위험)")

if err_msg:
    st.error("🚨 **Datasheet Warning:** " + " / ".join(err_msg))
else:
    st.success("✅ **Datasheet Validated:** 모든 공정, 수력학, 기계적 제약 조건을 통과했습니다.")

# =========================================================
# [K] 6. 3D 형상 렌더링 (🌟 Real 3D Mesh Tube)
# =========================================================
st.markdown("---")
st.subheader("6. 3D 코일 형상 (Schematic Representation)")

if Turns_per_Tube > 0 and Turns_per_Tube < 2000 and d_i > 0 and not lmtd_error:
    fig = go.Figure()
    
    t_max_full = Turns_per_Tube * 2 * np.pi
    coil_height = (Lead_m * 1000.0 / (2 * np.pi)) * t_max_full if Turns_per_Tube > 0 else 1.0
    
    # 🌟 Real 3D Mesh 렌더링 (Option 1: 브라우저 부하 방지를 위해 최대 3바퀴까지만 실제 볼륨으로 렌더링)
    render_turns = min(Turns_per_Tube, 3.0)
    t_max_render = render_turns * 2 * np.pi
    num_t = int(max(render_turns * 40, 50))
    num_theta = 12
    t_vals = np.linspace(0, t_max_render, num_t)
    theta_vals = np.linspace(0, 2 * np.pi, num_theta)
    T_grid, Theta_grid = np.meshgrid(t_vals, theta_vals)
    
    R_c = st.session_state['D_c'] / 2.0
    r_tube = st.session_state['d_o'] / 2.0
    c_val = (Lead_m * 1000.0) / (2 * np.pi)
    denom = np.sqrt(R_c**2 + c_val**2) if (R_c**2 + c_val**2) > 0 else 1.0
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i in range(int(N_p_val)):
        angle_offset = i * (2 * np.pi / N_p_val)
        t_shifted = T_grid + angle_offset
        
        # 튜브 중심선
        C_x = R_c * np.cos(t_shifted)
        C_y = R_c * np.sin(t_shifted)
        C_z = c_val * T_grid
        
        # 기하학적 법선 벡터 (Normal)
        N_x = -np.cos(t_shifted)
        N_y = -np.sin(t_shifted)
        N_z = np.zeros_like(t_shifted)
        
        # 기하학적 종법선 벡터 (Binormal)
        B_x = (c_val / denom) * np.sin(t_shifted)
        B_y = -(c_val / denom) * np.cos(t_shifted)
        B_z = (R_c / denom) * np.ones_like(t_shifted)
        
        # 매개변수 곡면 방정식 (Parametric Surface)
        X = C_x + r_tube * (N_x * np.cos(Theta_grid) + B_x * np.sin(Theta_grid))
        Y = C_y + r_tube * (N_y * np.cos(Theta_grid) + B_y * np.sin(Theta_grid))
        Z = C_z + r_tube * (N_z * np.cos(Theta_grid) + B_z * np.sin(Theta_grid))
        
        tube_color = colors[i % len(colors)]
        
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, tube_color], [1, tube_color]],
            showscale=False,
            name=f'Coil {i+1} (Real OD)',
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.5, roughness=0.5),
            hoverinfo='skip'
        ))
        
    # 만약 실제 코일이 3바퀴보다 크다면, 상단에 잘렸음을 명시하는 3D 텍스트 추가
    if Turns_per_Tube > 3.0:
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[c_val * t_max_render + st.session_state['d_o'] * 2.0],
            mode='text',
            text=["(Coil Rendering Truncated to 3 Turns for Performance)"],
            textposition="top center",
            textfont=dict(color='red', size=14),
            name='Truncation Info',
            hoverinfo='skip'
        ))
        
    z_surf = np.linspace(0, coil_height, 20)
    theta_surf = np.linspace(0, 2*np.pi, 25)
    theta_grid, z_grid = np.meshgrid(theta_surf, z_surf)
    
    x_man = (st.session_state['D_mandrel'] / 2) * np.cos(theta_grid)
    y_man = (st.session_state['D_mandrel'] / 2) * np.sin(theta_grid)
    fig.add_trace(go.Surface(
        x=x_man, y=y_man, z=z_grid, 
        opacity=0.6, 
        colorscale=[[0, '#666666'], [1, '#999999']], 
        showscale=False, name='Mandrel', 
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.5), hoverinfo='skip'
    ))
    
    x_shell = (st.session_state['D_s'] / 2) * np.cos(theta_grid)
    y_shell = (st.session_state['D_s'] / 2) * np.sin(theta_grid)
    fig.add_trace(go.Surface(x=x_shell, y=y_shell, z=z_grid, opacity=0.08, colorscale='Blues', showscale=False, name='Shell', hoverinfo='skip'))
    
    noz_h = st.session_state['d_o'] * 3.0
    in_x = (st.session_state['D_c'] / 2) * np.cos(0)
    in_y = (st.session_state['D_c'] / 2) * np.sin(0)
    fig.add_trace(go.Scatter3d(x=[in_x, in_x], y=[in_y, in_y], z=[coil_height, coil_height + noz_h], mode='lines', line=dict(color='red', width=12), name='Tube Inlet'))
    out_x = (st.session_state['D_c'] / 2) * np.cos(t_max_full % (2 * np.pi))
    out_y = (st.session_state['D_c'] / 2) * np.sin(t_max_full % (2 * np.pi))
    fig.add_trace(go.Scatter3d(x=[out_x, out_x], y=[out_y, out_y], z=[0, -noz_h], mode='lines', line=dict(color='red', width=12), name='Tube Outlet'))
    
    sh_in_r = st.session_state['D_s'] / 2.0
    fig.add_trace(go.Scatter3d(x=[sh_in_r, sh_in_r + noz_h], y=[0, 0], z=[Lead_m*1000/2.0, Lead_m*1000/2.0], mode='lines', line=dict(color='blue', width=12), name='Shell Inlet'))
    fig.add_trace(go.Scatter3d(x=[-sh_in_r, -sh_in_r - noz_h], y=[0, 0], z=[coil_height - Lead_m*1000/2.0, coil_height - Lead_m*1000/2.0], mode='lines', line=dict(color='blue', width=12), name='Shell Outlet'))
    
    top_z = coil_height + 50.0
    pos_man = st.session_state['D_mandrel'] / 2.0
    pos_inner_clr = pos_man + inner_clearance_rad / 2.0
    pos_coil = st.session_state['D_c'] / 2.0
    pos_outer_clr = pos_coil + st.session_state['d_o']/2.0 + outer_clearance_rad / 2.0
    pos_shell = st.session_state['D_s'] / 2.0
    
    text_x = [pos_man, pos_inner_clr, pos_coil, pos_outer_clr, pos_shell]
    text_y = [0, 0, 0, 0, 0]
    text_z = [top_z, top_z, top_z, top_z, top_z]
    text_labels = [
        f"Mandrel OD<br>{st.session_state['D_mandrel']}",
        f"Inner Clr.<br>{inner_clearance_rad:.1f}",
        f"Coil D_c {st.session_state['D_c']}<br>(Tube OD {st.session_state['d_o']})",
        f"Outer Clr.<br>{outer_clearance_rad:.1f}",
        f"Shell ID<br>{st.session_state['D_s']}"
    ]
    
    fig.add_trace(go.Scatter3d(
        x=text_x, y=text_y, z=text_z,
        mode='text+markers',
        text=text_labels,
        textposition="top center",
        marker=dict(size=4, color='black'),
        name='Clearance Info',
        hoverinfo='skip'
    ))

    fig.update_layout(scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Height (mm)', aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0), height=700, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("형상을 렌더링할 수 없습니다. 물리적 변수를 다시 확인하십시오.")

# =========================================================
# [L] 7. 2D 엔지니어링 도면 및 열저항 해석
# =========================================================
st.markdown("---")
st.subheader("7. 2D 엔지니어링 도면 및 열저항 해석")

# --- [L-1] 열전달 저항 분해 (Thermal Resistance Breakdown) ---
_R_i = u_result['R_i']
_R_fi = u_result['R_fi']
_R_w = u_result['R_w']
_R_fo = u_result['R_fo']
_R_o = u_result['R_o']
_R_tot = u_result['R_tot']

with st.expander("📊 열전달 저항 분해 (Thermal Resistance Breakdown)", expanded=True):
    _rl = ['Shell 외측 필름 (1/h_o)', 'Shell 오염 (R_fo)', '튜브 벽 (R_wall)', 'Tube 오염 (R_fi)', 'Tube 내측 필름 (1/h_i)']
    _rv = [_R_o, _R_fo, _R_w, _R_fi, _R_i]
    _rp = [v / max(_R_tot, 1e-12) * 100 for v in _rv]
    _rc = ['#2E7D32', '#66BB6A', '#FF9800', '#42A5F5', '#1565C0']
    fig_rb = go.Figure()
    fig_rb.add_trace(go.Bar(y=_rl, x=_rv, orientation='h', marker_color=_rc,
        text=[f'{v:.6f} ({p:.1f}%)' for v, p in zip(_rv, _rp)],
        textposition='outside', textfont=dict(size=11)))
    fig_rb.update_layout(
        title=dict(text=f'<b>총 열저항</b> = {_R_tot:.6f} m²K/W  →  U = {1.0/max(_R_tot,1e-12):.1f} W/m²K', font=dict(size=13)),
        xaxis_title='열저항 (m²·K/W)', height=280,
        margin=dict(l=10, r=130, t=40, b=35),
        yaxis=dict(autorange='reversed'), plot_bgcolor='white')
    st.plotly_chart(fig_rb, use_container_width=True)
    _mi = _rp.index(max(_rp))
    st.caption(f"💡 **지배 저항:** {_rl[_mi]} ({_rp[_mi]:.1f}%) — 이 저항을 줄이면 U-value가 가장 효과적으로 향상됩니다.")

# --- [L-2] 2D 도면 탭 ---
if Turns_per_Tube > 0 and d_i > 0 and not lmtd_error:
    tab_cs, tab_ls, tab_uw = st.tabs(["⭕ 횡단면도 (Cross-Section)", "📐 종단면도 (Longitudinal)", "📏 코일 전개도 (Unwound)"])

    # ========== TAB 1: 횡단면도 ==========
    with tab_cs:
        st.caption("🔍 코일 중심축에 직교하는 횡단면 — Shell / Mandrel / Coil 동심 구조 및 Clearance")
        fig_cs = go.Figure()
        _th = np.linspace(0, 2*np.pi, 200)
        _r_sod = shell_od / 2.0
        _r_sid = st.session_state['D_s'] / 2.0
        _r_man = st.session_state['D_mandrel'] / 2.0
        _r_cl = st.session_state['D_c'] / 2.0
        _r_to = st.session_state['d_o'] / 2.0
        _r_ti = d_i / 2.0

        # Shell wall ring (OD→ID)
        fig_cs.add_trace(go.Scatter(
            x=np.concatenate([_r_sod*np.cos(_th), _r_sid*np.cos(_th[::-1])]).tolist(),
            y=np.concatenate([_r_sod*np.sin(_th), _r_sid*np.sin(_th[::-1])]).tolist(),
            fill='toself', fillcolor='rgba(120,144,156,0.3)', mode='lines',
            line=dict(color='#455A64', width=2),
            name=f'Shell Wall (t={st.session_state["shell_thick"]:.0f}mm)'))

        # Mandrel (solid)
        fig_cs.add_trace(go.Scatter(
            x=(_r_man*np.cos(_th)).tolist(), y=(_r_man*np.sin(_th)).tolist(),
            fill='toself', fillcolor='rgba(117,117,117,0.3)', mode='lines',
            line=dict(color='#616161', width=2),
            name=f'Mandrel Ø{st.session_state["D_mandrel"]:.1f}'))

        # Coil CL (D_c)
        fig_cs.add_trace(go.Scatter(
            x=(_r_cl*np.cos(_th)).tolist(), y=(_r_cl*np.sin(_th)).tolist(),
            mode='lines', line=dict(color='#E91E63', width=1.5, dash='dashdot'),
            name=f'Coil CL (D_c={st.session_state["D_c"]:.1f})'))

        # Tube cross-sections
        _tt = np.linspace(0, 2*np.pi, 80)
        _t_lc = ['#1565C0','#E65100','#2E7D32','#C62828','#6A1B9A','#4E342E','#00838F','#AD1457']
        _t_fc = ['rgba(21,101,192,0.3)','rgba(230,81,0,0.3)','rgba(46,125,50,0.3)','rgba(198,40,40,0.3)',
                 'rgba(106,27,154,0.3)','rgba(78,52,46,0.3)','rgba(0,131,143,0.3)','rgba(173,20,87,0.3)']
        for _i in range(N_p_val):
            _a = _i * (2*np.pi / N_p_val)
            _cx = _r_cl * np.cos(_a)
            _cy = _r_cl * np.sin(_a)
            _ci = _i % len(_t_lc)
            if _r_ti > 0:
                fig_cs.add_trace(go.Scatter(
                    x=np.concatenate([_cx+_r_to*np.cos(_tt), _cx+_r_ti*np.cos(_tt[::-1])]).tolist(),
                    y=np.concatenate([_cy+_r_to*np.sin(_tt), _cy+_r_ti*np.sin(_tt[::-1])]).tolist(),
                    fill='toself', fillcolor=_t_fc[_ci], mode='lines',
                    line=dict(color=_t_lc[_ci], width=1.5), name=f'Tube #{_i+1}'))
                fig_cs.add_trace(go.Scatter(
                    x=(_cx+_r_ti*np.cos(_tt)).tolist(), y=(_cy+_r_ti*np.sin(_tt)).tolist(),
                    fill='toself', fillcolor='rgba(240,248,255,0.8)', mode='lines',
                    line=dict(color=_t_lc[_ci], width=0.8, dash='dot'), showlegend=False))

        # Dimension annotations
        _shapes_cs = []
        _annots_cs = []

        # Inner Clearance (45° direction)
        _ie = _r_cl - _r_to
        if inner_clearance_rad > 2:
            _ad = np.pi / 4
            _shapes_cs.append(dict(type='line', x0=_r_man*np.cos(_ad), y0=_r_man*np.sin(_ad),
                x1=_ie*np.cos(_ad), y1=_ie*np.sin(_ad), line=dict(color='#D32F2F', width=1.5, dash='dot')))
            _annots_cs.append(dict(x=(_r_man+_ie)/2*np.cos(_ad), y=(_r_man+_ie)/2*np.sin(_ad),
                text=f'<b>Inner Clr.={inner_clearance_rad:.1f}</b>', showarrow=False,
                font=dict(size=10, color='#D32F2F'), bgcolor='rgba(255,255,255,0.9)', yshift=12))

        # Outer Clearance (45° direction)
        _oe = _r_cl + _r_to
        if outer_clearance_rad > 2:
            _ad = np.pi / 4
            _shapes_cs.append(dict(type='line', x0=_oe*np.cos(_ad), y0=_oe*np.sin(_ad),
                x1=_r_sid*np.cos(_ad), y1=_r_sid*np.sin(_ad), line=dict(color='#1565C0', width=1.5, dash='dot')))
            _annots_cs.append(dict(x=(_oe+_r_sid)/2*np.cos(_ad), y=(_oe+_r_sid)/2*np.sin(_ad),
                text=f'<b>Outer Clr.={outer_clearance_rad:.1f}</b>', showarrow=False,
                font=dict(size=10, color='#1565C0'), bgcolor='rgba(255,255,255,0.9)', yshift=12))

        # Shell ID dimension (top)
        _dy = _r_sod + 20
        _shapes_cs.extend([
            dict(type='line', x0=-_r_sid, y0=_dy, x1=_r_sid, y1=_dy, line=dict(color='#333', width=1)),
            dict(type='line', x0=-_r_sid, y0=_r_sid+5, x1=-_r_sid, y1=_dy+10, line=dict(color='#333', width=0.5)),
            dict(type='line', x0=_r_sid, y0=_r_sid+5, x1=_r_sid, y1=_dy+10, line=dict(color='#333', width=0.5))])
        _annots_cs.append(dict(x=0, y=_dy, text=f'<b>Shell ID = {st.session_state["D_s"]:.1f} mm</b>',
            showarrow=False, font=dict(size=11, color='#333'), bgcolor='rgba(255,255,255,0.9)', yshift=14))

        # Mandrel OD dimension (bottom)
        _dy2 = -(_r_sod + 20)
        _shapes_cs.extend([
            dict(type='line', x0=-_r_man, y0=_dy2, x1=_r_man, y1=_dy2, line=dict(color='#616161', width=1)),
            dict(type='line', x0=-_r_man, y0=-_r_man-5, x1=-_r_man, y1=_dy2-10, line=dict(color='#616161', width=0.5)),
            dict(type='line', x0=_r_man, y0=-_r_man-5, x1=_r_man, y1=_dy2-10, line=dict(color='#616161', width=0.5))])
        _annots_cs.append(dict(x=0, y=_dy2, text=f'Mandrel OD = {st.session_state["D_mandrel"]:.1f} mm',
            showarrow=False, font=dict(size=10, color='#616161'), bgcolor='rgba(255,255,255,0.9)', yshift=-14))

        # D_c dimension (left)
        _dx = -(_r_sod + 20)
        _shapes_cs.extend([
            dict(type='line', x0=_dx, y0=-_r_cl, x1=_dx, y1=_r_cl, line=dict(color='#E91E63', width=1)),
            dict(type='line', x0=-_r_cl-5, y0=-_r_cl, x1=_dx-10, y1=-_r_cl, line=dict(color='#E91E63', width=0.5)),
            dict(type='line', x0=-_r_cl-5, y0=_r_cl, x1=_dx-10, y1=_r_cl, line=dict(color='#E91E63', width=0.5))])
        _annots_cs.append(dict(x=_dx, y=0, text=f'D_c={st.session_state["D_c"]:.1f}',
            showarrow=False, font=dict(size=10, color='#E91E63'), bgcolor='rgba(255,255,255,0.9)',
            xshift=-5, textangle=-90))

        # Tube callout
        if N_p_val > 0:
            _annots_cs.append(dict(x=_r_cl+_r_to+5, y=_r_to+15,
                text=f'OD={st.session_state["d_o"]:.1f}<br>ID={d_i:.1f}<br>t={st.session_state["t_thick"]:.2f}',
                showarrow=True, arrowhead=2, arrowcolor='#1565C0', ax=50, ay=-30,
                font=dict(size=9, color='#1565C0'), bgcolor='rgba(255,255,255,0.9)'))

        fig_cs.update_layout(
            xaxis=dict(scaleanchor='y', scaleratio=1, showgrid=False, zeroline=False, title='mm'),
            yaxis=dict(showgrid=False, zeroline=False, title='mm'),
            height=650, margin=dict(l=60, r=40, t=30, b=40),
            plot_bgcolor='white', shapes=_shapes_cs, annotations=_annots_cs,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255,255,255,0.9)'))
        st.plotly_chart(fig_cs, use_container_width=True)

    # ========== TAB 2: 종단면도 (전면 재작성) ==========
    with tab_ls:
        st.caption("🔍 중심축을 관통하는 종단면 — Shell 길이, Pitch, 노즐 배치 (헬리컬 좌/우 피치 오프셋 반영)")
        fig_ls = go.Figure()
        
        # --- 기본 치수 ---
        _Ds = st.session_state['D_s']          # Shell ID (mm)
        _Dm = st.session_state['D_mandrel']    # Mandrel OD (mm)
        _Dc = st.session_state['D_c']          # Coil CL dia (mm)
        _do2 = st.session_state['d_o']         # Tube OD (mm)
        _pmm = st.session_state['pitch']       # Pitch per tube (mm)
        _lmm = _pmm * N_p_val                  # Lead = pitch × N_p (mm)
        _st2 = st.session_state['shell_thick'] # Shell wall thickness (mm)
        
        # 2:1 Elliptical dish head depth = ID/4
        _hd = _Ds / 4.0
        
        # --- 좌표계 정의 ---
        # X: 반경 방향 (-Shell/2 ~ +Shell/2), 좌우 대칭
        # Y: 수직 방향 (0=하단 T/L → Shell_TT_Length=상단 T/L)
        # Coil 영역: Head 높이(_hd)에서 시작, Coil Height만큼 올라감
        _y_bot = 0.0                            # Shell 하단 T/L (tangent line)
        _y_top = Shell_TT_Length_mm             # Shell 상단 T/L
        _coil_y0 = _hd                          # Coil 하단 시작 (하단 Head 높이 이후)
        _coil_y1 = _coil_y0 + L_shell_mm       # Coil 상단 끝
        
        # --- [1] Shell 벽체 (좌/우 두겹 직사각형) ---
        # 좌측 벽
        fig_ls.add_trace(go.Scatter(
            x=[-_Ds/2-_st2, -_Ds/2, -_Ds/2, -_Ds/2-_st2, -_Ds/2-_st2],
            y=[_y_bot, _y_bot, _y_top, _y_top, _y_bot],
            fill='toself', fillcolor='rgba(120,144,156,0.35)', mode='lines',
            line=dict(color='#455A64', width=2), name=f'Shell Wall (t={_st2:.0f}mm)'))
        # 우측 벽
        fig_ls.add_trace(go.Scatter(
            x=[_Ds/2, _Ds/2+_st2, _Ds/2+_st2, _Ds/2, _Ds/2],
            y=[_y_bot, _y_bot, _y_top, _y_top, _y_bot],
            fill='toself', fillcolor='rgba(120,144,156,0.35)', mode='lines',
            line=dict(color='#455A64', width=2), showlegend=False))

        # --- [2] Dish Heads (2:1 Elliptical) ---
        _thh = np.linspace(0, np.pi, 100)
        # 하단 Head: T/L에서 아래로 볼록
        fig_ls.add_trace(go.Scatter(
            x=(_Ds/2*np.cos(_thh)).tolist(), 
            y=(_y_bot - _hd*np.sin(_thh)).tolist(),
            mode='lines', line=dict(color='#455A64', width=2.5), name='Dish Head (2:1 Ellip.)'))
        # 상단 Head: T/L에서 위로 볼록
        fig_ls.add_trace(go.Scatter(
            x=(_Ds/2*np.cos(_thh)).tolist(), 
            y=(_y_top + _hd*np.sin(_thh)).tolist(),
            mode='lines', line=dict(color='#455A64', width=2.5), showlegend=False))

        # --- [3] Mandrel (코일 영역 내 중앙에 위치) ---
        # Mandrel은 코일 전체 영역에 걸쳐 존재
        fig_ls.add_trace(go.Scatter(
            x=[-_Dm/2, _Dm/2, _Dm/2, -_Dm/2, -_Dm/2],
            y=[_coil_y0, _coil_y0, _coil_y1, _coil_y1, _coil_y0],
            fill='toself', fillcolor='rgba(158,158,158,0.20)', mode='lines',
            line=dict(color='#9E9E9E', width=1.5, dash='dot'), name=f'Mandrel Ø{_Dm:.0f}'))
        
        # Mandrel 해칭선 (회색 사선 패턴 효과)
        _hatch_step = max(30, _Dm / 5)
        _hatch_x, _hatch_y = [], []
        for _hy in np.arange(_coil_y0, _coil_y1, _hatch_step):
            _hatch_x.extend([-_Dm/2, _Dm/2, None])
            _hatch_y.extend([_hy, _hy + _hatch_step * 0.5, None])
        fig_ls.add_trace(go.Scatter(
            x=_hatch_x, y=_hatch_y, mode='lines',
            line=dict(color='rgba(158,158,158,0.15)', width=0.5), showlegend=False, hoverinfo='skip'))

        # --- [4] Tube 단면 — 헬리컬 좌/우 피치 오프셋 반영 ---
        # 종단면에서 코일을 자르면:
        # - 우측(+D_c/2): 각 tube의 z 위치 = z_off + k * Lead
        # - 좌측(-D_c/2): 각 tube의 z 위치 = z_off + Lead/2 + k * Lead (반 바퀴 회전 후)
        _tr2 = _do2 / 2.0
        _max_vis = 50  # 성능을 위해 한 쪽당 최대 표시 개수
        _ls_lc = ['#1565C0','#E65100','#2E7D32','#C62828','#6A1B9A','#4E342E','#00838F','#AD1457']
        _ls_fc = ['rgba(21,101,192,0.40)','rgba(230,81,0,0.40)','rgba(46,125,50,0.40)','rgba(198,40,40,0.40)',
                  'rgba(106,27,154,0.40)','rgba(78,52,46,0.40)','rgba(0,131,143,0.40)','rgba(173,20,87,0.40)']
        _trunc = False
        
        # 튜브 단면을 Plotly shapes (layout.shapes)로 그려서 scaleanchor와 무관하게 진원 유지
        _tube_shapes = []
        
        for _ti in range(N_p_val):
            _ci2 = _ti % len(_ls_lc)
            _z_off = _ti * _pmm  # 각 가닥의 시작 피치 오프셋
            _cr_count = 0
            _cl_count = 0
            
            for _k in range(int(np.ceil(Turns_per_Tube)) + 1):
                # 우측 단면 위치 (코일이 앞으로 올 때)
                _rz = _z_off + _k * _lmm
                if 0 <= _rz <= L_shell_mm and _cr_count < _max_vis:
                    _tube_shapes.append(dict(
                        type='circle',
                        x0=_Dc/2 - _tr2, x1=_Dc/2 + _tr2,
                        y0=_coil_y0 + _rz - _tr2, y1=_coil_y0 + _rz + _tr2,
                        fillcolor=_ls_fc[_ci2],
                        line=dict(color=_ls_lc[_ci2], width=1.5)))
                    _cr_count += 1
                elif _rz <= L_shell_mm:
                    _trunc = True
                
                # 좌측 단면 위치 (코일이 뒤로 갈 때 = Lead/2 오프셋)
                _lz = _z_off + _lmm / 2.0 + _k * _lmm
                if 0 <= _lz <= L_shell_mm and _cl_count < _max_vis:
                    _tube_shapes.append(dict(
                        type='circle',
                        x0=-_Dc/2 - _tr2, x1=-_Dc/2 + _tr2,
                        y0=_coil_y0 + _lz - _tr2, y1=_coil_y0 + _lz + _tr2,
                        fillcolor=_ls_fc[_ci2],
                        line=dict(color=_ls_lc[_ci2], width=1.5)))
                    _cl_count += 1
                elif _lz <= L_shell_mm:
                    _trunc = True
        
        # 범례를 위한 더미 마커 (shapes는 범례 미지원이므로)
        for _ti in range(N_p_val):
            _ci2 = _ti % len(_ls_lc)
            fig_ls.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=10, color=_ls_fc[_ci2], line=dict(color=_ls_lc[_ci2], width=2)),
                name=f'Tube #{_ti+1}'))
        
        if _trunc:
            st.caption("⚠️ 렌더링 성능을 위해 일부 튜브 단면이 생략되었습니다.")

        # --- [5] 노즐 ---
        _noz_len = max(_do2 * 3.0, 60)  # 노즐 길이 (최소 60mm 보장)
        _noz_w = max(_do2 * 1.0, 20)    # 노즐 폭
        
        # Shell inlet: 우측, 코일 하단 근처
        _sin_y = _coil_y0 + _pmm / 2
        fig_ls.add_trace(go.Scatter(
            x=[_Ds/2+_st2, _Ds/2+_st2+_noz_len, _Ds/2+_st2+_noz_len, _Ds/2+_st2],
            y=[_sin_y-_noz_w/2, _sin_y-_noz_w/2, _sin_y+_noz_w/2, _sin_y+_noz_w/2],
            fill='toself', fillcolor='rgba(33,150,243,0.25)', mode='lines',
            line=dict(color='#1976D2', width=2), name='Shell Nozzle'))
        
        # Shell outlet: 좌측, 코일 상단 근처
        _sout_y = _coil_y1 - _pmm / 2
        fig_ls.add_trace(go.Scatter(
            x=[-_Ds/2-_st2, -_Ds/2-_st2-_noz_len, -_Ds/2-_st2-_noz_len, -_Ds/2-_st2],
            y=[_sout_y-_noz_w/2, _sout_y-_noz_w/2, _sout_y+_noz_w/2, _sout_y+_noz_w/2],
            fill='toself', fillcolor='rgba(33,150,243,0.25)', mode='lines',
            line=dict(color='#1976D2', width=2), showlegend=False))
        
        # Tube inlet: 상단 Head 위
        fig_ls.add_trace(go.Scatter(
            x=[-_noz_w/2, _noz_w/2, _noz_w/2, -_noz_w/2, -_noz_w/2],
            y=[_y_top+_hd*0.3, _y_top+_hd*0.3, _y_top+_hd*0.3+_noz_len, _y_top+_hd*0.3+_noz_len, _y_top+_hd*0.3],
            fill='toself', fillcolor='rgba(229,57,53,0.25)', mode='lines',
            line=dict(color='#D32F2F', width=2), name='Tube Nozzle'))
        
        # Tube outlet: 하단 Head 아래
        fig_ls.add_trace(go.Scatter(
            x=[-_noz_w/2, _noz_w/2, _noz_w/2, -_noz_w/2, -_noz_w/2],
            y=[_y_bot-_hd*0.3, _y_bot-_hd*0.3, _y_bot-_hd*0.3-_noz_len, _y_bot-_hd*0.3-_noz_len, _y_bot-_hd*0.3],
            fill='toself', fillcolor='rgba(229,57,53,0.25)', mode='lines',
            line=dict(color='#D32F2F', width=2), showlegend=False))

        # --- [6] 치수선 (Dimension lines) ---
        _sh_ls = list(_tube_shapes)  # shapes 리스트에 Tube circles 추가
        _an_ls = []
        _dim_gap = max(_noz_len + 30, 80)

        # T/T Length (우측)
        _dx_r = _Ds/2 + _st2 + _dim_gap
        _sh_ls.extend([
            dict(type='line', x0=_dx_r, y0=_y_bot, x1=_dx_r, y1=_y_top, line=dict(color='#333', width=1.5)),
            dict(type='line', x0=_Ds/2+_st2+10, y0=_y_bot, x1=_dx_r+15, y1=_y_bot, line=dict(color='#333', width=0.5)),
            dict(type='line', x0=_Ds/2+_st2+10, y0=_y_top, x1=_dx_r+15, y1=_y_top, line=dict(color='#333', width=0.5))])
        _an_ls.append(dict(x=_dx_r+5, y=_y_top/2, text=f'<b>T/T = {Shell_TT_Length_mm:,.0f} mm</b>',
            showarrow=False, font=dict(size=11, color='#333'), bgcolor='rgba(255,255,255,0.95)', xshift=5, textangle=-90))

        # Coil Height (좌측)
        _dx_l = -(_Ds/2 + _st2 + _dim_gap)
        _sh_ls.extend([
            dict(type='line', x0=_dx_l, y0=_coil_y0, x1=_dx_l, y1=_coil_y1, line=dict(color='#E91E63', width=1.5)),
            dict(type='line', x0=-(_Ds/2+_st2+10), y0=_coil_y0, x1=_dx_l-15, y1=_coil_y0, line=dict(color='#E91E63', width=0.5)),
            dict(type='line', x0=-(_Ds/2+_st2+10), y0=_coil_y1, x1=_dx_l-15, y1=_coil_y1, line=dict(color='#E91E63', width=0.5))])
        _an_ls.append(dict(x=_dx_l-5, y=(_coil_y0+_coil_y1)/2, text=f'<b>Coil H = {L_shell_mm:,.0f} mm</b>',
            showarrow=False, font=dict(size=10, color='#E91E63'), bgcolor='rgba(255,255,255,0.95)', xshift=-5, textangle=-90))

        # Shell ID (상단 수평 치수)
        _dim_y_top = _y_top + _hd + _noz_len + 30
        _sh_ls.extend([
            dict(type='line', x0=-_Ds/2, y0=_dim_y_top, x1=_Ds/2, y1=_dim_y_top, line=dict(color='#455A64', width=1)),
            dict(type='line', x0=-_Ds/2, y0=_y_top+5, x1=-_Ds/2, y1=_dim_y_top+10, line=dict(color='#455A64', width=0.5)),
            dict(type='line', x0=_Ds/2, y0=_y_top+5, x1=_Ds/2, y1=_dim_y_top+10, line=dict(color='#455A64', width=0.5))])
        _an_ls.append(dict(x=0, y=_dim_y_top+5, text=f'<b>Shell ID = {_Ds:.0f} mm</b>',
            showarrow=False, font=dict(size=10, color='#455A64'), bgcolor='rgba(255,255,255,0.95)'))

        # Pitch 치수 (우측 튜브 사이, 적색-남색 계열의 pitch 화살표)
        # 우측 첫 2개 튜브 위치 찾기
        _right_positions = []
        for _ti in range(N_p_val):
            _z_off = _ti * _pmm
            for _k in range(int(np.ceil(Turns_per_Tube)) + 1):
                _rz = _z_off + _k * _lmm
                if 0 <= _rz <= L_shell_mm:
                    _right_positions.append(_coil_y0 + _rz)
                    break
        _right_positions.sort()
        if len(_right_positions) >= 2:
            _pz0 = _right_positions[0]
            _pz1 = _right_positions[1]
            _px = _Dc/2 + _tr2 + 20
            _sh_ls.extend([
                dict(type='line', x0=_px, y0=_pz0, x1=_px, y1=_pz1, line=dict(color='#FF6F00', width=1.5)),
                dict(type='line', x0=_px-8, y0=_pz0, x1=_px+8, y1=_pz0, line=dict(color='#FF6F00', width=0.8)),
                dict(type='line', x0=_px-8, y0=_pz1, x1=_px+8, y1=_pz1, line=dict(color='#FF6F00', width=0.8))])
            _an_ls.append(dict(x=_px+5, y=(_pz0+_pz1)/2, text=f'<b>p={_pmm:.1f}</b>',
                showarrow=False, font=dict(size=10, color='#FF6F00'), bgcolor='rgba(255,255,255,0.95)', xshift=5))

        # --- [7] 레이블 ---
        _an_ls.extend([
            # 노즐 레이블
            dict(x=_Ds/2+_st2+_noz_len/2, y=_sin_y, text='<b>Shell In →</b>', 
                 showarrow=False, font=dict(size=10, color='#1976D2'), bgcolor='rgba(255,255,255,0.9)', yshift=_noz_w/2+12),
            dict(x=-(_Ds/2+_st2+_noz_len/2), y=_sout_y, text='<b>← Shell Out</b>',
                 showarrow=False, font=dict(size=10, color='#1976D2'), bgcolor='rgba(255,255,255,0.9)', yshift=_noz_w/2+12),
            dict(x=0, y=_y_top+_hd*0.3+_noz_len+8, text='<b>Tube In ↓</b>', 
                 showarrow=False, font=dict(size=10, color='#D32F2F'), bgcolor='rgba(255,255,255,0.9)'),
            dict(x=0, y=_y_bot-_hd*0.3-_noz_len-8, text='<b>↑ Tube Out</b>', 
                 showarrow=False, font=dict(size=10, color='#D32F2F'), bgcolor='rgba(255,255,255,0.9)'),
            # 중심선 (CL)
            dict(x=0, y=(_coil_y0+_coil_y1)/2, text='<b>CL</b>', showarrow=False, 
                 font=dict(size=11, color='rgba(0,0,0,0.15)'), opacity=0.5),
            # Mandrel 레이블
            dict(x=0, y=_coil_y0 + 15, text=f'Mandrel Ø{_Dm:.0f}', showarrow=False,
                 font=dict(size=9, color='#9E9E9E'), bgcolor='rgba(255,255,255,0.8)'),
        ])
        
        # 중심선 (점선)
        _sh_ls.append(dict(type='line', x0=0, y0=_y_bot-_hd-_noz_len-20, x1=0, y1=_y_top+_hd+_noz_len+40,
                          line=dict(color='rgba(0,0,0,0.08)', width=1, dash='dashdot')))

        # --- 축 범위를 데이터에 딱 맞게 설정 ---
        _margin_x = max(_noz_len + 50, _dim_gap + 30)  # 좌우 치수선/노즐 여유
        _x_min = -(_Ds/2 + _st2 + _margin_x)
        _x_max = _Ds/2 + _st2 + _margin_x
        _margin_y_bot = _hd + _noz_len + 30  # 하단 Head + 노즐 + 여유
        _margin_y_top = _hd + _noz_len + 50  # 상단 Head + 노즐 + 치수선 여유
        _y_min = _y_bot - _margin_y_bot
        _y_max = _y_top + _margin_y_top
        
        # 종횡비에 기반한 동적 차트 높이 계산
        _data_width = _x_max - _x_min
        _data_height = _y_max - _y_min
        _aspect_ratio = _data_height / max(_data_width, 1.0)
        # 차트 가용 폭(px)을 추정 (Streamlit use_container_width 기준, margin 제외)
        _chart_usable_width_px = 700  # 대략적인 Plotly 가용 폭
        _chart_h = int(max(500, min(1600, _chart_usable_width_px * _aspect_ratio)))
        
        fig_ls.update_layout(
            xaxis=dict(scaleanchor='y', scaleratio=1, showgrid=False, zeroline=False, title='Width (mm)',
                       range=[_x_min, _x_max]),
            yaxis=dict(showgrid=False, zeroline=False, title='Height (mm)',
                       range=[_y_min, _y_max]),
            height=_chart_h, margin=dict(l=80, r=80, t=30, b=40),
            plot_bgcolor='white', shapes=_sh_ls, annotations=_an_ls,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255,255,255,0.9)')
        )
        st.plotly_chart(fig_ls, use_container_width=True)

    # ========== TAB 3: 코일 전개도 ==========
    with tab_uw:
        st.caption("🔍 헬리컬 코일 1턴을 평면에 펼친 전개도 — Lead Angle(α)과 전개 길이 관계")
        fig_uw = go.Figure()
        _circ = np.pi * st.session_state['D_c']
        _lead = Lead_m * 1000.0
        _lpt = Length_per_Turn * 1000.0
        _alpha = np.degrees(np.arctan2(_lead, _circ))

        # Right triangle: Base=circumference, Height=Lead, Hypotenuse=Length/Turn
        fig_uw.add_trace(go.Scatter(
            x=[0, _circ, 0, 0], y=[0, 0, _lead, 0],
            fill='toself', fillcolor='rgba(21,101,192,0.06)', mode='lines',
            line=dict(color='#90CAF9', width=1.5), name='1-Turn Envelope'))
        fig_uw.add_trace(go.Scatter(
            x=[0, _circ], y=[_lead, 0],
            mode='lines', line=dict(color='#D32F2F', width=3),
            name=f'Tube/Turn = {_lpt:,.1f} mm'))

        _an_uw = []
        _an_uw.append(dict(x=_circ/2, y=0, text=f'<b>πD_c = {_circ:,.1f} mm</b>',
            showarrow=False, font=dict(size=12, color='#1565C0'), yshift=-20))
        _an_uw.append(dict(x=0, y=_lead/2,
            text=f'<b>Lead = {_lead:,.1f}</b><br>(p×N_p = {st.session_state["pitch"]:.1f}×{N_p_val})',
            showarrow=False, font=dict(size=11, color='#2E7D32'), xshift=-10, textangle=-90))
        _an_uw.append(dict(x=_circ/2, y=_lead/2,
            text=f'<b>L/turn = {_lpt:,.1f} mm</b>',
            showarrow=False, font=dict(size=11, color='#D32F2F'),
            bgcolor='rgba(255,255,255,0.9)', borderpad=3, xshift=30))

        # Helix angle arc
        _arc_r = min(_circ, _lead) * 0.15
        _arc_th = np.linspace(0, np.radians(_alpha), 30)
        fig_uw.add_trace(go.Scatter(
            x=(_circ - _arc_r*np.cos(_arc_th)).tolist(),
            y=(_arc_r*np.sin(_arc_th)).tolist(),
            mode='lines', line=dict(color='#FF6F00', width=2), showlegend=False))
        _an_uw.append(dict(
            x=_circ - _arc_r*1.4*np.cos(np.radians(_alpha/2)),
            y=_arc_r*1.4*np.sin(np.radians(_alpha/2)),
            text=f'<b>α = {_alpha:.1f}°</b>',
            showarrow=False, font=dict(size=11, color='#FF6F00'),
            bgcolor='rgba(255,255,255,0.9)', borderpad=2))

        # Summary
        fig_uw.add_trace(go.Scatter(
            x=[_circ/2], y=[-_lead*0.2], mode='text',
            text=[f'Total Turns = {Turns_per_Tube:.1f}  |  Total Length/Tube = {Length_per_Tube:,.1f} m  |  N_p = {N_p_val} 가닥'],
            textfont=dict(size=12, color='#333'), showlegend=False))

        fig_uw.update_layout(
            xaxis=dict(scaleanchor='y', scaleratio=1, showgrid=True, gridcolor='rgba(0,0,0,0.05)',
                       zeroline=False, title='Circumference (mm)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', zeroline=False, title='Height (mm)'),
            height=500, margin=dict(l=60, r=40, t=30, b=40),
            plot_bgcolor='white', annotations=_an_uw,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255,255,255,0.9)'))
        st.plotly_chart(fig_uw, use_container_width=True)
else:
    st.info("2D 도면을 렌더링하려면 유효한 설계 파라미터가 필요합니다.")

# =========================================================
# [M] 8. 파라메트릭 감도 분석 (Parametric Sensitivity Analysis)
# =========================================================
st.markdown("---")
st.subheader("8. 파라메트릭 감도 분석 (Sensitivity Analysis)")
st.caption("주요 설계 변수 변경에 따른 U-value, 압력 손실, Shell 길이의 변화를 인터랙티브하게 탐색합니다.")

if not lmtd_error and d_i > 0:
    sens_var = st.radio("📈 분석할 변수 선택", ["Coil Dia. (D_c)", "Parallel Tubes (N_p)", "Pitch (p)"], horizontal=True)
    
    if sens_var == "Coil Dia. (D_c)":
        base_val = st.session_state['D_c']
        sweep_vals = np.linspace(max(st.session_state['d_o'] * 10, base_val * 0.5), base_val * 2.0, 30)
        x_label = "Coil Diameter D_c (mm)"
    elif sens_var == "Parallel Tubes (N_p)":
        base_val = st.session_state['N_p']
        sweep_vals = np.arange(1, min(20, base_val * 3 + 1), 1)
        x_label = "Parallel Tubes N_p"
    else:
        base_val = st.session_state['pitch']
        sweep_vals = np.linspace(st.session_state['d_o'], st.session_state['d_o'] * 3.0, 30)
        x_label = "Pitch (mm)"
    
    sens_U, sens_dpT, sens_dpS, sens_LTT = [], [], [], []
    
    for sv in sweep_vals:
        # Override the sweep variable, keep all others at current values
        if sens_var == "Coil Dia. (D_c)":
            _sv_Dc = sv
            _sv_Np = N_p_val
            _sv_pitch = st.session_state['pitch']
        elif sens_var == "Parallel Tubes (N_p)":
            _sv_Dc = st.session_state['D_c']
            _sv_Np = max(1, int(sv))
            _sv_pitch = st.session_state['pitch']
        else:
            _sv_Dc = st.session_state['D_c']
            _sv_Np = N_p_val
            _sv_pitch = sv
        
        _sv_Dc_m = _sv_Dc / 1000.0
        _sv_p_m = _sv_pitch / 1000.0
        _sv_Lead_m = _sv_p_m * _sv_Np
        _sv_cr = d_i / _sv_Dc if _sv_Dc > 0 else 0
        
        # Tube side
        _sv_m_per_tube = (m_t / 3600.0) / max(1, _sv_Np)
        _sv_v_tube = _sv_m_per_tube / (st.session_state['t_rho'] * A_c)
        _sv_Re = (st.session_state['t_rho'] * _sv_v_tube * d_i_m) / max(EPSILON, t_mu_pa) if "Liquid" in st.session_state['fluid_type'] else Re
        _sv_tube = calc_tube_side(_sv_Re, Pr, _sv_cr, d_i_m, st.session_state['t_k'], _sv_v_tube, st.session_state['t_rho'], 1.0, st.session_state['d_o'], d_i)
        
        # Shell side
        _sv_LpT = np.sqrt((np.pi * _sv_Dc_m)**2 + _sv_Lead_m**2) if _sv_Dc_m > 0 else 1.0
        _sv_A_blocked = _sv_Np * A_tube_cross * safe_div(_sv_LpT, _sv_Lead_m)
        _sv_A_free = max(A_annulus * 0.1, A_annulus - _sv_A_blocked)
        _sv_v_shell = safe_div(m_cold_kg_s, st.session_state['s_rho'] * _sv_A_free) if _sv_A_free > 0 else 0.0
        _sv_Re_shell = (st.session_state['s_rho'] * _sv_v_shell * D_e_shell) / max(EPSILON, s_mu_pa)
        _sv_pr = _sv_pitch / max(EPSILON, st.session_state['d_o'])
        _sv_shell = calc_shell_side(_sv_Re_shell, Pr_shell, st.session_state['s_k'], d_o_m, _sv_pr)
        
        # U
        _sv_u = calc_overall_U(_sv_tube['h_i'], _sv_shell['h_o'], st.session_state['R_fi'], st.session_state['R_fo'], R_wall, st.session_state['d_o'], d_i)
        sens_U.append(_sv_u['U'])
        
        # Area → Length → dp
        _sv_Area = (Q_kW * 1000.0) / (_sv_u['U'] * LMTD)
        _sv_Area_d = _sv_Area * (1.0 + st.session_state['overdesign_pct'] / 100.0)
        _sv_TL = safe_div(_sv_Area_d, np.pi * d_o_m)
        _sv_LpTube = _sv_TL / _sv_Np
        _sv_Turns = _sv_LpTube / _sv_LpT
        
        _sv_tube_dp = calc_tube_side(_sv_Re, Pr, _sv_cr, d_i_m, st.session_state['t_k'], _sv_v_tube, st.session_state['t_rho'], _sv_LpTube, st.session_state['d_o'], d_i)
        sens_dpT.append(_sv_tube_dp['dp_bar'])
        
        _sv_L_shell = _sv_Turns * _sv_Lead_m
        # Shell ΔP: Zukauskas tube bank crossflow (감도 분석용)
        _sv_N_rows = max(1, _sv_Turns * _sv_Np)
        if _sv_Re_shell < SHELL_RE_TRANSITION:
            _sv_Eu = 10.0 / max(_sv_Re_shell, 1.0)**0.5
        else:
            _sv_Eu = 1.0 / max(_sv_Re_shell, 1.0)**0.2
        _sv_dp_shell = (_sv_Eu * _sv_N_rows * st.session_state['s_rho'] * _sv_v_shell**2 / 2.0) / 100000.0
        sens_dpS.append(_sv_dp_shell)
        
        _sv_LTT = (_sv_L_shell + D_s_m / 2.0) * 1000.0  # 2:1 Ellip. Head depth = D_s/4 × 2
        sens_LTT.append(_sv_LTT)
    
    # Create multi-axis sensitivity chart
    fig_sens = make_subplots(rows=2, cols=2,
                             subplot_titles=('U-value (W/m²K)', 'Tube ΔP (bar)', 'Shell ΔP (bar)', 'Shell T/T Length (mm)'),
                             horizontal_spacing=0.14, vertical_spacing=0.18)
    
    x_arr = sweep_vals.tolist() if isinstance(sweep_vals, np.ndarray) else [float(v) for v in sweep_vals]
    
    fig_sens.add_trace(go.Scatter(x=x_arr, y=sens_U, mode='lines+markers', 
                                   line=dict(color='#1565C0', width=2), marker=dict(size=4),
                                   name='U-value'), row=1, col=1)
    # Current value marker
    if sens_var == "Coil Dia. (D_c)":
        cv = st.session_state['D_c']
    elif sens_var == "Parallel Tubes (N_p)":
        cv = st.session_state['N_p']
    else:
        cv = st.session_state['pitch']
    fig_sens.add_trace(go.Scatter(x=[cv], y=[U_calc], mode='markers',
                                   marker=dict(size=12, color='red', symbol='star'),
                                   name='현재 설계점', showlegend=True), row=1, col=1)
    
    fig_sens.add_trace(go.Scatter(x=x_arr, y=sens_dpT, mode='lines+markers',
                                   line=dict(color='#E65100', width=2), marker=dict(size=4),
                                   name='Tube ΔP'), row=1, col=2)
    # Allowable line
    fig_sens.add_hline(y=st.session_state['allowable_dp_tube'], line_dash='dash', line_color='red',
                       annotation_text=f'Allow={st.session_state["allowable_dp_tube"]}', row=1, col=2)
    
    fig_sens.add_trace(go.Scatter(x=x_arr, y=sens_dpS, mode='lines+markers',
                                   line=dict(color='#2E7D32', width=2), marker=dict(size=4),
                                   name='Shell ΔP'), row=2, col=1)
    fig_sens.add_hline(y=st.session_state['allowable_dp_shell'], line_dash='dash', line_color='red',
                       annotation_text=f'Allow={st.session_state["allowable_dp_shell"]}', row=2, col=1)
    
    fig_sens.add_trace(go.Scatter(x=x_arr, y=sens_LTT, mode='lines+markers',
                                   line=dict(color='#6A1B9A', width=2), marker=dict(size=4),
                                   name='T/T Length'), row=2, col=2)
    fig_sens.add_hline(y=MAX_EQUIPMENT_LENGTH_M * 1000, line_dash='dash', line_color='red',
                       annotation_text=f'Max={MAX_EQUIPMENT_LENGTH_M*1000:.0f}mm', row=2, col=2)
    
    # Y축 개별 제목
    fig_sens.update_yaxes(title_text='W/m²K', showgrid=True, gridcolor='rgba(0,0,0,0.06)', row=1, col=1)
    fig_sens.update_yaxes(title_text='bar', showgrid=True, gridcolor='rgba(0,0,0,0.06)', row=1, col=2)
    fig_sens.update_yaxes(title_text='bar', showgrid=True, gridcolor='rgba(0,0,0,0.06)', row=2, col=1)
    fig_sens.update_yaxes(title_text='mm', showgrid=True, gridcolor='rgba(0,0,0,0.06)', row=2, col=2)

    # X축: 하단 row에만 라벨 표시하여 중첩 방지
    fig_sens.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.06)', row=1, col=1)
    fig_sens.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.06)', row=1, col=2)
    fig_sens.update_xaxes(title_text=x_label, showgrid=True, gridcolor='rgba(0,0,0,0.06)', row=2, col=1)
    fig_sens.update_xaxes(title_text=x_label, showgrid=True, gridcolor='rgba(0,0,0,0.06)', row=2, col=2)

    fig_sens.update_layout(
        height=850, showlegend=True, plot_bgcolor='white',
        margin=dict(t=60, b=60, l=60, r=30),
        legend=dict(yanchor="bottom", y=-0.12, xanchor="center", x=0.5, orientation='h',
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1)
    )
    
    st.plotly_chart(fig_sens, use_container_width=True)
    st.caption("⭐ 빨간 별표 = 현재 설계점 | 빨간 점선 = 허용 한계")
else:
    st.info("감도 분석을 수행하려면 유효한 설계 파라미터가 필요합니다.")