import streamlit as st
import numpy as np
import plotly.graph_objects as go
import json

from calculator import (
    INIT_STATE,
    apply_loaded_data,
    calculate_design,
    find_optimal_geometry,
    format_datasheet_markdown,
    format_html_report,
)

st.set_page_config(page_title="Helical Tube Heat Exchanger Designer", layout="wide")
st.title("플랜트 공정 설계: Helical Tube Heat Exchanger 최적화")
st.markdown("---")

# =========================================================
# [A] 글로벌 상태(Session State) 초기화 (이전 코드 유지)
# =========================================================
init_state = INIT_STATE.copy()

for k, v in init_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

def apply_json():
    json_str = st.session_state['json_input_text']
    if not json_str.strip():
        st.warning("JSON 데이터를 입력하십시오.")
        return
    try:
        parsed_data = apply_loaded_data(json_str)
        for k, v in parsed_data.items():
            st.session_state[k] = v
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
est_m_cold = (Q_kW * 3600.0) / (s_cp_kJ * max(0.1, delta_T_shell)) if delta_T_shell > 0 else 0.0

if is_tube_heating:
    est_T_cold_out = s_in - ((Q_kW * 3600.0) / (max(0.1, m_s) * s_cp_kJ))
    op_mode = "🔥 Heater Mode"
else:
    est_T_cold_out = s_in + ((Q_kW * 3600.0) / (max(0.1, m_s) * s_cp_kJ))
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
    st.caption(f"💡 밀착 제작: **{min_pitch:.1f} mm** / TEMA 여유: **{min_pitch*1.25:.1f} mm**")
    
    mat_dict = {'Stainless Steel 316 (k=16)': 16.0, 'Titanium (k=22)': 22.0, 'Custom (직접 입력)': -1}
    mat_keys = list(mat_dict.keys())
    mat_vals = list(mat_dict.values())
    curr_k = st.session_state.get('tube_k_wall', 16.0)
    try: mat_idx = mat_vals.index(curr_k)
    except ValueError: mat_idx = len(mat_keys) - 1

    selected_mat = st.selectbox("Tube Material", mat_keys, index=mat_idx)
    st.session_state['tube_material'] = selected_mat
    if "Custom" in selected_mat:
        st.session_state['tube_k_wall'] = st.number_input("열전도도 입력", value=float(curr_k), min_value=0.01)
    else:
        st.session_state['tube_k_wall'] = mat_dict[selected_mat]

with col_g3:
    st.number_input("Mandrel OD (mm)", step=5.0, key='D_mandrel', help="Coil 내측 공간을 채워 Shell 유체의 바이패스를 막는 코어 기둥입니다.")
    rec_mandrel = max(10.0, st.session_state['D_c'] - st.session_state['d_o'] - 10.0)
    st.caption(f"💡 추천 최적값: **{rec_mandrel:.1f} mm** (Coil 내측 직경에서 조립/용접 여유 10mm 제외)")
    
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

    rec_Ds = st.session_state['D_c'] + st.session_state['d_o'] + 40.0
    st.caption(f"💡 추천 최소값: **{rec_Ds:.1f} mm** (Coil 외경에서 열팽창/조립 여유 40mm 확보)")

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

current_inputs = {k: st.session_state[k] for k in init_state.keys()}
calc_result = calculate_design(current_inputs)
calc = calc_result.values
for error in calc_result.errors:
    st.error(f"🚨 {error}")
for warning in calc_result.warnings:
    st.warning(f"⚠️ {warning}")

P_mpa = st.session_state['design_p_shell'] / 10.0
R_mm = st.session_state['D_s'] / 2.0
S_mpa = st.session_state['allow_s_shell']
E_eff = st.session_state['joint_e']
C_A = st.session_state['ca_shell']

d_i = calc.get('d_i', st.session_state['d_o'] - 2 * st.session_state['t_thick'])
t_req = calc.get('t_req', 0.0)
t_final = calc.get('t_final', st.session_state['shell_thick'])
st.session_state['shell_thick'] = t_final
shell_od = calc.get('shell_od', st.session_state['D_s'] + 2.0 * t_final)

if calc_result.is_valid:
    st.info(f"✓ 상업용 Shell Thickness: **{t_final:.0f} mm** 확정 (ASME 이론 두께: {t_req:.2f} mm)")
else:
    st.info("입력 검증 오류가 해소되면 두께/수력학 계산이 자동으로 갱신됩니다.")

# =========================================================
# [G] 백그라운드 수력학/열역학 코어 연산 (검증/순수 함수 기반)
# =========================================================
t_mu_pa = st.session_state.get('t_mu', 1.0) / 1000.0
s_mu_pa = st.session_state.get('s_mu', 1.0) / 1000.0
curvature_ratio = calc.get('curvature_ratio', 0.0)
m_hot_per_tube = calc.get('m_hot_per_tube', 0.0)
A_c = calc.get('A_c', 1e-6)
v_tube = calc.get('v_tube', 0.0)
Re = calc.get('Re', 0.0)
Pr = calc.get('Pr', 0.0)
De = calc.get('De', 0.0)
Re_crit = calc.get('Re_crit', 0.0)
f_c = calc.get('f_c', 0.0)
Nu_straight = calc.get('Nu_straight', 0.0)
Nu_calc = calc.get('Nu_calc', 0.0)
h_i = calc.get('h_i', 0.0)
m_cold_kg_s = calc.get('m_cold_kg_s', m_s / 3600.0)
D_s_m = calc.get('D_s_m', st.session_state['D_s'] / 1000.0)
D_man_m = calc.get('D_man_m', st.session_state['D_mandrel'] / 1000.0)
d_o_m = calc.get('d_o_m', st.session_state['d_o'] / 1000.0)
N_p_val = calc.get('N_p_val', max(1, st.session_state['N_p']))
p_m = calc.get('p_m', st.session_state['pitch'] / 1000.0)
D_c_m = calc.get('D_c_m', st.session_state['D_c'] / 1000.0)
Lead_m = calc.get('Lead_m', p_m * N_p_val)
Length_per_Turn = calc.get('Length_per_Turn', 1.0)
A_annulus = calc.get('A_annulus', 0.0)
A_tube_cross = (np.pi / 4.0) * (d_o_m**2)
A_blocked = calc.get('A_blocked', 0.0)
A_free_flow = calc.get('A_free_flow', 0.0)
v_shell = calc.get('v_shell', 0.0)
D_e_shell = calc.get('D_e_shell', 0.0)
Re_shell = calc.get('Re_shell', 0.0)
Pr_shell = calc.get('Pr_shell', 0.0)
Nu_shell = calc.get('Nu_shell', 0.0)
h_o = calc.get('h_o', 0.0)
pitch_ratio = calc.get('pitch_ratio', st.session_state['pitch'] / max(1e-6, st.session_state['d_o']))
penalty_factor = calc.get('penalty_factor', 1.0)
R_wall = calc.get('R_wall', 0.0)
U_calc = calc.get('U_calc', 0.0)
Area_req = calc.get('Area_req', 0.0)
Area_design = calc.get('Area_design', 0.0)
Total_Tube_Length = calc.get('Total_Tube_Length', 0.0)
Length_per_Tube = calc.get('Length_per_Tube', 0.0)
Turns_per_Tube = calc.get('Turns_per_Tube', 0.0)
dp_tube_bar = calc.get('dp_tube_bar', 0.0)
L_shell_m = calc.get('L_shell_m', 0.0)
L_shell_mm = calc.get('L_shell_mm', 0.0)
f_s = calc.get('f_s', 0.0)
dp_shell_bar = calc.get('dp_shell_bar', 0.0)
inner_clearance_rad = calc.get('inner_clearance_rad', ((st.session_state['D_c'] - st.session_state['d_o']) - st.session_state['D_mandrel']) / 2.0)
outer_clearance_rad = calc.get('outer_clearance_rad', (st.session_state['D_s'] - (st.session_state['D_c'] + st.session_state['d_o'])) / 2.0)

# =========================================================
# [H] AI 최적화 제안 (Optimizer) (이전 코드 유지)
# =========================================================
opt_results = find_optimal_geometry(current_inputs, calc) if calc_result.is_valid else {"opt_best_Dc": None, "opt_min_LTT": None, "opt_best_Dm": None, "opt_best_Ds": None}
opt_best_Dc = opt_results["opt_best_Dc"]
opt_min_LTT = opt_results["opt_min_LTT"]
opt_best_Dm = opt_results["opt_best_Dm"]
opt_best_Ds = opt_results["opt_best_Ds"]

# =========================================================
# [I] 실시간 Bounding Box 렌더링 (이전 코드 유지)
# =========================================================
Shell_TT_Length_m = calc.get('Shell_TT_Length_m', L_shell_m + (2.0 * D_s_m))
Shell_TT_Length_mm = calc.get('Shell_TT_Length_mm', Shell_TT_Length_m * 1000.0)
shell_od_m = shell_od / 1000.0

if "Vertical" in st.session_state['orientation']:
    Footprint_Area = calc.get('Footprint_Area', (np.pi / 4.0) * (shell_od_m ** 2))
else:
    Footprint_Area = calc.get('Footprint_Area', shell_od_m * Shell_TT_Length_m)

with bbox_placeholder.container():
    st.markdown("#### 📐 실시간 장비 예상 규격 (Estimated Bounding Box)")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Shell OD (외경)", f"{shell_od:,.1f} mm")
    b2.metric("Coiled Section Height", f"{L_shell_mm:,.0f} mm")
    if Shell_TT_Length_m > 10.0:
        b3.metric("🚨 Shell Length (T/T)", f"{Shell_TT_Length_mm:,.0f} mm", delta="과도한 길이! 배관 불가", delta_color="inverse")
    else:
        b3.metric("Shell Length (T/T)", f"{Shell_TT_Length_mm:,.0f} mm", delta="안정적 구조", delta_color="normal")
    b4.metric("장비 바닥 면적 (Footprint)", f"{Footprint_Area:,.2f} m²", help="설치 방향(Vertical/Horizontal)에 따라 달라집니다.")
    
    if opt_best_Dc:
        st.success(f"💡 **[AI 최적화 제안]** N_p={N_p_val}가닥 기준, 최소 Shell 길이({opt_min_LTT*1000:,.0f} mm)를 달성하는 최적 콤보: **Coil D_c = {opt_best_Dc:,.0f} mm** (이때 D_m={opt_best_Dm:,.0f}, D_s={opt_best_Ds:,.0f}, Pitch={st.session_state['d_o']*1.25:.1f})")
    st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# [J] 5. 상업용 데이터시트 검증 (Datasheet) (이전 코드 유지)
# =========================================================
st.markdown("---")
st.subheader("5. 열전달 및 수력학 검증 (Datasheet & Report)")
st.caption(f"Generated on: {calc.get('generated_at', 'N/A')}")

with st.expander("💡 설계 유속(Velocity) 가이드라인 및 판정 기준"):
    st.markdown("""
    | 유체 경로 | 권장 유속 범위 | 초과/미달 시 발생 문제 |
    | :--- | :--- | :--- |
    | **Tube 측 (액체)** | 1.0 ~ 2.5 m/s | **< 1.0:** 침전물/오염 유발 <br> **> 3.0:** Tube 침식(Erosion) 및 파열 |
    | **Shell 측 (액체)** | 0.3 ~ 1.0 m/s | **< 0.2:** 열전달 사각지대 발생 <br> **> 1.5:** 유체 유발 진동(FIV)으로 코일 파손 |
    """)

datasheet_md = format_datasheet_markdown(current_inputs, calc) if calc_result.is_valid else "설계 입력값을 수정하면 데이터시트가 생성됩니다."
st.markdown(datasheet_md)

html_report = format_html_report(current_inputs, calc, calc_result.warnings) if calc_result.is_valid else "<html><body><p>설계 입력값을 수정하면 보고서가 생성됩니다.</p></body></html>"

col_dl1, col_dl2 = st.columns([1, 2])
with col_dl1:
    st.download_button(label="📄 Datasheet 다운로드 (HTML/PDF용)", data=html_report, file_name=f"{st.session_state['tag_no']}_Datasheet.html", mime="text/html", disabled=not calc_result.is_valid)
with col_dl2:
    st.info("💡 폰트 에러 없는 PDF 출력을 위해 HTML로 내보냅니다. 브라우저 인쇄(Ctrl+P) 기능을 활용하세요.")

all_messages = calc_result.errors + calc_result.warnings
if all_messages:
    st.error("🚨 **Datasheet Review Required:** " + " / ".join(all_messages))
else:
    st.success("✅ **Datasheet Validated:** 모든 공정, 수력학, 기계적 제약 조건을 통과했습니다.")

# =========================================================
# [K] 6. 3D 형상 렌더링 (🌟 Real 3D Mesh Tube)
# =========================================================
st.markdown("---")
st.subheader("6. 3D 코일 형상 (Schematic Representation)")

if calc_result.is_valid and Turns_per_Tube > 0 and Turns_per_Tube < 2000 and d_i > 0:
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
    fig.add_trace(go.Scatter3d(x=[sh_in_r, sh_in_r + noz_h], y=[0, 0], z=[p_m*1000/2.0, p_m*1000/2.0], mode='lines', line=dict(color='blue', width=12), name='Shell Inlet'))
    fig.add_trace(go.Scatter3d(x=[-sh_in_r, -sh_in_r - noz_h], y=[0, 0], z=[coil_height - p_m*1000/2.0, coil_height - p_m*1000/2.0], mode='lines', line=dict(color='blue', width=12), name='Shell Outlet'))
    
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
