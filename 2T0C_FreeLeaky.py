import numpy as np
import matplotlib.pyplot as plt
from MicroSpice import Circuit, solve_transient, get_voltage, get_current

"""
2T0C_FreeLeaky.py - 多泄露电压偏移 (Multi-Leakage Offset) 版本
每个器件拥有独立的泄露阶段 WWL 电压，实现异构动力学响应。
"""


# ==========================================
# 1. 2T0C 物理器件层 (异构泄露电压)
# ==========================================
class IGZO_2T0C_Array:
    def __init__(self, num_devices, v_leak_range=(0.6, 0.9), sigma=0.1):
        self.n = num_devices
        np.random.seed(42)

        # 器件参数差异性
        self.vth_w_list = np.random.normal(2.1, 2.1 * sigma, num_devices)
        self.c_store_list = np.random.normal(210e-15, 210e-15 * sigma, num_devices)

        # --- 核心修改：生成等间隔分布的泄露电压 ---
        # 例如 20个器件在 0.6V 到 0.9V 之间等间隔分布
        self.v_leak_list = np.linspace(v_leak_range[0], v_leak_range[1], num_devices)
        print(f"Device Leakage Voltages (WWL_leak): {self.v_leak_list}")

        # 状态保存字典
        self.device_states = [{3: 0.0} for _ in range(num_devices)]

        # 瞬态记录缓冲区
        self.vsn_trace_all = [[] for _ in range(num_devices)]
        self.time_trace_all = []
        self.global_time_offset = 0.0

    def _get_nmos_write_model(self, vth_val):
        def nmos_model(vg, vd, vs):
            Is_sq, W_L, Vth = 1.766e-09, 2.0, vth_val
            Is = Is_sq * W_L
            n, Vt, gamma, lam = 4.0, 0.026, 0.212, 0.0492
            I_leak_floor, v_factor = 1e-22, 2 * n * Vt

            def calculate_current_core(v_g, v_d, v_s):
                if v_d >= v_s:
                    v_d_p, v_s_p, sign = v_d, v_s, 1.0
                else:
                    v_d_p, v_s_p, sign = v_s, v_d, -1.0
                vgs_eff, vds_eff = v_g - v_s_p, v_d_p - v_s_p

                def ekv_interp(x):
                    if x > 50:
                        return x * x
                    elif x < -50:
                        return np.exp(x)
                    return np.log(1 + np.exp(x)) ** 2

                xf, xr = (vgs_eff - Vth) / v_factor, (vgs_eff - Vth - vds_eff) / v_factor
                Ff, Fr = ekv_interp(xf), ekv_interp(xr)
                mob = (1 + np.sqrt(Ff) * v_factor) ** gamma
                return Is * (Ff - Fr) * mob * (1 + lam * vds_eff) * sign + I_leak_floor

            eps = 1e-5
            i_c = calculate_current_core(vg, vd, vs)
            return i_c, (calculate_current_core(vg + eps, vd, vs) - i_c) / eps, \
                        (calculate_current_core(vg, vd + eps, vs) - i_c) / eps, \
                        (calculate_current_core(vg, vd, vs + eps) - i_c) / eps

        return nmos_model

    def _nmos_read_model(self, vg, vd, vs):
        Is_sq, W_L, Vth, n, Vt, gamma, lam = 1e-7, 2.0, 2.0, 15.0, 0.026, 0.1, 0.002
        Is = Is_sq * W_L
        v_factor = 2 * n * Vt

        def calculate_current_core(v_g, v_d, v_s):
            if v_d >= v_s:
                v_d_p, v_s_p, sign = v_d, v_s, 1.0
            else:
                v_d_p, v_s_p, sign = v_s, v_d, -1.0
            vgs_eff, vds_eff = v_g - v_s_p, v_d_p - v_s_p

            def ekv_interp(x):
                if x > 50:
                    return x * x
                elif x < -50:
                    return np.exp(x)
                return np.log(1 + np.exp(x)) ** 2

            xf, xr = (vgs_eff - Vth) / v_factor, (vgs_eff - Vth - vds_eff) / v_factor
            Ff, Fr = ekv_interp(xf), ekv_interp(xr)
            mob = (1 + np.sqrt(Ff) * v_factor) ** gamma
            return Is * (Ff - Fr) * mob * (1 + lam * vds_eff) * sign + 1e-22

        eps = 1e-5
        i_c = calculate_current_core(vg, vd, vs)
        return i_c, (calculate_current_core(vg + eps, vd, vs) - i_c) / eps, \
                    (calculate_current_core(vg, vd + eps, vs) - i_c) / eps, \
                    (calculate_current_core(vg, vd, vs + eps) - i_c) / eps

    def update_and_read(self, u_in):
        read_currents = []
        T_write, T_leak = 0.5, 2.5
        T_total = T_write + T_leak

        for i in range(self.n):
            ckt = Circuit()
            v_write_val = 1.0 + 0.5 * u_in
            v_leak_val = self.v_leak_list[i]  # 获取该器件特定的泄露电压

            # 使用默认参数捕获 v_write_val 和 v_leak_val 确保 lambda 逻辑正确
            ckt.add_voltage_source('WWL', 2, 0,
                                   func_v_t=lambda t, vw=v_write_val, vl=v_leak_val: vw if t < T_write else vl)
            ckt.add_voltage_source('WBL', 1, 0, func_v_t=lambda t: 1.0 if t < T_write else 0.0)
            ckt.add_voltage_source('RBL', 4, 0, func_v_t=lambda t: 1.0)

            ckt.add_nmos('MW', 1, 2, 3, self._get_nmos_write_model(self.vth_w_list[i]))
            ckt.add_nmos('MR', 4, 3, 0, self._nmos_read_model)

            ckt.add_capacitor('C_store', 3, 0, self.c_store_list[i])
            ckt.add_capacitor('C_wwl_sn', 2, 3, 46e-15)
            ckt.add_capacitor('C_rbl_sn', 4, 3, 70e-15)

            t_sim, res = solve_transient(ckt, 0, T_total, dt=0.01, v_init_map=self.device_states[i])

            vsn_trace = get_voltage(ckt, res, 3)
            self.vsn_trace_all[i].extend(vsn_trace.tolist())
            if i == 0:
                self.time_trace_all.extend((t_sim + self.global_time_offset).tolist())

            final_v_sn = vsn_trace[-1]
            # 更新状态字典，记录节点 2, 3, 4 的末态电压
            self.device_states[i] = {3: final_v_sn, 2: get_voltage(ckt, res, 2)[-1], 4: get_voltage(ckt, res, 4)[-1]}

            final_i_read, _, _, _ = self._nmos_read_model(final_v_sn, 1.0, 0.0)
            read_currents.append(final_i_read)

        self.global_time_offset += T_total
        return np.array(read_currents)


# ==========================================
# 2. 储备池计算框架
# ==========================================
class T2C0Reservoir:
    def __init__(self, num_devices=20, virtual_nodes=50, sigma=0.1, v_leak_range=(0.6, 0.9)):
        self.num_devices = num_devices
        self.virtual_nodes = virtual_nodes
        self.devices = IGZO_2T0C_Array(num_devices, v_leak_range=v_leak_range, sigma=sigma)
        self.history = np.zeros((virtual_nodes, num_devices))

    def process_step(self, u_in):
        i_read_vec = self.devices.update_and_read(u_in)
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = i_read_vec
        return self.history.flatten()


# ==========================================
# 3. 任务执行与预测逻辑 (MG序列)
# ==========================================
# (数据加载部分保持不变...)
try:
    data_raw = np.load(r'D:\PyCharm\Py_Projects\Spice1\Dataset\mackey_glass_1.npy')
except:
    data_raw = np.sin(np.linspace(0, 100, 1000))
    print('Warning: Data not found, using Sine.')

data = (data_raw - np.min(data_raw)) / (np.max(data_raw) - np.min(data_raw))

NUM_DEVICES = 20
VIRTUAL_NODES = 50
warmup_steps = 50
train_steps = 200
test_steps = 35
T_step = 3.0

# 可以在此处通过 v_leak_range 自由编辑泄露电压的起始和终止值
# res_sys = T2C0Reservoir(NUM_DEVICES, VIRTUAL_NODES, sigma=0.001, v_leak_range=(0.02, 1.7))
res_sys = T2C0Reservoir(NUM_DEVICES, VIRTUAL_NODES, sigma=0.1, v_leak_range=(0.02, 1.7))

print("--- Training ---")
X_train, Y_train = [], []
for i in range(warmup_steps + train_steps):
    state = res_sys.process_step(data[i])
    if i >= warmup_steps:
        X_train.append(state)
        Y_train.append(data[i + 1])
    if i % 10 == 0: print(f"Step {i}")

X_train, Y_train = np.array(X_train), np.array(Y_train)
W_out = np.linalg.pinv(X_train) @ Y_train

print("--- Testing ---")
predictions, ground_truth = [], []
last_u = data[warmup_steps + train_steps]
for i in range(test_steps):
    state = res_sys.process_step(last_u)
    pred = state @ W_out
    predictions.append(pred)
    ground_truth.append(data[warmup_steps + train_steps + i + 1])
    last_u = pred

# ==========================================
# 4. Visualization (IEEE Style)
# ==========================================

plt.rcParams.update({
    # ---- 字体 ----
    'font.family': 'Times New Roman',
    'font.size': 9,
    'font.weight': 'bold',

    # ---- 坐标轴 ----
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.2,

    # ---- 刻度 ----
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,

    # ---- 线条 ----
    'lines.linewidth': 1.0,

    # ---- 网格 ----
    'grid.linewidth': 0.5,
    'grid.linestyle': '--',
    'grid.alpha': 0.4,

    # ---- 导出 ----
    'savefig.dpi': 300,
    'figure.dpi': 300,
})

fig, axes = plt.subplots(
    2,
    1,
    figsize=(3.5, 5.2)   # IEEE single-column figure
)

ax_res, ax_vsn = axes

# -------------------------------------------------
# Subplot 1: Mackey–Glass Prediction Performance
# -------------------------------------------------
t_gt = range(len(ground_truth))
t_pr = range(len(predictions))

ax_res.plot(
    t_gt,
    ground_truth,
    color='black',
    linestyle='-',
    label='Ground Truth'
)

ax_res.plot(
    t_pr,
    predictions,
    color='dimgray',
    linestyle='--',
    label='Prediction'
)

ax_res.set_title("MSO Time Series Prediction")
ax_res.set_xlabel("Time Steps")
ax_res.set_ylabel("Value")

ax_res.legend(
    loc='upper left',
    frameon=False,
    fontsize=9
)

ax_res.grid(True)

# -------------------------------------------------
# Subplot 2: Heterogeneous Vsn Dynamics
# -------------------------------------------------
t_global = np.array(res_sys.devices.time_trace_all)

indices_to_show = [
    0,
    NUM_DEVICES - 1
]

line_colors = ['black', 'gray']

for idx, color in zip(indices_to_show, line_colors):
    v_leak = res_sys.devices.v_leak_list[idx]
    ax_vsn.plot(
        t_global,
        res_sys.devices.vsn_trace_all[idx],
        color=color,
        linewidth=1.6,
        label=f'Device {idx} (V_leak={v_leak:.2f} V)'
    )

# ---- Background time-step shading (IEEE-safe) ----
for s in range(warmup_steps + train_steps + test_steps):
    ax_vsn.axvspan(
        s * T_step,
        s * T_step + 0.5,
        color='gray',
        alpha=0.05
    )

# ======== 只显示前 200 s ========
ax_vsn.set_xlim(0, 200)

ax_vsn.set_title("$V_{sn}$ Dynamics")
ax_vsn.set_xlabel("Time (s)")
ax_vsn.set_ylabel("$V_{sn}$ (V)")

ax_vsn.legend(
    loc='lower right',
    frameon=False,
    fontsize=9
)

ax_vsn.grid(True)

plt.tight_layout()

# # ---- IEEE export ----
# plt.savefig(
#     "MG_MSO_Prediction_Vsn_IEEE.pdf",
#     bbox_inches='tight'
# )
# plt.savefig(
#     "MG_MSO_Prediction_Vsn_IEEE.png",
#     dpi=300,
#     bbox_inches='tight'
# )

plt.show()
