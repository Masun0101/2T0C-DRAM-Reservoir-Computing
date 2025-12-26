import numpy as np
import matplotlib.pyplot as plt
from MicroSpice import Circuit, solve_transient, get_voltage, get_current

"""
不考虑 WWL-SN，WBL-SN 耦合
"""

# ==========================================
# 1. 2T0C 物理器件层 (增加数据记录接口)
# ==========================================
class IGZO_2T0C_Array:
    def __init__(self, num_devices, sigma=0.1):
        self.n = num_devices
        np.random.seed(42)
        self.vth_w_list = np.random.normal(2.1, 2.1 * sigma, num_devices)
        self.c_store_list = np.random.normal(210e-15, 210e-15 * sigma, num_devices)
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
            return i_c, (calculate_current_core(vg + eps, vd, vs) - i_c) / eps, (
                        calculate_current_core(vg, vd + eps, vs) - i_c) / eps, (
                                    calculate_current_core(vg, vd, vs + eps) - i_c) / eps

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
        return i_c, (calculate_current_core(vg + eps, vd, vs) - i_c) / eps, (
                    calculate_current_core(vg, vd + eps, vs) - i_c) / eps, (
                                calculate_current_core(vg, vd, vs + eps) - i_c) / eps

    def update_and_read(self, u_in):
        read_currents = []
        T_write, T_leak = 0.5, 2.5
        T_total = T_write + T_leak

        for i in range(self.n):
            ckt = Circuit()
            v_wwl_write = 1.0 + 0.6 * u_in
            ckt.add_voltage_source('WWL', 2, 0, func_v_t=lambda t: v_wwl_write if t < T_write else 0.55)
            ckt.add_voltage_source('WBL', 1, 0, func_v_t=lambda t: 1.0 if t < T_write else 0.0)
            ckt.add_voltage_source('RBL', 4, 0, func_v_t=lambda t: 1.0)
            ckt.add_nmos('MW', 1, 2, 3, self._get_nmos_write_model(self.vth_w_list[i]))
            ckt.add_nmos('MR', 4, 3, 0, self._nmos_read_model)
            ckt.add_capacitor('C_store', 3, 0, self.c_store_list[i])

            # 瞬态仿真
            t_sim, res = solve_transient(ckt, 0, T_total, dt=0.05, v_init_map=self.device_states[i])

            # 记录数据用于调试可视化
            vsn_trace = get_voltage(ckt, res, 3)
            self.vsn_trace_all[i].extend(vsn_trace.tolist())
            if i == 0:
                self.time_trace_all.extend((t_sim + self.global_time_offset).tolist())

            # 状态更新
            final_v_sn = vsn_trace[-1]
            self.device_states[i] = {3: final_v_sn}
            final_i_read, _, _, _ = self._nmos_read_model(final_v_sn, 1.0, 0.0)
            read_currents.append(final_i_read)

        self.global_time_offset += T_total
        return np.array(read_currents)


# ==========================================
# 2. 储备池计算框架
# ==========================================
class T2C0Reservoir:
    def __init__(self, num_devices=20, virtual_nodes=50, sigma=0.1):
        self.num_devices = num_devices
        self.virtual_nodes = virtual_nodes
        self.devices = IGZO_2T0C_Array(num_devices, sigma=sigma)
        self.history = np.zeros((virtual_nodes, num_devices))

    def process_step(self, u_in):
        i_read_vec = self.devices.update_and_read(u_in)
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = i_read_vec
        return self.history.flatten()

# ==========================================
# 3. 任务执行与数据处理
# ==========================================
try:
    data_raw = np.load('mackey_glass_p5.npy')
except:
    data_raw = np.sin(np.linspace(0, 100, 1000))
    print('Warning: mackey_glass_3.npy not found, using placeholder.')

data = (data_raw - np.min(data_raw)) / (np.max(data_raw) - np.min(data_raw))

# 参数配置
NUM_DEVICES = 20
VIRTUAL_NODES = 50
warmup_steps = 50
train_steps = 200
test_steps = 20
T_step = 3.0  # T_write + T_leak

res_sys = T2C0Reservoir(NUM_DEVICES, VIRTUAL_NODES, sigma=0.1)

# 阶段 1: 训练
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

# 阶段 2: 测试
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
# 4. 全景可视化调试看板
# ==========================================
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
(ax_res, ax_in, ax_vsn) = axes

# --- 子图 1: 预测 vs 真实 (时域步长) ---
ax_res.plot(range(len(ground_truth)), ground_truth, 'b-o', label='Ground Truth', markersize=4)
ax_res.plot(range(len(predictions)), predictions, 'r--x', label='Prediction', markersize=4)
ax_res.set_title("Mackey-Glass Prediction Performance")
ax_res.set_ylabel("Normalized Value")
ax_res.legend()
ax_res.grid(True, alpha=0.3)

# --- 构建全局时间输入信号 ---
total_steps = warmup_steps + train_steps + test_steps
t_global_axis = np.array(res_sys.devices.time_trace_all)
u_global_trace = np.zeros_like(t_global_axis)
for s in range(total_steps):
    mask = (t_global_axis >= s * T_step) & (t_global_axis < (s + 1) * T_step)
    u_global_trace[mask] = data[s] if s < (warmup_steps + train_steps) else (
        predictions[s - (warmup_steps + train_steps)] if (s - (warmup_steps + train_steps)) < len(predictions) else 0)

# --- 子图 2: 全局输入序列 u(t) ---
ax_in.plot(t_global_axis, u_global_trace, 'k', label='Input u(t)')
ax_in.set_title("Input Signal Flow (Global Time)")
ax_in.set_ylabel("Input u")
ax_in.grid(True, alpha=0.3)

# --- 子图 3: 内部物理状态 Vsn (选取前 3 个器件) ---
for i in range(min(3, NUM_DEVICES)):
    ax_vsn.plot(t_global_axis, res_sys.devices.vsn_trace_all[i], label=f'Device {i} Vsn')

# 标注写入和泄露区域
for s in range(total_steps):
    ax_vsn.axvspan(s * T_step, s * T_step + 0.5, color='green', alpha=0.1)  # 写入区阴影
    ax_vsn.axvline(x=s * T_step, color='gray', linestyle=':', alpha=0.5)

ax_vsn.set_title("Internal State (Vsn) Dynamics [Green Shaded = Write Phase]")
ax_vsn.set_xlabel("Global Time (s)")
ax_vsn.set_ylabel("Vsn (V)")
ax_vsn.legend(loc='upper right')
ax_vsn.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()