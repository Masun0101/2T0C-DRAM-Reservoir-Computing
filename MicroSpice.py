"""
MicroSpice V1.2 - Reservoir Computing Edition
---------------------------------------------
Changelog:
- Added v_init_map support in solve_transient for state continuity.
- Enhanced Jacobian Regularization and Newton Damping.
- Optimized for repetitive calling in RC loops.
"""

import numpy as np


# ==========================================
# 1. 电路定义类 (Circuit Definition)
# ==========================================
class Circuit:
    def __init__(self):
        self.nodes = set([0])  # 0 is always Ground
        self.components = []
        self.node_map = {}  # Map user node ID to matrix row index
        self.comp_by_name = {}  # Map component name to index

    def _add_component(self, name, comp_dict):
        if name in self.comp_by_name:
            raise ValueError(f"Component name '{name}' already exists!")
        comp_dict['name'] = name
        self.components.append(comp_dict)
        self.comp_by_name[name] = len(self.components) - 1

    def add_resistor(self, name, node1, node2, value):
        self._add_component(name, {'type': 'R', 'n1': node1, 'n2': node2, 'value': value})
        self.nodes.update([node1, node2])

    def add_capacitor(self, name, node1, node2, value):
        self._add_component(name, {'type': 'C', 'n1': node1, 'n2': node2, 'value': value})
        self.nodes.update([node1, node2])

    def add_variable_capacitor(self, name, node1, node2, c_v_func):
        self._add_component(name, {'type': 'VC', 'n1': node1, 'n2': node2, 'c_func': c_v_func})
        self.nodes.update([node1, node2])

    def add_voltage_source(self, name, node_p, node_n, func_v_t=None, dc_value=None):
        if func_v_t is None and dc_value is not None:
            func_v_t = lambda t: dc_value
        self._add_component(name, {
            'type': 'V', 'n1': node_p, 'n2': node_n,
            'func': func_v_t, 'dc_value': dc_value
        })
        self.nodes.update([node_p, node_n])

    def add_nmos(self, name, drain, gate, source, model_func):
        self._add_component(name, {
            'type': 'MOS', 'd': drain, 'g': gate, 's': source,
            'model': model_func
        })
        self.nodes.update([drain, gate, source])

    def build_matrix_structure(self):
        sorted_nodes = sorted(list(self.nodes))
        if 0 in sorted_nodes: sorted_nodes.remove(0)
        self.node_map = {node: i for i, node in enumerate(sorted_nodes)}
        self.dim_nodes = len(sorted_nodes)
        self.v_sources = [c for c in self.components if c['type'] == 'V']
        self.dim_vs = len(self.v_sources)
        return self.dim_nodes + self.dim_vs


# ==========================================
# 2. 核心求解逻辑 (Solver Core V1.2)
# ==========================================
def _solve_step(circuit, dim, t, x_guess, v_prev_map, dt, is_dc=False):
    x = x_guess.copy()
    G_MIN = 1e-15  # Regularization
    V_LIMIT = 0.5  # Newton Damping
    MAX_ITER = 50

    def get_idx(node):
        return -1 if node == 0 else circuit.node_map[node]

    for iter_k in range(MAX_ITER):
        G = np.zeros((dim, dim))
        RHS = np.zeros(dim)

        # Jacobian Regularization
        for i in range(circuit.dim_nodes):
            G[i, i] += G_MIN

        # 1. Resistors
        for comp in [c for c in circuit.components if c['type'] == 'R']:
            n1, n2, g = get_idx(comp['n1']), get_idx(comp['n2']), 1.0 / comp['value']
            if n1 >= 0: G[n1, n1] += g
            if n2 >= 0: G[n2, n2] += g
            if n1 >= 0 and n2 >= 0: G[n1, n2] -= g; G[n2, n1] -= g

        # 2. Capacitors (Trap Rule / Backward Euler Equivalent for RC)
        if not is_dc:
            for comp in [c for c in circuit.components if c['type'] in ['C', 'VC']]:
                n1, n2 = get_idx(comp['n1']), get_idx(comp['n2'])
                v1_old = v_prev_map.get(comp['n1'], 0.0)
                v2_old = v_prev_map.get(comp['n2'], 0.0)
                v_bias_old = v1_old - v2_old

                C_val = comp['value'] if comp['type'] == 'C' else comp['c_func'](v_bias_old)
                g_eq = C_val / dt
                i_eq = g_eq * v_bias_old

                if n1 >= 0: G[n1, n1] += g_eq; RHS[n1] += i_eq
                if n2 >= 0: G[n2, n2] += g_eq; RHS[n2] -= i_eq
                if n1 >= 0 and n2 >= 0: G[n1, n2] -= g_eq; G[n2, n1] -= g_eq

        # 3. Voltage Sources
        for i, comp in enumerate(circuit.v_sources):
            n1, n2 = get_idx(comp['n1']), get_idx(comp['n2'])
            idx_vs = circuit.dim_nodes + i
            val = comp['func'](t)
            if n1 >= 0: G[idx_vs, n1] = 1; G[n1, idx_vs] = 1
            if n2 >= 0: G[idx_vs, n2] = -1; G[n2, idx_vs] = -1
            RHS[idx_vs] = val

        # 4. NMOS
        for comp in [c for c in circuit.components if c['type'] == 'MOS']:
            d, g, s = get_idx(comp['d']), get_idx(comp['g']), get_idx(comp['s'])
            vd, vg, vs = (x[i] if i >= 0 else 0 for i in (d, g, s))
            ids, gg, gd, gs = comp['model'](vg, vd, vs)
            i_eq_rhs = ids - (gg * vg + gd * vd + gs * vs)
            if d >= 0:
                RHS[d] -= i_eq_rhs
                G[d, d] += gd
                if g >= 0: G[d, g] += gg
                if s >= 0: G[d, s] += gs
            if s >= 0:
                RHS[s] += i_eq_rhs
                G[s, s] -= gs  # Corrected for MNA
                if d >= 0: G[s, d] -= gd
                if g >= 0: G[s, g] -= gg

        try:
            x_target = np.linalg.solve(G, RHS)
        except np.linalg.LinAlgError:
            return None

        delta = x_target - x
        max_delta = np.max(np.abs(delta))
        x_new = x + delta * (V_LIMIT / max_delta) if max_delta > V_LIMIT else x_target

        if np.linalg.norm(x_new - x) < 1e-6: return x_new
        x = x_new
    return x


# ==========================================
# 3. 分析功能 (Modified for State Continuity)
# ==========================================
def solve_transient(circuit, t_start, t_stop, dt, v_init_map=None):
    dim = circuit.build_matrix_structure()
    time_steps = np.arange(t_start, t_stop, dt)
    results = np.zeros((len(time_steps), dim))
    x = np.zeros(dim)

    # Initialize node-to-voltage map
    v_prev_map = {node: 0.0 for node in circuit.nodes}
    if v_init_map is not None:
        v_prev_map.update(v_init_map)
        # Pre-fill guess vector to help convergence
        for node, val in v_init_map.items():
            if node in circuit.node_map:
                x[circuit.node_map[node]] = val

    for ti, t in enumerate(time_steps):
        x = _solve_step(circuit, dim, t, x, v_prev_map, dt)
        if x is None: break
        results[ti, :] = x
        # Update map for next time step
        for node, idx in circuit.node_map.items():
            v_prev_map[node] = x[idx]

    return time_steps, results


# ==========================================
# 4. 数据提取
# ==========================================
def get_voltage(circuit, results, node_id):
    if node_id == 0: return np.zeros(results.shape[0])
    idx = circuit.node_map.get(node_id)
    return results[:, idx] if idx is not None else None


def get_current(circuit, results, comp_name, x_axis_vals):
    comp = circuit.components[circuit.comp_by_name[comp_name]]
    n1, n2 = comp.get('n1', comp.get('d')), comp.get('n2', comp.get('s'))
    v1, v2 = get_voltage(circuit, results, n1), get_voltage(circuit, results, n2)
    v_diff = v1 - v2
    if comp['type'] == 'R': return v_diff / comp['value']
    if comp['type'] == 'MOS':
        vg = get_voltage(circuit, results, comp['g'])
        vs = get_voltage(circuit, results, comp['s'])
        vd = get_voltage(circuit, results, comp['d'])
        return np.array([comp['model'](vg[i], vd[i], vs[i])[0] for i in range(len(v1))])
    return None