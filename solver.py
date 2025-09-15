
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import MCPhaseGate, StatePreparation, QFTGate, IntegerComparator, ZGate
from qiskit.quantum_info import Operator
from qiskit_aer import Aer
from collections import Counter
from math import tau, ceil, log2, pi, sqrt, floor, sin
import random
import numpy as np

# =========================
# 0) 환경/파라미터
# =========================
random.seed(1234)
N = 4
SYMMETRIC = True
D = [[0]*N for _ in range(N)]
if SYMMETRIC:
    for i in range(N):
        for j in range(i+1, N):
            w = random.randint(5, 20)
            D[i][j] = w
            D[j][i] = w
else:
    for i in range(N):
        for j in range(N):
            if i == j: D[i][j] = 0
            else: D[i][j] = random.randint(5, 20)

Cmax_upper = 4 * max(max(row) for row in D)
S = 1 << ceil(log2(Cmax_upper + 1))
t = 5

# =========================
# 2) 유틸 및 게이트 생성 함수
# =========================
def print_gate_matrix(gate_name, gate):
    """게이트의 행렬 표현을 출력하는 함수"""
    try:
        matrix = Operator(gate).data
        print(f"--- Matrix for {gate_name} (size: {matrix.shape}) ---")
        print(matrix)
        print("-" * (len(gate_name) + 20))
    except Exception as e:
        print(f"Could not get matrix for {gate_name}: {e}")

def controls_for_value(bits, v):
    binv = [(v >> k) & 1 for k in range(len(bits))]
    prepared, need_flip = [], []
    for qb, want1 in zip(bits, binv):
        if want1 == 1: prepared.append(qb)
        else: need_flip.append(qb)
    return prepared, need_flip

def apply_x_on_list(qc, qubits):
    for q in qubits: qc.x(q)

def bits_from_label(v):
    assert v in (1,2,3)
    return (v & 1, (v >> 1) & 1)

def index_from_lsb_bits(bits_lsb):
    idx = 0
    for i, b in enumerate(bits_lsb): idx |= (b & 1) << i
    return idx

def get_general_qpe_gate(unitary_op, counting_anc_reg, state_qregs):
    """
    주어진 유니타리 연산자(unitary_op)에 대한
    범용 양자 위상 추정(QPE) 게이트를 생성합니다.
    """
    num_counting_qubits = len(counting_anc_reg)
    qpe_qc = QuantumCircuit(counting_anc_reg, *state_qregs, name="General_QPE")
    state_qubits = [q for reg in state_qregs for q in reg]
    
    # 1. 카운팅 큐비트에 H 게이트를 적용
    qpe_qc.h(counting_anc_reg)

    # 2. 제어-U 연산을 반복
    for k in range(num_counting_qubits):
        num_repeats = 2**k
        controlled_op = unitary_op.control(1)
        
        for _ in range(num_repeats):
            control_qubit = counting_anc_reg[k]
            qpe_qc.append(controlled_op, [control_qubit] + state_qubits)

    # 3. <<-- 여기가 최종 수정 사항 -->>
    #    QFTGate를 먼저 생성한 후, .inverse() 메서드를 호출하여 IQFT 게이트를 만듭니다.
    iqft_gate = QFTGate(num_counting_qubits).inverse()
    iqft_gate.name = "IQFT" # 게이트 이름 설정
    qpe_qc.append(iqft_gate, counting_anc_reg)

    return qpe_qc.to_gate()

def get_tour_preparation_gate(pos_qubits, unique_sym=False):
    tours = [(1,2,3),(1,3,2),(2,1,3),(2,3,1),(3,1,2),(3,2,1)]
    if unique_sym: tours = [(1,2,3),(1,3,2),(2,3,1)]
    amps = np.zeros(1 << len(pos_qubits), dtype=complex)
    for (a,b,c) in tours:
        a_bits,b_bits,c_bits = bits_from_label(a),bits_from_label(b),bits_from_label(c)
        bits_lsb = [a_bits[0],a_bits[1],b_bits[0],b_bits[1],c_bits[0],c_bits[1]]
        idx = index_from_lsb_bits(bits_lsb)
        amps[idx] = 1.0
    amps = amps / np.sqrt(len(tours))
    return StatePreparation(amps)

def U_cost_block(qc, pos0, pos1, pos2, pos3, ph, extra_ctrl=None, scale_factor=1):
    def add_edge(posA, posB):
        for i in range(N):
            for j in range(N):
                if i == j: continue
                angle = tau * D[i][j] * scale_factor / S
                ctrlA, undoA = controls_for_value(posA, i)
                ctrlB, undoB = controls_for_value(posB, j)
                apply_x_on_list(qc, undoA + undoB)
                all_controls = ctrlA + ctrlB
                if extra_ctrl is not None: all_controls.append(extra_ctrl)
                gate = MCPhaseGate(angle, num_ctrl_qubits=len(all_controls))
                qc.append(gate, all_controls + [ph[0]])
                apply_x_on_list(qc, undoA + undoB)
    add_edge(pos0, pos1); add_edge(pos1, pos2); add_edge(pos2, pos3); add_edge(pos3, pos0)

def get_qpe_gate(anc, qpe_registers):
    pos0, pos1, pos2, pos3, _, ph = qpe_registers
    qpe_qc = QuantumCircuit(*qpe_registers, name="QPE_Block")
    qpe_qc.h(anc)
    for k, a in enumerate(anc):
        scale = 2**k
        U_cost_block(qpe_qc, pos0, pos1, pos2, pos3, ph, extra_ctrl=a, scale_factor=scale)
    qpe_qc.append(QFTGate(len(anc), inverse=True), anc)
    return qpe_qc.decompose().to_gate(label="QPE")

def get_flexible_oracle_gate(num_tour_qubits, num_anc_qubits, L):
    """
    임의의 임계값 L을 기준으로 해답을 표식하는 유연한 오라클을 생성합니다.
    "측정된 비용(anc) < L" 이면 전체 상태의 위상을 바꿉니다.
    """
    # 1. IntegerComparator '회로'를 생성합니다.
    comparator_circuit = IntegerComparator(
        num_state_qubits=num_anc_qubits, 
        value=L, 
        geq=False
    )
    
    # <<-- 여기가 핵심 수정 사항 1 -->>
    # 2. 이 회로를 하나의 '게이트'로 변환합니다.
    comparator_gate = comparator_circuit.to_gate(label="cmp")

    num_comp_ancillas = comparator_circuit.num_ancillas

    tour_reg = QuantumRegister(num_tour_qubits, name='tour')
    anc_reg = QuantumRegister(num_anc_qubits, name='anc')
    result_qubit = QuantumRegister(1, name="res")
    comp_anc_reg = QuantumRegister(num_comp_ancillas, name='comp_anc')

    oracle = QuantumCircuit(tour_reg, anc_reg, result_qubit, comp_anc_reg, name=f"Oracle(L<{L})")

    qargs = anc_reg[:] + comp_anc_reg[:] + result_qubit[:]
    
    # <<-- 여기가 핵심 수정 사항 2 -->>
    # 3. 회로가 아닌 '게이트'를 append 합니다.
    oracle.append(comparator_gate, qargs)
    oracle.cz(result_qubit[0], tour_reg[0])
    # 게이트의 역행렬을 append 합니다.
    oracle.append(comparator_gate.inverse(), qargs)

    # 4. 이제 Oracle 회로는 내부 부품들이 모두 게이트이므로, 자신도 게이트로 변환될 수 있습니다.
    return oracle.to_gate()

def get_precise_diffuser_gate(state_preparation_gate):
    """
    주어진 StatePreparation 게이트(A)를 사용하여
    그 초기 상태 |S⟩를 정확히 반사점으로 삼는 정밀한 Diffuser 게이트를 생성합니다.
    """
    num_qubits = state_preparation_gate.num_qubits
    
    A_dagger = state_preparation_gate.inverse()
    A_dagger.name = "A_dg"

    # 1. |0...0> 상태에 대한 반사 '회로'를 생성합니다.
    reflection_circuit = QuantumCircuit(num_qubits, name="Ref_0")
    reflection_circuit.x(range(num_qubits))
    reflection_circuit.append(ZGate().control(num_qubits - 1), range(num_qubits))
    reflection_circuit.x(range(num_qubits))

    # <<-- 여기가 핵심 수정 사항 -->>
    # 2. 이 반사 회로를 하나의 '게이트'로 변환합니다.
    reflection_gate = reflection_circuit.to_gate()

    # 3. 이제 모든 부품(A_dagger, reflection_gate, state_preparation_gate)이
    #    게이트이므로, 이것들을 조립한 diffuser 회로도 게이트로 변환될 수 있습니다.
    diffuser = QuantumCircuit(num_qubits, name="PreciseDiffuser")
    diffuser.append(A_dagger, range(num_qubits))
    diffuser.append(reflection_gate, range(num_qubits))
    diffuser.append(state_preparation_gate, range(num_qubits))
    
    return diffuser.to_gate()

def run_qca_with_majority(backend, qca_qc, *, repeats=5, shots=2048, num_counting_qubits=6, N_total=6):
    """
    QCA 회로를 여러 번 실행하고, 모든 측정 결과를 누적하여 가장 많이 나온 결과를 선택합니다.
    """
    tqc = transpile(qca_qc, backend)
    total_counts = Counter()

    # 'repeats' 만큼 실행하여 모든 측정 결과를 'total_counts'에 누적합니다.
    for _ in range(repeats):
        result = backend.run(tqc, shots=shots).result()
        counts = result.get_counts()
        total_counts.update(counts)

    # 누적된 결과에서 가장 많이 나온 값을 최종 승자로 선택합니다.
    majority_winner = total_counts.most_common(1)[0][0]
    
    # 최종 승자(binary string)를 기반으로 M 값을 추정합니다.
    measured_int = int(majority_winner, 2)
    phi = measured_int / (2**num_counting_qubits)
    if phi > 0.5:
        phi = 1 - phi
    theta = 2 * pi * phi
    # M의 추정치는 sin^2(theta/2)에 비례합니다. 0이 되는 것을 방지하기 위해 max(1, ...) 사용
    M = max(1, round(N_total * (sin(theta / 2)**2)))

    # winners_per_run 대신 전체 누적 결과를 반환합니다.
    return majority_winner, phi, theta, M, total_counts
# ======================================================================

# =========================
# 3. 메인 실행 블록
# =========================
if __name__ == "__main__":
    
    # --- 추가된 헬퍼 함수들 (이전과 동일) ---
    def calculate_cost(D, path):
        cost = 0;
        for i in range(len(path) - 1): cost += D[path[i]][path[i+1]]
        return cost
    def binary_to_path(binary_str, N):
        bits = [int(b) for b in reversed(binary_str)]; a = (bits[1] << 1) | bits[0]; b = (bits[3] << 1) | bits[2]; c = (bits[5] << 1) | bits[4]
        return [0, a, b, c, 0] if N == 4 else None
    from itertools import permutations
    def calculate_classical_tsp(D):
        cities = list(range(1, N)); min_cost = float('inf'); best_path = None
        for p in permutations(cities):
            path = [0] + list(p) + [0]; cost = calculate_cost(D, path)
            if cost < min_cost: min_cost = cost; best_path = path
        return min_cost, best_path
    # --- 헬퍼 함수 끝 ---

    # --- 1. 초기 설정 및 큐비트 레지스터 정의 ---
    num_tour_qubits = (N - 1) * 2
    N_total = 6 # (N-1)! for N=4
    num_anc_qubits = t
    backend = Aer.get_backend('aer_simulator')
    
    tour_reg = QuantumRegister(num_tour_qubits, name='tour')
    anc_reg = QuantumRegister(num_anc_qubits, name='anc')
    res_qubit = QuantumRegister(1, name="res")
    temp_comp = IntegerComparator(num_anc_qubits, value=1)
    comp_anc_reg = QuantumRegister(temp_comp.num_ancillas, name='comp_anc')
    all_grover_qregs = [tour_reg, anc_reg, res_qubit, comp_anc_reg]
    
    print("="*40); print(f"Quantum TSP Solver for {N} cities"); print(f"Distance Matrix D:\n{np.array(D)}"); print(f"Scaling Factor (S): {S}, QPE Precision (t): {t}"); print("="*40)

    # --- 2. 정적(Static) 양자 게이트 생성 ---
    print("[Log] Creating static quantum gates...")
    state_prep_gate = get_tour_preparation_gate(tour_reg)
    diffuser_gate = get_precise_diffuser_gate(state_prep_gate)
    print("[Log] Static gates created successfully.\n")

    # --- 3. 완전 자동 최적화 루프 ---
    iteration = 1; current_threshold_L = float(S); best_cost_found = float(S); best_path_found = None
    
    print("="*40); print("🚀 Starting Automated Optimization Loop with QCA..."); print("="*40)

    while True:
        print(f"\n--- Iteration #{iteration} ---"); print(f"Searching for paths with cost < {current_threshold_L}")

        # 3.1) 오라클 및 그로버 연산자 게이트 생성
        oracle_gate = get_flexible_oracle_gate(
            num_tour_qubits, num_anc_qubits, int(current_threshold_L)
        )
        grover_op = QuantumCircuit(*all_grover_qregs, name="GroverOp")
        grover_op.append(oracle_gate, grover_op.qubits)
        grover_op.append(diffuser_gate, tour_reg)
        grover_op_gate = grover_op.to_gate()

        # 3.2) QCA 실행
        print("[Log] Running QCA to count solutions...")
        qca_anc_reg = QuantumRegister(num_anc_qubits, name='qca_anc')
        
        # <<-- 여기가 최종 수정 사항 1: QCA용 ClassicalRegister 추가 -->>
        qca_cr = ClassicalRegister(num_anc_qubits, name='qca_c')
        qca_qc = QuantumCircuit(qca_anc_reg, *all_grover_qregs, qca_cr)
        
        qca_qc.append(state_prep_gate, tour_reg)
        
        qca_gate = get_general_qpe_gate(
            unitary_op=grover_op_gate,
            counting_anc_reg=qca_anc_reg,
            state_qregs=all_grover_qregs
        )
        qca_qc.append(qca_gate, qca_qc.qubits)

        # <<-- 여기가 최종 수정 사항 2: qca_anc_reg만 정확히 측정 -->>
        qca_qc.measure(qca_anc_reg, qca_cr)
        
        _, _, _, M, _ = run_qca_with_majority(
            backend, qca_qc, repeats=5, shots=2048,
            num_counting_qubits=num_anc_qubits, N_total=N_total
        )
        print(f"  -> QCA Result: M ≈ {M}")

        if M == 0:
            print(f"\n[Log] QCA found no solutions. Assuming previous result is optimal.")
            if iteration == 1: best_path_found = "No solution found below S"
            break
            
        # 3.3) 그로버 탐색 실행
        k = floor(pi / 4 * sqrt(N_total / M))
        print(f"[Log] Running Grover's search with k = {k} iterations...")
        
        cr = ClassicalRegister(num_tour_qubits, name="c")
        search_qc = QuantumCircuit(*all_grover_qregs, cr)
        search_qc.append(state_prep_gate, tour_reg)
        for _ in range(k):
            search_qc.append(grover_op_gate, search_qc.qubits)
        search_qc.measure(tour_reg, cr)
        
        t_qc = transpile(search_qc, backend); result = backend.run(t_qc, shots=4096).result(); counts = result.get_counts()
        winner_binary = max(counts, key=counts.get)
        
        # 3.4) 임계값 갱신
        new_path = binary_to_path(winner_binary, N); C_new = calculate_cost(D, new_path)
        print(f"  -> Found new path: {new_path} with cost = {C_new}")

        if C_new >= best_cost_found:
            print(f"\n[Log] Found cost {C_new} is not better than best cost {best_cost_found}. Optimization finished.")
            break

        current_threshold_L = C_new; best_cost_found = C_new; best_path_found = new_path; iteration += 1

    # --- 4. 최종 결과 출력 ---
    print("\n" + "="*40); print("🏆 Optimization Complete! 🏆"); print("="*40)
    classical_cost, classical_path = calculate_classical_tsp(D)
    print(f"  - Classical Brute-force Path: {classical_path}"); print(f"  - Classical Brute-force Cost: {classical_cost}")
    print(f"  - Quantum Solver Found Path: {best_path_found}"); print(f"  - Quantum Solver Found Cost: {best_cost_found}")