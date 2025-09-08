from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import MCPhaseGate, StatePreparation, QFT, IntegerComparator
from qiskit_aer import Aer
from math import tau, ceil, log2, pi, sqrt, floor, sin
import random
import numpy as np



# Global variables and parameters

random.seed(13) # fix the seed for reproducibility
N = 4 # number of cities
SYMMETRIC = True
D = [[0]*N for _ in range(N)]
if SYMMETRIC:
    for i in range(N):
        for j in range(i+1, N):
            w = random.randint(5, 20) # weight between 5 and 20
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



# Main quantum functions

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

def get_tour_preparation_gate(pos_qubits, unique_sym=True):
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
    qpe_qc.append(QFT(len(anc), do_swaps=True, inverse=True), anc)
    return qpe_qc.decompose().to_gate(label="QPE")

def get_flexible_oracle_gate(anc_qubits, L_threshold, S, comparator_ancilla):
    oracle_qc = QuantumCircuit(anc_qubits, comparator_ancilla, name='Oracle')
    num_anc = len(anc_qubits)
    L_k = floor(L_threshold * (2**num_anc) / S)
    if L_k >= 2**num_anc: L_k = 2**num_anc - 1
    if L_k <= 0: return oracle_qc.to_gate()
    
    comparator = IntegerComparator(num_state_qubits=num_anc, value=L_k, geq=False)
    oracle_qc.append(comparator, [*anc_qubits, *comparator_ancilla])
    result_qubit = comparator_ancilla[-1]
    oracle_qc.z(result_qubit)
    oracle_qc.append(comparator.inverse(), [*anc_qubits, *comparator_ancilla])
    return oracle_qc.decompose().to_gate()

def get_diffuser_gate(tour_qubits, prep_gate):
    diffuser_qc = QuantumCircuit(tour_qubits, name='Diffuser')
    diffuser_qc.append(prep_gate.inverse(), tour_qubits)
    apply_x_on_list(diffuser_qc, tour_qubits)
    diffuser_qc.h(tour_qubits[-1]); diffuser_qc.mcx(tour_qubits[:-1], tour_qubits[-1]); diffuser_qc.h(tour_qubits[-1])
    apply_x_on_list(diffuser_qc, tour_qubits)
    diffuser_qc.append(prep_gate, tour_qubits)
    return diffuser_qc.to_gate()



# Main execution block

if __name__ == "__main__":
    # register definitions
    pos0=QuantumRegister(2,'p0'); pos1=QuantumRegister(2,'p1')
    pos2=QuantumRegister(2,'p2'); pos3=QuantumRegister(2,'p3')
    anc=QuantumRegister(t,'anc'); ph=QuantumRegister(1,'ph')
    comp_anc = QuantumRegister(t, 'comp_anc')
    
    tour_qubits = [*pos1, *pos2, *pos3]
    qpe_registers = [pos0, pos1, pos2, pos3, anc, ph]
    problem_registers = [pos0, pos1, pos2, pos3, anc, ph, comp_anc]
    problem_qubits = [q for reg in problem_registers for q in reg]
    N_total = 6

    # static gate preparations
    print("[¡] Static gate preparations...")
    prep_gate = get_tour_preparation_gate(tour_qubits)
    qpe_gate = get_qpe_gate(anc, qpe_registers)
    iqpe_gate = qpe_gate.inverse(); iqpe_gate.label = "QPE^-1"
    diffuser_gate = get_diffuser_gate(tour_qubits, prep_gate)
    print("[✓] Static gate preparations complete.")

    # generate classical tour map for lookup
    tours = [(0,1,2,3,0),(0,1,3,2,0),(0,2,1,3,0),(0,2,3,1,0),(0,3,1,2,0),(0,3,2,1,0)]
    tour_map = {}
    min_cost_classical = float('inf')
    print("[¡] Classical tour costs")
    for tour in tours:
        cost = sum(D[tour[k]][tour[k+1]] for k in range(4))
        if cost < min_cost_classical: min_cost_classical = cost
        print(f"{tour} -> 비용: {cost}")
        p1, p2, p3 = tour[1], tour[2], tour[3]
        b1,b2,b3=bits_from_label(p1),bits_from_label(p2),bits_from_label(p3)
        bitstring = f"{b3[1]}{b3[0]}{b2[1]}{b2[0]}{b1[1]}{b1[0]}"
        tour_map[bitstring] = (tour, cost)
    print(f"[✓] Classical minimun cost: {min_cost_classical}")
    print("===== ===== ===== \n")

    # iterative optimization loop
    L_current_best = S
    optimal_tour_info = None
    max_iterations = 10
    backend = Aer.get_backend('qasm_simulator')
    
    for i in range(max_iterations):
        print(f"Optimization iteration:{i+1}")
        print(f"Current best L = {L_current_best:.2f}")

        oracle_gate = get_flexible_oracle_gate(anc, L_current_best, S, comp_anc)
        
        grover_circuit = QuantumCircuit(*problem_registers, name='Grover_Iter')
        qpe_qubits = [q for reg in qpe_registers for q in reg]
        grover_circuit.append(qpe_gate, qpe_qubits)
        grover_circuit.append(oracle_gate, [*anc, *comp_anc])
        grover_circuit.append(iqpe_gate, qpe_qubits)
        grover_circuit.append(diffuser_gate, tour_qubits)
        grover_gate = grover_circuit.to_gate()

        num_counting_qubits = 4 # Accuracy of QCA measurement
        qca_anc = QuantumRegister(num_counting_qubits, 'qca_anc')
        c_qca = ClassicalRegister(num_counting_qubits, 'c_qca')
        qca_qc = QuantumCircuit(qca_anc, *problem_registers, c_qca)
        qca_qc.h(qca_anc)
        qca_qc.append(prep_gate, tour_qubits)
        
        controlled_grover = grover_gate.control(1)
        for i_qca, control_qubit in enumerate(qca_anc):
            for _ in range(2**i_qca):
                qca_qc.append(controlled_grover, [control_qubit] + problem_qubits)
                
        qca_qc.append(QFT(num_counting_qubits, do_swaps=True, inverse=True).decompose(), qca_anc)
        qca_qc.measure(qca_anc, c_qca)
        
        result_qca = backend.run(transpile(qca_qc, backend), shots=100).result()
        counts_qca = result_qca.get_counts()
        measured_binary = max(counts_qca, key=counts_qca.get)
        measured_int = int(measured_binary, 2)
        phi = measured_int / (2**num_counting_qubits)
        if phi > 0.5: phi = 1 - phi
        theta = 2 * pi * phi
        M = round(N_total * (sin(theta / 2)**2))
        print(f"QCA measurement: {measured_binary}, measured phase(phi): {phi:.3f}")
        print(f"QCA estimated result: Number of paths shorter than L (M) ≈ {M}")

        if M == 0:
            print("\nNo shorter paths found. Stopping optimization.")
            break
        
        k = floor((pi / 4) * sqrt(N_total / M))
        print(f"Optimal number of iterations k = {k}")
        
        c_tour = ClassicalRegister(len(tour_qubits))
        search_qc = QuantumCircuit(*problem_registers, c_tour)
        search_qc.x(ph)
        search_qc.append(prep_gate, tour_qubits)
        if k > 0:
            for _ in range(k):
                search_qc.append(grover_gate, problem_qubits)
                
        search_qc.measure(tour_qubits, c_tour)
        
        result_search = backend.run(transpile(search_qc, backend), shots=1024).result()
        counts_search = result_search.get_counts()
        best_bitstring = max(counts_search, key=counts_search.get)
        new_tour, new_cost = tour_map[best_bitstring]
        print(f"Search result: Tour {new_tour} (Cost: {new_cost})")

        if new_cost < L_current_best:
            L_current_best = new_cost
            optimal_tour_info = (new_tour, new_cost)
        else:
            print("Measured path is not better than current best. Stopping optimization.")
            break
        print("----------------------------\n")

    # final results
    print("\n Final Optimization Results ")
    if optimal_tour_info:
        tour, cost = optimal_tour_info
        print(f"Found optimal tour: {tour}")
        print(f"Optimal tour cost: {cost}")
    else:
        print("No optimal tour found. (No paths shorter than initial threshold)")