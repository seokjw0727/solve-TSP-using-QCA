import math
import random
import os
import numpy as np
import socket
import time
from collections import Counter
from typing import Tuple

from pyquil import Program
from pyquil.gates import H, X, Z, CNOT, CCNOT, CZ, SWAP, CPHASE, MEASURE
from pyquil.api import get_qc

os.environ.setdefault("QVM_URL",   "http://localhost:5000")
os.environ.setdefault("QUILC_URL", "http://localhost:5555")

def _wait_port(host: str, port: int, retries: int = 20, delay: float = 0.5):
    """
    host:port가 열릴 때까지 재시도하며 대기함. 열리면 True, 실패하면 False를 반환함.
    """
    for _ in range(retries):
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return True
        except OSError:
            time.sleep(delay)
    return False

def _run_with_retry(qc, exe, tries: int = 3, delay: float = 1.5):
    """
    PyQuil 실행을 네트워크/일시적 오류에 대비해 재시도하며 결과를 반환함.
    """
    for i in range(tries):
        try:
            return qc.run(exe)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(delay)

def gen_symmetric_dist_4(seed=42):
    """
    4개 도시의 대칭 거리행렬을 난수로 생성하여 반환함.
    """
    random.seed(seed)
    d01 = random.randint(1, 5)
    d02 = random.randint(1, 5)
    d03 = random.randint(1, 5)
    d12 = random.randint(1, 5)
    d13 = random.randint(1, 5)
    d23 = random.randint(1, 5)
    D = [[0, d01, d02, d03],
         [d01, 0, d12, d13],
         [d02, d12, 0, d23],
         [d03, d13, d23, 0]]
    return D

TOUR_LIST = [
    [0,1,2,3,0],
    [0,1,3,2,0],
    [0,2,1,3,0],
    [0,2,3,1,0],
    [0,3,1,2,0],
    [0,3,2,1,0],
]

def tour_lengths(D):
    """
    미리 정의한 6개의 투어에 대한 총 길이 리스트(길이 6)를 계산해 반환함.
    """
    def length(route):
        return sum(D[route[i]][route[i+1]] for i in range(len(route)-1))
    return [length(t) for t in TOUR_LIST]

def pretty_tours(D, lens):
    """
    거리행렬과 6개 투어 및 각 길이를 보기 좋게 출력함.
    """
    print("=== Quantum TSP (pyQuil, MVP A=H^3) ===")
    print("Distance matrix:")
    for r in D:
        print("  ", r)
    print("\nTours and lengths:")
    for i, t in enumerate(TOUR_LIST):
        print(f"  idx {i}: {t} -> {lens[i]}")

def qft(program: Program, qubits):
    """
    입력 큐빗열에 대해 표준 QFT(스왑 포함)를 적용함. qubits는 MSB→LSB 순서로 가정함.
    """
    n = len(qubits)
    for j in range(n):
        qj = qubits[j]
        program += H(qj)
        for k in range(j+1, n):
            qk = qubits[k]
            angle = math.pi / (2 ** (k - j))
            program += CPHASE(angle, qk, qj)
    for j in range(n // 2):
        program += SWAP(qubits[j], qubits[n-1-j])

def iqft(program: Program, qubits):
    """
    qft의 역연산(IQFT)을 적용함. qubits는 MSB→LSB 순서로 가정함.
    """
    n = len(qubits)
    for j in range(n // 2):
        program += SWAP(qubits[j], qubits[n-1-j])
    for j in reversed(range(n)):
        qj = qubits[j]
        for k in reversed(range(j+1, n)):
            qk = qubits[k]
            angle = -math.pi / (2 ** (k - j))
            program += CPHASE(angle, qk, qj)
        program += H(qj)

def and_two_controls(prog: Program, c0, c1, tgt):
    """
    tgt <- tgt XOR (c0 AND c1)을 수행함. tgt가 0에서 시작하면 tgt=c0&c1이 됨.
    """
    prog += CCNOT(c0, c1, tgt)

def and_chain(prog: Program, ctrls, anc_chain):
    """
    다수 컨트롤의 AND를 anc_chain을 사용해 누적 계산함. 마지막 anc가 AND(ctrls)를 보유함.
    """
    assert len(ctrls) >= 2
    assert len(anc_chain) >= len(ctrls) - 1
    and_two_controls(prog, ctrls[0], ctrls[1], anc_chain[0])
    for i in range(2, len(ctrls)):
        and_two_controls(prog, anc_chain[i-2], ctrls[i], anc_chain[i-1])
    return anc_chain[len(ctrls)-2]

def and_chain_uncompute(prog: Program, ctrls, anc_chain):
    """
    and_chain으로 만든 AND 누적을 역연산하여 anc들을 깨끗하게 되돌림.
    """
    for i in reversed(range(2, len(ctrls))):
        prog += CCNOT(anc_chain[i-2], ctrls[i], anc_chain[i-1])
    prog += CCNOT(ctrls[0], ctrls[1], anc_chain[0])

def compute_tour_eq(prog: Program, tour_bits, value, t0, eq_out):
    """
    tour_bits가 value(0~5)와 정확히 같을 때 eq_out을 1로 만드는 비교기를 계산함.
    """
    flips = []
    for b in range(3):
        bit = tour_bits[b]
        want = (value >> b) & 1
        if want == 0:
            prog += X(bit)
            flips.append(bit)
    prog += CCNOT(tour_bits[0], tour_bits[1], t0)
    prog += CCNOT(t0, tour_bits[2], eq_out)
    prog += CCNOT(tour_bits[0], tour_bits[1], t0)
    for bit in flips:
        prog += X(bit)

def uncompute_tour_eq(prog: Program, tour_bits, value, t0, eq_out):
    """
    compute_tour_eq로 만든 표식을 역연산하여 eq_out과 보조비트를 모두 원상복구함.
    """
    flips = []
    for b in range(3):
        bit = tour_bits[b]
        want = (value >> b) & 1
        if want == 0:
            flips.append(bit)
    for bit in flips:
        prog += X(bit)
    prog += CCNOT(tour_bits[0], tour_bits[1], t0)
    prog += CCNOT(t0, tour_bits[2], eq_out)
    prog += CCNOT(tour_bits[0], tour_bits[1], t0)
    for bit in flips:
        prog += X(bit)

def apply_U_power_controlled(prog: Program, tour_bits, ctrl, ph, selAnc, lengths, p, power=1):
    """
    각 투어 상태 |i>에 길이 lengths[i]를 위상으로 부여하는 U^(power)를 ph에 킥백으로 적용함.
    ctrl이 주어지면 ctrl=1일 때만 적용되도록 제어함.
    """
    t0, eq_out = selAnc
    for i in range(6):
        compute_tour_eq(prog, tour_bits, i, t0, eq_out)
        if ctrl is not None:
            prog += CCNOT(eq_out, ctrl, t0)
            anc2 = t0
        else:
            anc2 = eq_out
        theta = 2.0 * math.pi * ((power * lengths[i]) / (2 ** p))
        prog += CPHASE(theta, anc2, ph)
        if ctrl is not None:
            prog += CCNOT(eq_out, ctrl, t0)
        uncompute_tour_eq(prog, tour_bits, i, t0, eq_out)

def set_flag_if_phase_lt_L(prog: Program, phase_bits, flag, cmpAnc, L):
    """
    위상레지스터의 정수값이 L보다 작으면 flag를 토글함. (OR_{x<L} [phase==x])
    """
    p = len(phase_bits)
    phase_lsb = list(reversed(phase_bits))
    for x in range(L):
        flips = []
        ctrls = []
        for b in range(p):
            bit = phase_lsb[b]
            want = (x >> b) & 1
            if want == 1:
                ctrls.append(bit)
            else:
                prog += X(bit)
                flips.append(bit)
                ctrls.append(bit)
        if len(ctrls) == 1:
            prog += CNOT(ctrls[0], flag)
        else:
            last = and_chain(prog, ctrls, cmpAnc)
            prog += CNOT(last, flag)
            and_chain_uncompute(prog, ctrls, cmpAnc)
        for bit in flips:
            prog += X(bit)

def append_paper_oracle(prog: Program, phase, tour, flag, cmpAnc, ph, selAnc, lengths, p, L, ctrl_qubit=None):
    """
    논문식 오라클(O_<L>): QPE → 비교기(<L) → 중심위상(Z/CZ) → 비교기 역연산 → inv-QPE를 구성함.
    ctrl_qubit이 주어지면 오라클 전체를 그 큐빗으로 제어함.
    """
    for qb in phase:
        prog += H(qb)
    for j in range(p):
        power = 2 ** j
        ctrl = phase[p-1-j]
        apply_U_power_controlled(prog, tour, ctrl, ph, selAnc, lengths, p, power=power)
    iqft(prog, phase)
    set_flag_if_phase_lt_L(prog, phase, flag, cmpAnc, L)
    if ctrl_qubit is None:
        prog += Z(flag)
    else:
        prog += CZ(ctrl_qubit, flag)
    set_flag_if_phase_lt_L(prog, phase, flag, cmpAnc, L)
    qft(prog, phase)
    for j in reversed(range(p)):
        power = 2 ** j
        ctrl = phase[p-1-j]
        lengths_neg = [-ell for ell in lengths]
        apply_U_power_controlled(prog, tour, ctrl, ph, selAnc, lengths_neg, p, power=power)
    for qb in phase:
        prog += H(qb)

def apply_diffusion_MVP_ctrl(prog: Program, tour, anc, count_ctrl=None):
    """
    A=H^3 기반의 경량 확산자 D를 구성해 투어 서브스페이스에서 반사 연산을 수행함.
    count_ctrl이 주어지면 중앙 반사를 그 큐빗으로 제어함.
    """
    for qb in tour:
        prog += H(qb)
    for qb in tour:
        prog += X(qb)
    t = tour[-1]
    ctrls = list(tour[:-1])
    if count_ctrl is not None:
        ctrls = [count_ctrl] + ctrls
    prog += H(t)
    if len(ctrls) == 0:
        prog += Z(t)
    elif len(ctrls) == 1:
        prog += CNOT(ctrls[0], t)
    elif len(ctrls) == 2:
        prog += CCNOT(ctrls[0], ctrls[1], t)
    elif len(ctrls) == 3:
        prog += CCNOT(ctrls[0], ctrls[1], anc)
        prog += CCNOT(anc, ctrls[2], t)
        prog += CCNOT(ctrls[0], ctrls[1], anc)
    else:
        raise ValueError("Too many controls for 3-qubit tour in MVP diffusion.")
    prog += H(t)
    for qb in tour:
        prog += X(qb)
    for qb in tour:
        prog += H(qb)

def run_qca(lengths, p=5, m=5, seed=7):
    """
    논문식 오라클과 경량 확산자를 사용한 양자카운팅 회로를 생성·실행하고 추정치를 반환함.
    """
    random.seed(seed)
    n_count = m
    n_phase = p
    n_tour  = 3
    n_flag  = 1
    n_cmp   = max(0, p-1)
    n_sel   = 2
    n_ph    = 1
    n_diffA = 1
    base = 0
    idx_count = list(range(base, base+n_count)); base += n_count
    idx_phase = list(range(base, base+n_phase)); base += n_phase
    idx_tour  = list(range(base, base+n_tour));  base += n_tour
    idx_flag  = [base]; base += 1
    idx_cmp   = list(range(base, base+n_cmp));   base += n_cmp
    idx_sel   = list(range(base, base+n_sel));   base += n_sel
    idx_ph    = [base]; base += 1
    idx_dAnc  = [base]; base += 1
    prog = Program()
    prog += X(idx_ph[0])
    for qb in idx_tour:
        prog += H(qb)
    for qb in idx_count:
        prog += H(qb)
    def append_controlled_G_once(L, ctrl_bit):
        """
        제어 비트 하나로 오라클과 확산자를 한 번 적용함.
        """
        nonlocal prog
        append_paper_oracle(prog, idx_phase, idx_tour, idx_flag[0],
                            idx_cmp, idx_ph[0], idx_sel, lengths, p, L, ctrl_qubit=ctrl_bit)
        apply_diffusion_MVP_ctrl(prog, idx_tour, idx_dAnc[0], count_ctrl=ctrl_bit)
    for b in range(m):
        ctrl = idx_count[b]
        reps = 2 ** (m - 1 - b)
        for _ in range(reps):
            L = max(lengths)
            append_controlled_G_once(L, ctrl)
    iqft(prog, idx_count)
    ro = prog.declare('ro', 'BIT', m)
    for j, qb in enumerate(idx_count):
        prog += MEASURE(qb, ro[j])
    shots = 512  # QCA 샷 수
    prog.wrap_in_numshots_loop(shots)
    qc  = get_qc("25q-pyqvm", compiler_timeout=600)
    exe = qc.compile(prog)
    res = qc.run(exe)
    reg = res.get_register_map()
    prefer = ('count', 'ro')
    key = next((k for k in prefer if k in reg), next(iter(reg)))
    arr = np.asarray(reg[key])
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3:
        arr = np.squeeze(arr)
    if arr.shape[0] != shots and arr.shape[1] == shots:
        arr = arr.T
    if arr.shape[0] != shots and arr.shape[0] == 1 and arr.shape[1] == shots:
        arr = arr.T
    arr = arr.astype(int, copy=False)
    bitstrings = ["".join('1' if b else '0' for b in row) for row in arr]
    counts = Counter(bitstrings)
    total = sum(counts.values())
    print(f"[QCA] shots={shots}  unique={len(counts)}  total={total}")
    print("[QCA] counts(top5):", counts.most_common(5))
    argmax_bs, _ = counts.most_common(1)[0]
    argmax_int = int(argmax_bs, 2)
    theta = argmax_int / (2 ** m)
    M_est = round(8 * (math.sin(math.pi * theta) ** 2))
    return {
        "counts": dict(counts),
        "argmax": argmax_bs,
        "theta": theta,
        "M_est": M_est,
        "qasm_qubits": total
    }

if __name__ == "__main__":
    D = gen_symmetric_dist_4(seed=7)
    lens = tour_lengths(D)
    pretty_tours(D, lens)
    out = run_qca(lens, p=3, m=3, seed=1234)
    print("\n[QCA] shots=512")
    print("[QCA] counts(top5):", sorted(out["counts"].items(), key=lambda kv: kv[1], reverse=True)[:5])
    print("[QCA] argmax:", out["argmax"], "→ int", int(out["argmax"], 2))
    print(f"[QCA] theta≈{out['theta']:.4f},  M_est≈{out['M_est']}  (N_eff=8)")
    print(f"[info] total logical qubits used: {out['qasm_qubits']}")
