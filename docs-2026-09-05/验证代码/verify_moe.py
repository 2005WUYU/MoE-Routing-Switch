#!/usr/bin/env python3
"""CPU-only analytical/synthetic checks, not an MoE pretraining experiment.

Usage: python verify_moe.py --output-dir ../数值结果
Requires Python 3.10+ and NumPy. No network or accelerator is used.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import platform
import sys
from pathlib import Path
from typing import Any
import numpy as np


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.ndim != 2 or not 0 < k <= scores.shape[1]:
        raise ValueError("scores must be [tokens, experts] and 1 <= k <= experts")
    # Input column order is the global expert ID, hence stable ties are explicit.
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def route(scores: np.ndarray, outputs: np.ndarray, ids: np.ndarray,
          normalized: bool = True) -> np.ndarray:
    if outputs.shape[:2] != scores.shape or ids.shape[0] != scores.shape[0]:
        raise ValueError("Inconsistent token/expert dimensions")
    selected = np.take_along_axis(scores, ids, axis=1)
    if normalized:
        e = np.exp(selected - selected.max(axis=1, keepdims=True))
        weights = e / e.sum(axis=1, keepdims=True)
    else:
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        weights = np.take_along_axis(e / e.sum(axis=1, keepdims=True), ids, axis=1)
    picked = np.take_along_axis(outputs, ids[:, :, None], axis=1)
    return (picked * weights[:, :, None]).sum(axis=1)


def independent_dense_reference(scores: np.ndarray, outputs: np.ndarray,
                                ids: np.ndarray) -> np.ndarray:
    # Deliberately uses explicit scalar loops, not the gather/normalization path above.
    result = np.zeros((scores.shape[0], outputs.shape[2]), dtype=np.float64)
    for t in range(scores.shape[0]):
        denominator = sum(float(np.exp(scores[t, e])) for e in ids[t])
        for e in range(scores.shape[1]):
            if e in ids[t]:
                result[t] += float(np.exp(scores[t, e])) / denominator * outputs[t, e]
    return result


def energy(x: np.ndarray) -> float:
    return float(np.mean(np.square(x)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "数值结果")
    parser.add_argument("--grid-power", type=int, default=20)
    args = parser.parse_args()
    if not 16 <= args.grid_power <= 23:
        parser.error("grid-power must be between 16 and 23")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260905)
    tests: list[dict[str, Any]] = []

    def record(name: str, passed: bool, **details: Any) -> None:
        tests.append({"name": name, "passed": bool(passed), "details": details})

    z0 = rng.normal(size=(64, 8))
    z1 = z0 + rng.normal(scale=0.6, size=z0.shape)
    e0 = rng.normal(size=(64, 8, 5))
    e1 = e0 + rng.normal(scale=0.1, size=e0.shape)
    a0, a1 = topk(z0, 2), topk(z1, 2)
    y0, middle, y1 = route(z0, e0, a0), route(z1, e1, a0), route(z1, e1, a1)
    reference = independent_dense_reference(z1, e1, a0)
    ref_error = float(np.max(np.abs(middle - reference)))
    record("independent_reference", ref_error < 2e-14, max_error=ref_error)

    u, v, total = middle - y0, y1 - middle, y1 - y0
    S, J, C = energy(u), energy(v), float(np.mean(u*v))
    err = abs(energy(total) - S - J - 2*C)
    record("exact_energy_decomposition", err < 2e-14 and abs(C) <= np.sqrt(S*J)+1e-14,
           S=S, J=J, C=C, total=energy(total), error=err)

    bad_middle = middle + 0.25
    bad_reconstruction = float(np.max(np.abs((y1-y0) - (bad_middle-y0) - (y1-bad_middle))))
    bad_reference_error = float(np.max(np.abs(bad_middle-reference)))
    record("mutation_tautology_is_insufficient", bad_reconstruction < 2e-14 and bad_reference_error > 0.2,
           reconstruction_error=bad_reconstruction, independent_reference_error=bad_reference_error)

    b0, b1 = np.array([[0.1, 0.0]]), np.array([[0.0, 0.1]])
    m0 = float(np.sort(b0[0])[1] - np.sort(b0[0])[0])
    m1 = float(np.sort(b1[0])[1] - np.sort(b1[0])[0])
    r_after = float(b1[0, 0] - b1[0, 1])
    record("signed_margin_not_resorted_gap", m0 == m1 and r_after < 0 and topk(b0,1)[0,0] != topk(b1,1)[0,0],
           sorted_margin_before=m0, sorted_margin_after=m1, fixed_pair_margin_after=r_after)

    same = np.ones((1,3,4))
    z = np.array([[0.0,2.0,1.0]])
    old, new = np.array([[0]]), np.array([[1]])
    jnorm = energy(route(z,same,new) - route(z,same,old))
    jraw = energy(route(z,same,new,False) - route(z,same,old,False))
    record("identical_experts_gate_mass_condition", jnorm == 0.0 and jraw > 0.1,
           normalized_J=jnorm, unnormalized_J=jraw)

    shared = rng.normal(size=y1.shape)
    shared_error = float(np.max(np.abs(((y1+shared)-(middle+shared)) - v)))
    record("additive_shared_cancellation", shared_error < 2e-14, max_error=shared_error)

    n = 2**args.grid_power
    x = (np.arange(n, dtype=np.float64) + 0.5) * (2.0/n) - 1.0
    sign0 = np.sign(x)
    alphas = 2.0**np.arange(-12, -3, dtype=np.float64)
    rows = []
    for alpha in alphas:
        switched = np.sign(x-alpha) != sign0
        d_route = np.sign(x-alpha)-sign0
        d_smooth = np.full_like(x, alpha)
        p = float(np.mean(switched))
        j = energy(d_route)
        s = energy(d_smooth)
        c = float(np.mean(d_route*d_smooth))
        rows.append({"alpha": float(alpha), "switch_probability": p, "J": j,
                     "route_rms": float(np.sqrt(j)), "S": s, "smooth_rms": float(np.sqrt(s)),
                     "C": c, "total_energy": energy(d_route+d_smooth),
                     "expected_probability": float(alpha/2), "expected_J": float(2*alpha)})
    slopes = {key: float(np.polyfit(np.log(alphas), np.log([r[key] for r in rows]), 1)[0])
              for key in ("switch_probability", "route_rms", "smooth_rms")}
    max_j_error = max(abs(r["J"]-r["expected_J"]) for r in rows)
    record("hard_boundary_square_root_law", max_j_error < 1e-14 and
           abs(slopes["switch_probability"]-1)<1e-12 and abs(slopes["route_rms"]-0.5)<1e-12 and
           abs(slopes["smooth_rms"]-1)<1e-12, slopes=slopes, max_J_error=max_j_error)
    record("cross_term_exact_uniform_case", all(abs(r["C"] + r["alpha"]**2)<1e-14 and
           abs(r["total_energy"]-(2*r["alpha"]-r["alpha"]**2))<1e-14 for r in rows),
           expected_C="-alpha^2", expected_total="2*alpha-alpha^2")

    folded_velocity = -np.sign(x)
    correct_coefficient = float(np.mean(np.maximum(-folded_velocity,0)))
    wrong_coefficient = float(np.mean(np.abs(folded_velocity)))
    record("one_sided_flux_coefficient", correct_coefficient == 0.5 and wrong_coefficient == 1.0,
           actual_probability_coefficient=0.5, one_sided_coefficient=correct_coefficient,
           incorrect_absolute_velocity_coefficient=wrong_coefficient)

    finite = np.array([-0.8,-0.3,0.2,0.6])
    below = float(np.mean(np.sign(finite-0.01) != np.sign(finite)))
    above = float(np.mean(np.sign(finite-0.3) != np.sign(finite)))
    record("finite_probe_zero_switch_interval", below == 0 and above == 0.25,
           rate_at_alpha_0_01=below, rate_at_alpha_0_3=above, first_positive_crossing=0.2)

    relu_rows = []
    for alpha in alphas[2:]:
        old_output = np.maximum(x,0.0)
        new_output = np.maximum(x+alpha,0.0)
        old_support_output = (x>0)*(x+alpha)
        jr = energy(new_output-old_support_output)
        full = energy(new_output-old_output)
        relu_rows.append({"alpha":float(alpha), "support_J":jr, "expected_J":float(alpha**3/6),
                          "support_rms":float(np.sqrt(jr)), "total_energy":full,
                          "expected_total_energy":float(alpha**2/2+alpha**3/6)})
    relu_slope = float(np.polyfit(np.log([r["alpha"] for r in relu_rows]),
                                 np.log([r["support_rms"] for r in relu_rows]),1)[0])
    relu_rel_error = max(abs(r["support_J"]/r["expected_J"]-1) for r in relu_rows)
    # Midpoint quadrature error grows with the smallest number of crossing cells.
    relu_tol = 1.0/(n*float(alphas[2])/2)**2
    record("relu_support_change_is_continuous", relu_rel_error < relu_tol and abs(relu_slope-1.5)<max(1e-5,relu_tol),
           rms_slope=relu_slope, max_relative_quadrature_error=relu_rel_error, tolerance=relu_tol)

    alpha = 1/32
    h0, h1 = sign0, np.sign(x-alpha)
    # Layer 2 is the identity with unchanged routing. Layer 1 contains the hard change.
    local_S = energy(h1-h0)
    local_J = 0.0
    global_fixed_output = h0
    global_U = energy(global_fixed_output-h0)
    global_V = energy(h1-global_fixed_output)
    record("upstream_jump_contaminates_local_remainder", local_S > 0 and local_J == 0 and global_U == 0 and local_S == global_V,
           layer2_local_S=local_S, layer2_local_J=local_J, global_fixed_graph_energy=global_U,
           global_routing_energy=global_V)

    a, delta, eta = 2.0, 0.1, 0.25
    target = a*x+delta
    branch_gradient = float(np.mean(2*(sign0-target)))
    alpha = -eta*branch_gradient
    r0 = energy(sign0-target)
    r_fixed = energy(alpha+sign0-target)
    r_actual = energy(alpha+np.sign(x-alpha)-target)
    fixed_change, actual_change = r_fixed-r0, r_actual-r0
    harm_J = energy(np.sign(x-alpha)-sign0)
    tol = 8.0/n
    record("sgd_step_boundary_harm_counterexample", abs(branch_gradient+0.2)<1e-12 and
           abs(fixed_change+0.0075)<tol and abs(actual_change-0.0025)<tol and abs(harm_J-0.1)<tol,
           branch_gradient=branch_gradient, alpha=alpha, fixed_route_risk_change=fixed_change,
           actual_risk_change=actual_change, J=harm_J, expected_fixed_change=-0.0075,
           expected_actual_change=0.0025, expected_J=0.1, quadrature_tolerance=tol)

    eps = 1/128
    risk_plus = energy(eps+np.sign(x-eps)-target)
    risk_minus = energy(-eps+np.sign(x+eps)-target)
    central = (risk_plus-risk_minus)/(2*eps)
    record("expectation_derivative_boundary_term", abs(central)<1e-10 and abs(branch_gradient+0.2)<1e-12,
           population_derivative_finite_difference=central, mean_branch_derivative=branch_gradient,
           required_boundary_correction=0.2)

    B = 16
    individual_gradient = 2*(sign0-target)
    batch_gradient_variance = float(np.var(individual_gradient))/B
    second_theta_moment = eta**2*(branch_gradient**2+batch_gradient_variance)
    expected_sgd_actual = (a-1)*second_theta_moment
    expected_sgd_fixed = -4*eta*delta**2+second_theta_moment
    record("finite_batch_sgd_expected_harm", abs(expected_sgd_actual-(0.0025+1/(12*B)))<1e-10 and
           abs(expected_sgd_fixed-(-0.0075+1/(12*B)))<1e-10 and expected_sgd_actual>0 and expected_sgd_fixed<0,
           batch_size=B, batch_gradient_variance=batch_gradient_variance,
           expected_actual_risk_change=expected_sgd_actual, expected_fixed_route_risk_change=expected_sgd_fixed,
           method="Moments checked using deterministic quadrature; no stochastic training run.")

    values = np.array([0.1,0.5,2.0,7.0])
    pi = np.array([0.1,0.2,0.5,0.9])
    true_mean = float(values.mean())
    expectation, second, naive = 0.0, 0.0, 0.0
    for bits in itertools.product([0,1],repeat=4):
        indicators = np.array(bits)
        prob = float(np.prod(np.where(indicators,pi,1-pi)))
        estimate = float(np.mean(indicators*values/pi))
        expectation += prob*estimate
        second += prob*estimate**2
        naive += prob*float(np.mean(indicators*values))
    var_formula = float(np.sum((1-pi)/pi*values**2)/16)
    var_enum = second-expectation**2
    record("horvitz_thompson_exact_enumeration", abs(expectation-true_mean)<1e-13 and abs(var_enum-var_formula)<1e-12,
           population_mean=true_mean, expected_HT=expectation, variance_enumerated=var_enum,
           variance_formula=var_formula, expected_unweighted_zero_filled_mean=naive)

    scores = rng.normal(size=(10000,8))
    num_eps = 0.05
    perturbed = scores+rng.uniform(-num_eps,num_eps,size=scores.shape)
    ids = topk(scores,2)
    sorted_scores = np.sort(scores,axis=1)[:,::-1]
    margins = sorted_scores[:,1]-sorted_scores[:,2]
    certified = margins > 2*num_eps
    same_set = np.all(np.sort(ids,axis=1)==np.sort(topk(perturbed,2),axis=1),axis=1)
    violations = int(np.sum(certified & ~same_set))
    record("numerical_margin_certificate", violations == 0 and int(certified.sum())>0,
           certified_tokens=int(certified.sum()), violations=violations, max_score_error=num_eps)

    d, p, c, eta_group, beta = 7,11,0.7,0.02,0.3
    phi = rng.normal(size=p)
    wi,wj = rng.normal(size=(d,p)),rng.normal(size=(d,p))
    D = c*(wj-wi)@phi
    grad_j = 2*c*np.outer(D,phi)/d
    grad_i = -grad_j
    D_next = c*((wj-eta_group*beta*grad_j)-(wi-eta_group*beta*grad_i))@phi
    factor = 1-4*eta_group*beta*c*c*float(phi@phi)/d
    factor_error = float(np.max(np.abs(D_next-factor*D)))
    record("expert_agreement_exact_sgd_factor", factor_error<2e-14 and energy(D_next)<energy(D),
           contraction_factor=factor, formula_error=factor_error, Q_before=energy(D),Q_after=energy(D_next))

    widths = [16,64]
    fixed_c_factors = [1-4*0.02*1.0*w/w for w in widths]
    scaled_c_factors = [1-4*0.02*(1/w)*w/w for w in widths]
    record("normalization_alone_does_not_transfer_beta", fixed_c_factors[0]==fixed_c_factors[1] and
           scaled_c_factors[0]!=scaled_c_factors[1], widths=widths,
           fixed_c_factors=fixed_c_factors, inverse_sqrt_width_c_factors=scaled_c_factors)

    # A task reads coordinate 0 only; arbitrarily large changes in coordinate 1 have no effect.
    before = np.array([1.0,0.0])
    after = np.array([1.0,1000.0])
    task_before,task_after = float((before[0]-2)**2),float((after[0]-2)**2)
    record("large_J_zero_task_effect", task_before==task_after and energy(after-before)>100000,
           J=energy(after-before), task_loss_difference=task_after-task_before)

    report = {"scope":"Analytical toy models and numerical/unit checks only; no LLM pretraining was run.",
              "seed":20260905,"grid_points":n,"python":platform.python_version(),"numpy":np.__version__,
              "test_count":len(tests),"passed":sum(t["passed"] for t in tests),"tests":tests}
    (out/"results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for filename, data in [("hard_boundary_scan.csv",rows),("relu_boundary_scan.csv",relu_rows)]:
        with (out/filename).open("w",newline="",encoding="utf-8") as f:
            writer = csv.DictWriter(f,fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps({"tests":report["test_count"],"passed":report["passed"],"output":str(out)},ensure_ascii=False))
    failed=[t["name"] for t in tests if not t["passed"]]
    if failed:
        print("FAILED: "+", ".join(failed),file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
