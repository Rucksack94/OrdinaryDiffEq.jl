isfsal(::SDC) = false
issplit(::SDC) = true

"""
    sdc_iteration_order(alg)

Order reached by the SDC iteration itself, before the collocation cap.

One order is gained per sweep, two on the first sweep for a second-order
sweeper, and one more if the step update is the quadrature rule rather than the
last node value. This is `qmat.utils.sdc.getOrderSDC` without its table of
empirically found bonus cases, so for some node/sweeper combinations (notably
`SDCSweeper.LU` and `SDCSweeper.MIN_SR_NS`) the observed order is higher than
this predicts.
"""
function sdc_iteration_order(alg::SDC)
    K = alg.num_sweeps
    order = 0
    if K > 0
        order += alg.sweeper in SDC_SECOND_ORDER_SWEEPERS ? 2 : 1
        order += K - 1
    end
    alg.step_update === SDCStepUpdate.Quadrature && (order += 1)
    return order
end

function alg_order(alg::SDC)
    coll = sdc_collocation_order(alg.num_nodes, alg.node_type, alg.quad_type)
    return max(1, min(sdc_iteration_order(alg), coll))
end

# The embedded solution is the step update formed one sweep earlier, so it is
# one order lower until the collocation ceiling flattens both.
alg_adaptive_order(alg::SDC) = max(1, alg_order(alg) - 1)

# With no sweep there is no earlier step update to compare against.
isadaptive(alg::SDC) = alg.num_sweeps > 0

"""
    sdc_validate_explicit(explicit_sweeper)

The explicit preconditioner has to be strictly lower triangular, or `f2` would
have to be evaluated at the value being solved for.
"""
function sdc_validate_explicit(explicit_sweeper::SDCSweeper.T)
    explicit_sweeper in (SDCSweeper.FE, SDCSweeper.Picard) || throw(
        ArgumentError(
            "SDC: `explicit_sweeper` must be `SDCSweeper.FE` or `SDCSweeper.Picard`, " *
                "got $(explicit_sweeper)"
        )
    )
    return nothing
end

"""
    sdc_validate(num_nodes, quad_type, num_sweeps, step_update)

Reject unusable parameter combinations at algorithm construction time.

This deliberately does not go through `prepare_alg`: the generic
`prepare_alg(::OrdinaryDiffEqImplicitAlgorithm, ::AbstractArray, …)` in
`OrdinaryDiffEqDifferentiation` does the autodiff preparation every implicit
solver needs, and adding an `SDC` method here would make the two ambiguous.
"""
function sdc_validate(
        num_nodes::Int, quad_type::SDCQuadrature.T,
        num_sweeps::Int, step_update::SDCStepUpdate.T,
        sweeper::SDCSweeper.T, threading
    )
    if isthreaded(threading) && !(sweeper in SDC_DIAGONAL_SWEEPERS)
        throw(
            ArgumentError(
                "SDC: `threading` needs a diagonal `sweeper`, since any other one " *
                    "couples the nodes within a sweep. Got $(sweeper); the diagonal " *
                    "sweepers are $(SDC_DIAGONAL_SWEEPERS)"
            )
        )
    end
    num_sweeps >= 0 ||
        throw(ArgumentError("SDC: `num_sweeps` must be ≥ 0, got $(num_sweeps)"))
    endpoint_on_left = quad_type in (SDCQuadrature.Lobatto, SDCQuadrature.RadauLeft)
    minimum_nodes = endpoint_on_left ? 2 : 1
    num_nodes >= minimum_nodes || throw(
        ArgumentError(
            "SDC: `num_nodes` must be ≥ $(minimum_nodes) for `quad_type = $(quad_type)`, " *
                "got $(num_nodes)"
        )
    )
    endpoint_on_right = quad_type in (SDCQuadrature.RadauRight, SDCQuadrature.Lobatto)
    if step_update === SDCStepUpdate.LastNode
        endpoint_on_right || throw(
            ArgumentError(
                "SDC: `step_update = SDCStepUpdate.LastNode` needs the last node to be " *
                    "the right endpoint, so `quad_type` must be `SDCQuadrature.RadauRight` " *
                    "or `SDCQuadrature.Lobatto`, got $(quad_type)"
            )
        )
        # With no sweep the node values are still the copy initialisation, so the
        # last of them is uₙ and the step would return the initial value forever.
        num_sweeps >= 1 || throw(
            ArgumentError(
                "SDC: `step_update = SDCStepUpdate.LastNode` needs `num_sweeps ≥ 1`, " *
                    "got $(num_sweeps)"
            )
        )
    end
    return nothing
end

"""
    sdc_validate_inexact(num_sweeps, inexact_sweeps, inexact_factor)

Reject unusable parameter combinations for the inexact SDC.

The final sweep must stay exact to produce the returned node values. On the 
adaptive version, the embedded estimate is the difference between the last 
two sweeps, thus a loose final sweep would corrupt both.

`inexact_factor` is the assumed per-sweep contraction of the SDC iteration,
so it must lie in `(0, 1]` and the node-solve tolerance of sweep `k` is 
loosened by `1/inexact_factor` per inexact sweep.
"""
function sdc_validate_inexact(num_sweeps::Int, inexact_sweeps::Int, inexact_factor)
    inexact_sweeps >= 0 || throw(ArgumentError(
        "SDC: `inexact_sweeps` must be ≥ 0, got $(inexact_sweeps)"
    ))
    (num_sweeps == 0 || inexact_sweeps < num_sweeps) || throw(ArgumentError(
        "SDC: `inexact_sweeps` must leave the final sweep exact, thus needs to be " *
            "< num_sweeps = $(num_sweeps), got $(inexact_sweeps)"
    ))
    inexact_factor > 0 || throw(ArgumentError(
        "SDC: `inexact_factor` is the assumed per-sweep contraction of the SDC " *
            "iteration and needs to be in (0, 1], but got $(inexact_factor)"
    ))
    inexact_factor <= 1 || throw(ArgumentError(
        "SDC: `inexact_factor` is the assumed per-sweep contraction of the SDC " *
            "iteration and needs to be in (0, 1], but got $(inexact_factor)"
    ))
    return nothing
end

# Functions to get `κ` and `max_iter` from the `nlsolve` algorithm, and 
# including the shipped defaults if an algorithm does not name them. 
# These are the settings the exact sweeps always fall back to.
_nlalg_get_κ(nlalg) = hasproperty(nlalg, :κ) ? nlalg.κ : 1 // 100
_nlalg_get_maxiters(nlalg) = hasproperty(nlalg, :max_iter) ? nlalg.max_iter : 10

"""
    sdc_nlsolve_κ(alg, sweep)

The convergence `κ` the node implicit solves of sweep `sweep` use, following 
inexact spectral deferred correction [Speck et al., "Inexact spectral deferred
corrections", 2016]. A node-solve error committed in sweep `k` is damped by
the sweeps after it, so early sweeps need less accurate solves.

The inexact sweeps form a geometric `κ`-ramp anchored at the first exact
sweep, in the spirit of the forcing terms of [Eisenstat and Walker, "Choosing
the forcing terms in an inexact Newton method", 1996]: sweep
`k ≤ inexact_sweeps` runs at `κ_base / factor^(inexact_sweeps - k + 1)`, one
loosening step below the exact regime at the boundary, where `κ_base` is the
`κ` of the `nlsolve` algorithm. Sweeps beyond `inexact_sweeps` run at `κ_base`. 
The ramp sits on the conservative side of the matched loosening, since sweep 
`k` has `num_sweeps - k` sweeps of damping ahead of it, which always exceeds 
its exponent. Inexact solves in this form are used in production SDC codes 
[Freese et al., "Parallel performance of shared memory parallel spectral deferred 
corrections", 2026].

The schedule plugs into the framework's own convergence test, which accepts a
Newton iteration when `η * ndz < κ` with `η = θ / (1 - θ)` (Eisenstat and Walker) 
estimated from the observed contraction, so a loose `κ` lets a node solve accept 
after its first Newton step.

`inexact_sweeps = 0` (the default) leaves every sweep at `κ_base`.
"""
function sdc_nlsolve_κ(alg::SDC, sweep::Int)
    κ = _nlalg_get_κ(alg.nlsolve)
    exponent = alg.inexact_sweeps - sweep + 1
    exponent <= 0 && return κ
    return κ * inv(alg.inexact_factor)^exponent
end

"""
    sdc_set_nlsolve_settings!(nlsolvers, alg, sweep)

Write the sweep-dependent node-solve settings into every per-node `NLSolver`.

`nlsolve!` reads `κ` and `maxiters` from mutable NLSolver` on every call.
The inexact sweep just sets them before sweeping. Made unconditionally, on 
exact sweeps writes the default settings from the the constructor back into fields, 
which keeps a single unbranched code path.

Loosening `κ` widens the threshold of the framework's own convergence test, which accepts 
a Newton iteration when `η * ndz < κ` with `η = θ / (1 - θ)` (Eisenstat and Walker) 
estimated from the observed contraction, so a loose `κ` lets a node implicit solve 
accept after reduced number of Newton steps.

`maxiters` is never touched, since a Newton loop that runs out of iterations reports 
`Divergence`, so truncating the iteration would fail the node and reject the step. 
Inexact sweeps stop early by meeting a loose convergence criterion, never by truncation.
"""
function sdc_set_nlsolve_settings!(nlsolvers, alg::SDC, sweep::Int)
    κ = sdc_nlsolve_κ(alg, sweep)
    maxiters = _nlalg_get_maxiters(alg.nlsolve)
    for nls in nlsolvers
        nls.κ = κ
        nls.maxiters = maxiters
    end
    return nothing
end