@doc generic_solver_docstring(
    "Spectral Deferred Correction.

Solves the collocation problem on `num_nodes` quadrature nodes by preconditioned
iteration. On each step the collocation system

```
u_m = u_n + Δt Σ_j Q[m,j] f(u_j, t_n + τ_j Δt),   m = 1, …, M
```

is not solved directly. Instead `num_sweeps` sweeps of

```
u^{k+1} - Δt QΔ f^{k+1} = u_n + Δt (Q - QΔ) f^k
```

are taken, where `QΔ` is a cheap lower-triangular approximation of the dense
quadrature matrix `Q` chosen by `sweeper`. Because `QΔ` is lower triangular the
sweep decomposes into `M` successive implicit solves of exactly the shape a DIRK
stage solves, so each sweep costs about as much as one `M`-stage DIRK step.

Each sweep raises the order of the method by (at least) one until the order of
the underlying collocation method is reached, so the accuracy is tuned by
`num_sweeps` and `num_nodes` rather than by picking a different tableau.

Adaptive whenever `num_sweeps > 0`. The embedded estimate is the difference between the step updates
formed from the last two sweeps, which costs a handful of `axpy`s because both
iterates are already in the cache.

On a `SplitODEProblem` the sweep is semi-implicit: `f1` goes through `QΔ` and a
nonlinear solve per node as above, while `f2` goes through the strictly lower
triangular `explicit_sweeper` and is only ever evaluated at solved node values.

Dense output is the collocation polynomial through the final sweep's node values.",
    "SDC",
    "Spectral Deferred Correction method.",
    """@article{dutt2000spectral,
    title={Spectral deferred correction methods for ordinary differential equations},
    author={Dutt, Alok and Greengard, Leslie and Rokhlin, Vladimir},
    journal={BIT Numerical Mathematics},
    volume={40},
    number={2},
    pages={241--266},
    year={2000},
    publisher={Springer}}
    @article{minion2003semi,
    title={Semi-implicit spectral deferred correction methods for ordinary differential equations},
    author={Minion, Michael L},
    journal={Communications in Mathematical Sciences},
    volume={1},
    number={3},
    pages={471--500},
    year={2003}}
    @article{weiser2015faster,
    title={Faster {SDC} convergence on non-equidistant grids by {DIRK} sweeps},
    author={Weiser, Martin},
    journal={BIT Numerical Mathematics},
    volume={55},
    number={4},
    pages={1219--1241},
    year={2015},
    publisher={Springer}}
    @inproceedings{speck2016inexactSDC,
    title={Inexact spectral deferred corrections},
    author={Speck, Robert and Ruprecht, Daniel and Minion, Matthew and Emmett, Matthew and Krause, Rolf},
    booktitle={Domain Decomposition Methods in Science and Engineering XXII},
    pages={389--396},
    year={2016},
    publisher={Springer}}
    @article{eisenstat1996inexactNewton,
    title={Choosing the forcing terms in an inexact Newton method},
    author={Eisenstat, Stanley C. and Walker, Homer F.},
    journal={SIAM Journal on Scientific Computing},
    volume={17},
    number={1},
    pages={16--32},
    year={1996},
    publisher={SIAM}}
    @article{freese2026parallelSDC,
    title={Parallel performance of shared memory parallel spectral deferred corrections},
    author={Freese, Philip and G{\"o}tschel, Sebastian and Lunet, Thibaut and Ruprecht, Daniel and Schreiber, Martin},
    journal={The International Journal of High Performance Computing Applications},
    volume={40},
    number={5},
    pages={643--655},
    year={2026},
    publisher={SAGE Publications}}""",
    """
    - `num_nodes`: number of collocation nodes `M` per step.
    - `node_type`: node distribution, `SDCNodes.Legendre` or `.Equidistant`.
    - `quad_type`: which interval endpoints are nodes, `SDCQuadrature.Gauss`
        (neither), `.RadauLeft` (left), `.RadauRight` (right) or `.Lobatto` (both).
    - `num_sweeps`: number of sweeps `K` per step.
    - `sweeper`: the preconditioner `QΔ`. `SDCSweeper.BE` (implicit Euler between
        the nodes), `.FE` (explicit Euler between the nodes), `.Trapezoid`, `.LU`
        (Weiser's LU trick), `.Picard` (`QΔ = 0`), `.BEpar` (diagonal, implicit
        Euler from the step start to each node) or `.MIN_SR_NS` (diagonal,
        `diag(τ)/M`).
    - `explicit_sweeper`: the preconditioner for `f2` when the problem is a
        `SplitODEProblem`, `SDCSweeper.FE` (explicit Euler between the nodes) or
        `.Picard` (`f2` lagged a whole sweep). Ignored otherwise.
    - `step_update`: how the step solution is formed from the node values,
        `SDCStepUpdate.Quadrature` (`u_n + Δt Σ_m w_m f_m`) or `.LastNode`
        (`u_M`, which requires the last node to be the right endpoint).
    - `threading`: run the `M` node solves of each sweep concurrently on the
        in-place path. `false` (the default) or `true`, or one of
        `OrdinaryDiffEqCore`'s `Sequential()`, `BaseThreads()` and
        `PolyesterThreads()` for control over the backend. Only valid for a
        diagonal `sweeper`, since any other one couples the nodes within a
        sweep. Out-of-place problems sweep serially whatever this is set to. On a
        `SplitODEProblem` it also needs `explicit_sweeper = SDCSweeper.Picard`,
        since explicit Euler couples each node to the ones before it.
    - `inexact_sweeps`: number of leading sweeps whose node solves may use a
        loosened convergence tolerance, in the spirit of the forcing terms of
        Eisenstat and Walker: a node-solve error committed in sweep `k` is
        damped by the sweeps after it, so early sweeps need less accurate
        solves. The inexact sweeps form a geometric ramp anchored at the first
        exact sweep: sweep `k ≤ inexact_sweeps` runs its node solves at
        `κ_base / factor^(inexact_sweeps - k + 1)`, one loosening step below
        the exact regime at the boundary, where `κ_base` is the `κ` of the
        `nlsolve` algorithm. Sweeps beyond `inexact_sweeps` — always including
        the final one — run at the `nlsolve` algorithm's own settings.
        `0` (the default) leaves every solve exact.
    - `inexact_factor`: the assumed per-sweep contraction of the SDC
        iteration; the node-solve tolerance of an inexact sweep is loosened by
        `1/inexact_factor` per ramp step. `1.0e-1` (the default) is
        conservative; a stiff problem with a fast-contracting sweeper such as
        `SDCSweeper.LU` typically justifies `1.0e-2`. Must lie in `(0, 1]`.
    """,
    """
    num_nodes = 3,
    node_type = SDCNodes.Legendre,
    quad_type = SDCQuadrature.RadauRight,
    num_sweeps = 3,
    sweeper = SDCSweeper.BE,
    explicit_sweeper = SDCSweeper.FE,
    step_update = SDCStepUpdate.Quadrature,
    threading = false,
    """
)
struct SDC{AD, F, F2, TO, CJ} <: OrdinaryDiffEqNewtonAdaptiveAlgorithm
    num_nodes::Int
    node_type::SDCNodes.T
    quad_type::SDCQuadrature.T
    num_sweeps::Int
    sweeper::SDCSweeper.T
    explicit_sweeper::SDCSweeper.T
    step_update::SDCStepUpdate.T
    threading::TO
    linsolve::F
    nlsolve::F2
    autodiff::AD
    concrete_jac::CJ
    inexact_sweeps::Int
    inexact_factor::Float64
end

function SDC(;
        num_nodes::Int = 3,
        node_type::SDCNodes.T = SDCNodes.Legendre,
        quad_type::SDCQuadrature.T = SDCQuadrature.RadauRight,
        num_sweeps::Int = 3,
        sweeper::SDCSweeper.T = SDCSweeper.BE,
        explicit_sweeper::SDCSweeper.T = SDCSweeper.FE,
        step_update::SDCStepUpdate.T = SDCStepUpdate.Quadrature,
        threading = false,
        autodiff = AutoForwardDiff(), concrete_jac = nothing,
        linsolve = nothing, nlsolve = NLNewton(),
        inexact_sweeps::Int = 0, inexact_factor = 1.0e-1
    )
    sdc_validate(num_nodes, quad_type, num_sweeps, step_update, sweeper, threading)
    sdc_validate_explicit(explicit_sweeper)
    sdc_validate_inexact(num_sweeps, inexact_sweeps, inexact_factor)
    autodiff = _fixup_ad(autodiff)
    return SDC(
        num_nodes, node_type, quad_type, num_sweeps, sweeper, explicit_sweeper,
        step_update, threading, linsolve, nlsolve, autodiff, _unwrap_val(concrete_jac),
        inexact_sweeps, inexact_factor
    )
end
