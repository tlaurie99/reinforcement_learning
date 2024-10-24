using Revise
# import Pkg; Pkg.add("BayesianSafetyValidation")
using BayesianSafetyValidation
using Distributions

@with_kw mutable struct PyFlytWaypointSystem <: System.SystemParameters
    x1 = 0.0 
    x2 = 0.0 
    x3 = 0.0 
    x4 = 0.0 
end

function System.evaluate(sparams::PyFlytWaypointSystem, inputs::Vector; kwargs...)
    return [readchomp(`C:/Users/tyler/AppData/Local/anaconda3/python.exe src/pyflyt_evaluate.py --waypoints $(x[1]) $(x[2]) $(x[3]) $(x[4])`) == "success" for x in inputs]
end


px1 = OperationalParameters("x1", [-5.0, 5.0], Normal(0.0, 1.0))
px2 = OperationalParameters("x2", [-5.0, 5.0], Normal(0.0, 1.0))
px3 = OperationalParameters("x3", [-5.0, 5.0], Normal(0.0, 1.0))
px4 = OperationalParameters("x4", [-5.0, 5.0], Normal(0.0, 1.0))
model = [px1, px2, px3, px4]
#sysparams: 
system_params = PyFlytWaypointSystem()

@info "Fitting surrogate"
surrogate = bayesian_safety_validation(system_params, model; T=10)
# warm start - modify BSV to accept points
# surrogate = gp_fit(initialize_gp(; gp_args...), X, Y)
X_failures = falsification(surrogate.x, surrogate.y)
ml_failure = most_likely_failure(surrogate.x, surrogate.y, model)
p_failure  = p_estimate(surrogate, model)

@info "Truth estimate"
#truth = truth_estimate(system_params, model) # when using simple systems
@info "Plotting"
p = plot_surrogate_truth_combined(surrogate, model, system_params; hide_model=false, num_steps=10, show_data=true)
savefig(p, "pyflyt_bsv.png")
# t = plot_data!(surrogate.x, surrogate.y)
# savefig(t, "lunarlander_points.png")
j = plot_most_likely_failure(surrogate, model)
savefig(j, "pyflyt_mlf.png")
l = plot_distribution_of_failures(surrogate, model)
savefig(l, "dist_of_failures_pyflyt.png")
