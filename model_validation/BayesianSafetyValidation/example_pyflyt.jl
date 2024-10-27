using Revise
# import Pkg; Pkg.add("BayesianSafetyValidation")
using BayesianSafetyValidation
using Distributions

@with_kw mutable struct PyFlytWaypointSystem <: System.SystemParameters
    x1 = 0.0
    y1 = 0.0
    z1 = 0.0
end

function System.evaluate(sparams::PyFlytWaypointSystem, inputs::Vector; kwargs...)
    @info "Evaluating with parameters:"
    @info "System Parameters: x1=$(sparams.x1), y1=$(sparams.y1), z1=$(sparams.z1)"
    @info "Inputs: $inputs"

    results = [strip(readchomp(`wsl python3 src/pyflyt_evaluate.py --waypoints $(x[1]) $(x[2]) $(x[3])`)) for x in inputs]

    for (i, result) in enumerate(results)
        println("Raw result from Python evaluation for input $i: $result")

        # !!! important: system convention: failure = 1, success = 0
        status = if occursin("success", result)
            0
        elseif occursin("failure", result)
            1
        else
            "unknown"
        end
        @info "Evaluation result for input $i: $status"
    end

    # bool array based on 1=failure, 0=success
    final_results = [occursin("failure", result) ? 1 : 0 for result in results]
    println("Final interpreted results for BSV: $final_results")

    return final_results
end



px1 = OperationalParameters("x1", [0.5, 2.0], Normal(1.0, 0.50))
py1 = OperationalParameters("y1", [-5.0, 5.0], Normal(0.0, 2.0))
pz1 = OperationalParameters("z1", [1.5, 1.5], Normal(1.0, 0.5))
model = [px1, py1, pz1]
#sysparams: 
system_params = PyFlytWaypointSystem()

@info "Fitting surrogate"
surrogate = bayesian_safety_validation(system_params, model; T=100, m=50, d=50)
@info "Done with surrogate"
# warm start - modify BSV to accept points
# surrogate = gp_fit(initialize_gp(; gp_args...), X, Y)
X_failures = falsification(surrogate.x, surrogate.y)
ml_failure = most_likely_failure(surrogate.x, surrogate.y, model)
p_failure  = p_estimate(surrogate, model)

@info "Plotting"
# (gp, models, sparams; inds=[:, :, fill(1, length(models) - 2)...], num_steps=200, m=fill(num_steps, length(models)), latex_labels=true, show_data=false, overlay=false, tight=true, use_heatmap=false, hide_model=true, hide_ranges=false, titlefontsize=12, add_phantom_point=true)

p = plot_surrogate_truth_combined(surrogate, model, system_params; hide_model=false, num_steps=5, show_data=true)
savefig(p, "pyflyt_bsv.png")
# t = plot_data!(surrogate.x, surrogate.y)
# savefig(t, "lunarlander_points.png")
j = plot_most_likely_failure(surrogate, model)
savefig(j, "pyflyt_mlf.png")
l = plot_distribution_of_failures(surrogate, model)
savefig(l, "dist_of_failures_pyflyt.png")
