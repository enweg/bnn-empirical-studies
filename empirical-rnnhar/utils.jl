function get_observed_quantiles(y, posterior_yhat, target_q = 0.05:0.05:0.95)
    qs = [quantile(yr, target_q) for yr in eachrow(posterior_yhat)]
    qs = reduce(hcat, qs)
    observed_q = mean(reshape(y, 1, :) .< qs; dims = 2)
    return observed_q
end
function plot_quantile_comparison(y, posterior_yhat, target_q = 0.05:0.05:0.95)
    observed_q = get_observed_quantiles(y, posterior_yhat, target_q)
    plot(target_q, observed_q, label = "Observed", legend_position = :topleft, 
        xlab = "Quantile of Posterior Draws", 
        ylab = "Percent Observations below"
    )
    plot!(x -> x, minimum(target_q), maximum(target_q), label = "Theoretical")
end