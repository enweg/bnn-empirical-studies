using BFlux, Flux
using Random, Distributions
using JLD
using CSV, DataFrames
using DataFramesMeta
using Dates

################################################################################
#### Data Preparation

df = DataFrame(CSV.File("../empirical-var/data/wide-data.csv"))
df = @chain df begin
    @rtransform(:Date = Date(:Date[1:10]))
    @orderby(:Date)
end

spx_rv = findfirst(x -> x == "rv5_SPX", names(df))
data = Matrix(df[:, 2:end])
dates = df[!, :Date]

# Creating the subsequences. Keep in mind that returns are the second column in
# this tensor
seq_len = 60+1
tensor = BFlux.make_rnn_tensor(data, seq_len)
tensor = Float32.(tensor)
y = tensor[end, spx_rv, :]
x = tensor[1:end-1, :, :]

test_from_index = findfirst(d -> d >= Date("2019-01-01"), dates) - seq_len
y_train = y[1:test_from_index-1]
y_test = y[test_from_index:end]
x_train = x[:, :, 1:test_from_index-1]
x_test = x[:, :, test_from_index:end]

save(
    "data/informationset-spillover.jld", 
    "y-train", y_train, 
    "y-test", y_test, 
    "x-train", x_train, 
    "x-test", x_test
)

################################################################################
#### RNN-HAR definition

