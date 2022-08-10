# Network structures used in empirical exercise
using Flux
network_structures = [
    Flux.Chain(RNN(38, 38), Dense(38, 1)), 
    Flux.Chain(RNN(38, 19), Dense(19, 1)), 
    Flux.Chain(LSTM(38, 38), Dense(38, 1)), 
    Flux.Chain(LSTM(38, 19), Dense(19, 1)), 
]