# Network structures used in empirical exercise
using Flux
network_structures = [
    Flux.Chain(RNN(3, 3), Dense(3, 1)), 
    Flux.Chain(RNN(3, 20), Dense(20, 1)),
    Flux.Chain(LSTM(3, 3), Dense(3, 1)), 
    Flux.Chain(LSTM(3, 20), Dense(20, 1)),
]