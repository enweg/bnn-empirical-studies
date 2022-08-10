# Network structures used in empirical exercise
using Flux
network_structures = [
    Flux.Chain(RNN(3, 3), Dense(3, 1)), 
    Flux.Chain(RNN(3, 10), Dense(10, 1)), 
    Flux.Chain(RNN(3, 15), Dense(15, 1)), 
    Flux.Chain(RNN(3, 20), Dense(20, 1)),
    Flux.Chain(RNN(3, 3), Dense(3, 3, sigmoid), Dense(3, 1)), 
    Flux.Chain(RNN(3, 10), Dense(10, 10, sigmoid), Dense(10, 1)), 
    Flux.Chain(RNN(3, 3), Dense(3, 3, relu), Dense(3, 1)), 
    Flux.Chain(RNN(3, 10), Dense(10, 10, relu), Dense(10, 1)), 
    Flux.Chain(LSTM(3, 3), Dense(3, 1)), 
    Flux.Chain(LSTM(3, 10), Dense(10, 1)), 
    Flux.Chain(LSTM(3, 15), Dense(15, 1)), 
    Flux.Chain(LSTM(3, 20), Dense(20, 1)),
    Flux.Chain(LSTM(3, 3), Dense(3, 3, sigmoid), Dense(3, 1)), 
    Flux.Chain(LSTM(3, 10), Dense(10, 10, sigmoid), Dense(10, 1)), 
    Flux.Chain(LSTM(3, 3), Dense(3, 3, relu), Dense(3, 1)), 
    Flux.Chain(LSTM(3, 10), Dense(10, 10, relu), Dense(10, 1)), 
]