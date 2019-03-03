
PerioNet!
---


a peroidic NN component, maybe useful when fitting perodic functions?



Here is a few comparisons to regular NN.

The regular NN has 60 hidden neurons
The perioNN has 10 components


In all the examples below the NNs were trained 100 times on the observed data using Adam at every iteration

the "Classic" is the regular NN in the plots

see 
- [notebook](https://github.com/JonasRSV/PerioNet/blob/master/perionet/examples/Untitled.ipynb) to test any functions


#### sin(x)

![sin(x)](images/sin_demo.gif)

#### sin(x) + x + 4 * cos(x)

![sin cos](images/sin_cos.gif)

#### mixture of cos and sins over a big range

![mixture](images/big_range.gif)

#### Exponential and Polynomial combination

![exp poly](images/polyexp.gif)



