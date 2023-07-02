# type system of neural networks

The goal of the system is to simplify the coding of  

----

a type is actually similar to the gym Box. typed_builder is a builder that is able to select the default config based on the type of the input and output 
objects will be configed as 


the major use case is to specify the type of the input and output of the neural network; ideally, when we specify the type of the data, we can automatically generate the neural network based on some high-level specification; instead of choosing the networks by ourselves.
for example: we can define a seq(a) -> seq(a) network, and the type of the input and output is seq, then we can automatically generate the network based on the element of the seq(a)

another things to choose is that if we need to use the dynamic types? Infer the input and select the result? 


abstract operators for datas:
  - stack (generate sequence)
  - UNet (transformation)
  - batchify a sequence (sparse, or dense mode)
  - reshape a batch of data back in to packed sequences
  - Encoding, Decoding
  - reshape (only reshape the batch dimension)
 Also, we can support different ways of fusion different things. 


for probabilistic distributions, we have the operators
  - sample (size, ...) -> data, logp, entropy
  - rsample if possible (size, ...)
  - log_prob (evaluate the log prob of the data, ...) (if supported)

we build the stochastic computation graph for all elements
  - we can sample a element by sampling its all ancestors
  - we can condition a element by conditioning its all ancestors
  - one can evaluate the log prob of a element (together with conditions and sampled results) by evaluating the log prob of its all sampled ancestors