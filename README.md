# tripletloss-keras-tensorflow
There exist many code examples across the web for implementing triplet-loss objectives in Tensorflow using Keras. However, none of these examples place triplet-loss in the context of other commonly needed functionality, such as loading arbitrarily large non-standard datasets from text files, using data augmentation across the multiple inputs, using an optimizer with a variable learning rate across pre-trained and randomly initialized layers of a network, or loading models with custom optimizers and objective losses. 

Here, with the help of some of the other web sources, I have assembled an example of a triplet-loss objective with all of the common pieces one might need. Hope some find it helpful/useful.  
