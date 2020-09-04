def parameterize_model(model, weights):
    # function to parametrizes all the trainable variables of model using the stream of weight values in weights
    # This assumes weights are passed a single batch.
    weights = tf.reshape( weights, [-1] ) # reshape the parameters to a vector
    
    last_used = 0
    for i in range(len(model.layers)):
        # check to make sure only conv and fully connected layers are assigned weights.
        if 'conv' in model.layers[i].name or 'out' in model.layers[i].name or 'dense' in model.layers[i].name: 
            weights_shape = model.layers[i].kernel.shape
            no_of_weights = tf.reduce_prod(weights_shape)
            new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
            model.layers[i].kernel = new_weights
            last_used += no_of_weights
            
            if model.layers[i].use_bias:
              weights_shape = model.layers[i].bias.shape
              no_of_weights = tf.reduce_prod(weights_shape)
              new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
              model.layers[i].bias = new_weights
              last_used += no_of_weights