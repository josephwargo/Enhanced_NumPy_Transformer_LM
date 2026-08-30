import cupy as cp
import costs_and_activations as caa


class neuron_layer(object):
####################################
# Initializations #
####################################
    def __init__(self, input_shape, output_shape, activation,
                 clip_val, is_output_layer=False, adam=False, weight_decay=None):

        # layer info 
        self.input_shape = input_shape # length of vector that will be inputted into the weights
        self.output_shape = output_shape # desired length of weight output (and therefore layer output)
        layer_xavier = cp.sqrt(2/(self.input_shape+self.output_shape)) # value used to determine optimal
        # shape is (out_dim, in_dim) for hardware optimization. this is NOT untuitive
        self.layer_weights = cp.random.normal(0,layer_xavier, size=(self.output_shape, self.input_shape)).astype(cp.float32)
        self.bias = cp.zeros(shape=(output_shape)).astype(cp.float32) # initializing bias as 0s - no xavier here

        # activation
        self.activation = activation # activation function that will be applied to the output of the weights

        # hyperparameters
        self.clip_val = clip_val # used to set the upper and lower bounds for what we will let weight updates hit

        # hidden state
        self.hidden_state = None
        self.prev_layer_output = None

        # type of layer
        self.is_output_layer = is_output_layer

        self.dL_dW = cp.zeros_like(self.layer_weights).astype(cp.float32)
        self.dL_db = cp.zeros_like(self.bias).astype(cp.float32)

        # adam
        # TODO: implement
        self.adam = adam
        if adam:
            pass


####################################
# Forward Pass #
####################################
    def forward_pass(self, x, train=False):
        # if it is an output layer - no activation
        if self.is_output_layer:
            hidden_state = x @ self.layer_weights.T + self.bias
            # storing hidden state BEFORE any activation is done (could come later for output layers as these go into softmax for prediction)
            if train:
                self.hidden_state = hidden_state
                self.prev_layer_output = x
        # an other layer - yes activation
        else:
            hidden_state = x @ self.layer_weights.T + self.bias
            # storing hidden state BEFORE any activation is done
            if train:
                self.hidden_state = hidden_state
                self.prev_layer_output = x
            hidden_state = caa.activation(self.activation, hidden_state)
            
        return hidden_state

####################################
# Backward Pass #
####################################
    def backward_pass(self, dL_dY=None, logits=None, Y=None, pad_token_ind=0):

        if self.is_output_layer:
            
            # flattening and masking logits and Y
            Y_flat = Y.reshape(-1)
            mask = (Y_flat != pad_token_ind)
            Y_flat_masked = Y_flat[mask]
            logits_flat = logits.reshape(-1, logits.shape[-1])
            logits_flat_masked = logits_flat[mask]
            
            # grad for non-padded
            dL_dZ_active = caa.softmax_cross_entropy_grad(logits_flat_masked, Y_flat_masked)
            # reshaping to pre-flattened shape for dL_dZ
            dL_dZ_flat = cp.zeros_like(logits_flat)
            dL_dZ_flat[mask] = dL_dZ_active
            dL_dZ = dL_dZ_flat.reshape(logits.shape)
        
        # any other layer
        else:
            # calculating and flatting dL_dZ
            dY_dZ = caa.activation_grad(self.activation, self.hidden_state)
            dL_dZ = dL_dY * dY_dZ
            dL_dZ_flat = dL_dZ.reshape(-1, dL_dZ.shape[-1])
            
        # dL_dx
        dL_dx = dL_dZ @ self.layer_weights

        # dL_dW - requries flattening the previous layer hidden state and using the flattened dL_dZ
        prev_layer_output_flat = self.prev_layer_output.reshape(-1, self.prev_layer_output.shape[-1])
        
        self.dL_dW += dL_dZ_flat.T @ prev_layer_output_flat

        # dL_db
        self.dL_db += cp.sum(dL_dZ, axis=(0,1))

        return dL_dx

    def update(self, learning_rate):
        # clipping
        cp.clip(self.dL_dW, -self.clip_val, self.clip_val, out=self.dL_dW)
        cp.clip(self.dL_db, -self.clip_val, self.clip_val, out=self.dL_db)

        # adam
        # if self.adam:
        #     self.update_adam()
        # else:
        self.layer_weights += -learning_rate * self.dL_dW
        self.bias += -learning_rate * self.dL_db
    
    def clear_grad(self):
        self.dL_dW.fill(0)
        self.dL_db.fill(0)

    

