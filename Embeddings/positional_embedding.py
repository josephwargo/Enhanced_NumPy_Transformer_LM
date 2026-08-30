import cupy as cp

class positional_embedding(object):
####################################
# Initializations #
####################################
    def __init__(
            self, max_seq_len, input_layer_shape, clip_val
        ):

        self.max_seq_len = max_seq_len
        self.input_layer_shape = input_layer_shape
        self.clip_val = clip_val

        self.embeddings = cp.random.normal(0, .02, size=(self.max_seq_len, input_layer_shape)).astype(cp.float32)

        self.dL_dE = cp.zeros_like(self.embeddings).astype(cp.float32)

####################################
# Forward Pass #
####################################
    def forward_pass(self, x, seq_len, train=False):
        embeddings_sliced = self.embeddings[:seq_len]
        return x + embeddings_sliced

####################################
# Backward Pass #
####################################
    def backward_pass(self, dL_dY):
        # summing across batch axis so we can update pos embeddings
        # only updating gradients for positions that were in the seq_len

        self.dL_dE[:dL_dY.shape[1]] += cp.sum(dL_dY, axis=0)

        # returning so we can pass back to input layer
        return dL_dY

    
    def update(self, learning_rate):
        cp.clip(self.dL_dE, -self.clip_val, self.clip_val, out=self.dL_dE)
        self.embeddings += -learning_rate * self.dL_dE

    def clear_grad(self):
        self.dL_dE.fill(0)