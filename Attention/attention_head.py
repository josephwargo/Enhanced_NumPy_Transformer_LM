import cupy as cp
import costs_and_activations as caa

class attention_head(object):
####################################
# Initializations #
####################################
    def __init__(self, num_heads, d_model, clip_val):
        
        # storing dimensions
        self.num_heads = num_heads
        self.d_model = d_model
        self.clip_val = clip_val

        self.head_dimension = int(self.d_model / self.num_heads)
        self.dim_sqrt = cp.sqrt(self.head_dimension)

        # initialization of weights - using only Xavier for now
        self.dim_for_xavier = self.d_model + self.d_model
        xavier_val = cp.sqrt(2/(self.dim_for_xavier))

        # instead of having separate heads, initiating each weight matrix with the shape (d_model, d_model), as this is mathematically equivalent to having separate heads
        # does not matter whether in_dim or out_dim is first as both are d_model
        self.W_q = cp.random.normal(0, xavier_val, size=(self.d_model, self.d_model)).astype(cp.float32) # query weights
        self.W_k = cp.random.normal(0, xavier_val, size=(self.d_model, self.d_model)).astype(cp.float32) # key weights
        self.W_v = cp.random.normal(0, xavier_val, size=(self.d_model, self.d_model)).astype(cp.float32) # value weights

        # storing previous layer hidden state for backprop
        self.prev_layer_output = None

        # initialization of q, k, and v vectors
        self.q = None
        self.k = None
        self.v = None

        # for storing during train forward pass
        self.softmax_masked_score = None

        # storing grads
        self.dL_dW_q = cp.zeros_like(self.W_q).astype(cp.float32)
        self.dL_dW_k = cp.zeros_like(self.W_k).astype(cp.float32)
        self.dL_dW_v = cp.zeros_like(self.W_v).astype(cp.float32)

####################################
# Forward Pass #
####################################
    def calculate_q_k_v(self, x, train=False):
        # reshaping the output to be (batch, seq_len, num_heads, head_shape)
        # then transposing of shape (batch_size, num_heads, seq_len, head_shape)
        # as this is needed to calculate the attention score for different heads and must be done in two steps
        # will later reshape to (batch, seq_len, d_model) when we "concatenate" the separate heads in the attention block
        q_proj = (x @ self.W_q.T).reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dimension)
        self.q = q_proj.transpose(0,2,1,3)
        k_proj = (x @ self.W_k.T).reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dimension)
        self.k = k_proj.transpose(0,2,1,3)
        v_proj = (x @ self.W_v.T).reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dimension)
        self.v = v_proj.transpose(0,2,1,3)

        # if train, storing x
        if train:
            self.prev_layer_output = x

    def forward_pass(self, x, train=False):
        # calculating q, k, and v
        self.calculate_q_k_v(x, train)

        # calculating attention score & scaling
        score_matrix = self.q @ self.k.transpose(0,1,3,2)
        scaled_score_matrix = score_matrix / self.dim_sqrt

        # determining & applying the mask
        score_mask = cp.tril(cp.ones(shape=(x.shape[1], x.shape[1])))
        masked_score = cp.where(score_mask, scaled_score_matrix, -1e9)

        #applying softmax
        softmax_masked_score = caa.softmax(masked_score)

        # applying attention score to values to "weight" each word by its respective attention
        masked_self_attention = softmax_masked_score @ self.v
        if train:
            self.softmax_masked_score = softmax_masked_score
        return masked_self_attention

####################################
# Backward Pass #
####################################    
    def backward_pass(self, dL_dY):

        dL_dv = self.softmax_masked_score.transpose(0,1,3,2) @ dL_dY

        # gradient w.r.t. softmax_masked_scores
        dL_dSoftmax_masked_scores = dL_dY @ self.v.transpose(0,1,3,2)

        # softmax gradient
        dL_dMasked_scores = caa.softmax_grad(self.softmax_masked_score, dL_dSoftmax_masked_scores)
        
        # gradients for q and k
        # "unscaling"
        dL_dMasked_scores_unscaled = dL_dMasked_scores / self.dim_sqrt
        # dot products
        dL_dq = dL_dMasked_scores_unscaled @ self.k
        dL_dk = dL_dMasked_scores_unscaled.transpose(0,1,3,2) @ self.q

        # gradients for W_q, W_k, and W_v
        prev_layer_output_flat = self.prev_layer_output.reshape(-1, self.d_model)
        
        dL_dq_flat_1 = dL_dq.transpose(0,2,1,3).reshape(-1, self.d_model)
        dL_dk_flat_1 = dL_dk.transpose(0,2,1,3).reshape(-1, self.d_model)
        dL_dv_flat_1 = dL_dv.transpose(0,2,1,3).reshape(-1, self.d_model)

        self.dL_dW_q += dL_dq_flat_1.T @ prev_layer_output_flat
        self.dL_dW_k += dL_dk_flat_1.T @ prev_layer_output_flat
        self.dL_dW_v += dL_dv_flat_1.T @ prev_layer_output_flat

        # dL_dY output of the head - sum of dL_dY_q, dL_dY_k, and dL_dY_v
        # flattening to be 3d to pass back dL_dY
        dL_dq_flat_2 = dL_dq.transpose(0,2,1,3).reshape(dL_dq.shape[0], dL_dq.shape[2],  self.d_model)
        dL_dk_flat_2 = dL_dk.transpose(0,2,1,3).reshape(dL_dk.shape[0], dL_dk.shape[2],  self.d_model)
        dL_dv_flat_2 = dL_dv.transpose(0,2,1,3).reshape(dL_dv.shape[0], dL_dv.shape[2],  self.d_model)

        dL_dAttn_head_q = dL_dq_flat_2 @ self.W_q
        dL_dAttn_head_k = dL_dk_flat_2 @ self.W_k
        dL_dAttn_head_v = dL_dv_flat_2 @ self.W_v

        # taking the sume of all 3 to get the true dL_dY - this is mathmematically consistent with the chain rule
        dL_dAttn_head = dL_dAttn_head_q + dL_dAttn_head_k + dL_dAttn_head_v

        return dL_dAttn_head

    def update(self, learning_rate):
        # clipping
        cp.clip(self.dL_dW_q, -self.clip_val, self.clip_val, out=self.dL_dW_q)
        cp.clip(self.dL_dW_k, -self.clip_val, self.clip_val, out=self.dL_dW_k)
        cp.clip(self.dL_dW_v, -self.clip_val, self.clip_val, out=self.dL_dW_v)

        self.W_q += -learning_rate * self.dL_dW_q
        self.W_k += -learning_rate * self.dL_dW_k
        self.W_v += -learning_rate * self.dL_dW_v
    
    def clear_grad(self):
        self.dL_dW_q.fill(0)
        self.dL_dW_k.fill(0)
        self.dL_dW_v.fill(0)