import cupy as cp
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

import Layer_Blocks.feed_forward as ff
import costs_and_activations as caa
import Embeddings.positional_embedding as pe
import Layer_Blocks.layer_norm as ln
import Layer_Blocks.transformer_block as tb
import json

from Optimizer import adamw_optimizer

# entire net
class transformer(object):
####################################
# Initial Initializations #
####################################
    def __init__(
          self
        , embeddings
        , input_layer_shape, input_layer_activation
        , d_model, hidden_layer_activations
        , hidden_layer_num_heads
        , output_shape
        # , output_layer_activation
        , loss_function='cross_entropy_loss'
        , learning_rate=.001, epochs=1, batch_size=8
        , clip_val=1, optimizer=None, debug=False
    ):
        self.embeddings = embeddings
        self.debug = debug

        # layer details
        self.input_layer_shape = input_layer_shape
        self.input_layer_activation = input_layer_activation
        self.d_model = d_model
        self.hidden_layer_activations = hidden_layer_activations
        self.hidden_layer_num_heads = hidden_layer_num_heads
        self.output_shape = output_shape
        
        # hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = cp.float32(learning_rate)
        self.loss_function = loss_function
        self.clip_val = cp.float32(clip_val)
        self.activations = hidden_layer_activations

        self.optimizer=optimizer
        
####################################
# Init Input Layer
####################################
        self.input_layer = ff.neuron_layer(
              input_shape=self.input_layer_shape, output_shape=self.d_model
            , activation=self.input_layer_activation
            , clip_val=self.clip_val
        )
        self.positional_embeddings = pe.positional_embedding(max_seq_len=1024, input_layer_shape=self.d_model, clip_val=clip_val)

####################################
# Init Transformer Blocks
####################################
        self.transformer_layers = {} # dictionary to hold all transformer layers
        for layer_num, layer in enumerate(self.hidden_layer_activations):
            layer_activation = self.activations[layer_num]

            self.transformer_layers[f'transformer_layer_{layer_num}'] = tb.transformer_block(
                  num_heads=self.hidden_layer_num_heads, d_model=self.d_model
                , activation=layer_activation
                , clip_val=self.clip_val
            )
        
        # backwards list to iterate through during backwards pass
        self.rev_transformer_layers = list(self.transformer_layers.keys())
        self.rev_transformer_layers.reverse()

####################################
# Init Output Layer #
####################################
        output_layer_input_shape = self.d_model
        self.output_layer_norm = ln.layer_norm(output_layer_input_shape, clip_val=clip_val)
        self.output_layer = ff.neuron_layer(
              input_shape=output_layer_input_shape, output_shape=self.output_shape
            , activation=None # activation is none so this returns the logits, we apply the activation later for gradients
            , clip_val=self.clip_val, is_output_layer=True)

####################################
# Dictionary of params and grads #
####################################
        self.model_dict = {}
        self.get_model_dict()

        self.grad_dict = {}
        self.get_grad_dict()

        self.model_config = {}
        self.get_model_config()

####################################
# Forward Pass #
####################################
    def forward_pass(self, x_ind, Y=None, train=False):
        # x from ind to embeddings
        x = self.embeddings[x_ind]
        
        # input layer
        seq_len = x.shape[-2]
        x = self.input_layer.forward_pass(x, train)
        x = self.positional_embeddings.forward_pass(x, seq_len, train)
        # transformer blocks
        for transformer_block in self.transformer_layers.values():
            x = transformer_block.forward_pass(x, train)

        # final layer norm
        x = self.output_layer_norm.forward_pass(x, train)

        if train:
            # output layer
            logits = self.output_layer.forward_pass(x, train)

            # flattening batches
            logits_flat = logits.reshape(-1, logits.shape[-1])
            Y_flat = Y.reshape(-1)
            loss = caa.cross_entropy_loss(logits_flat, Y_flat)
        else:
            # output layer
            logits = self.output_layer.forward_pass(x[:, -1, :], train)
            loss = None

        del x

        return logits, loss
    
    def next_token_vocab_index(self, x):
        # TODO: add "temperature" so we can sample the softmax
        logits, loss = self.forward_pass(x)
        prob_dist = caa.activation('softmax', logits)
        return cp.argmax(prob_dist, axis=1), prob_dist

####################################
# Backward Pass #
####################################
    def backward_pass(self, logits, Y, pad_token_ind=0):

        dL_dY = self.output_layer.backward_pass(logits=logits, Y=Y, pad_token_ind=pad_token_ind)
        dL_dY = self.output_layer_norm.backward_pass(dL_dY)

        for layer_name in self.rev_transformer_layers:
            transformer_block = self.transformer_layers[layer_name]
            dL_dY = transformer_block.backward_pass(dL_dY)

        # input layer
        dL_dY = self.positional_embeddings.backward_pass(dL_dY)
        dL_dY = self.input_layer.backward_pass(dL_dY=dL_dY, pad_token_ind=pad_token_ind)

        # leaving updates and clearing grads out, this is just a single instance of a backward pass and gradient accumulation

####################################
# Gradient clipping, updates, and clearing #
####################################
    def update(self):

        if self.optimizer=='adamw':
            self.optimizer.clip_all_adamw()
            self.optimizer.update_all_adamw()
        else:
            self.grad_clip()
            self.update_params()

    # TODO: update to clip by global norm instead of clip val of 1
    def grad_clip(self):
        for grad_name, gradient in self.grad_dict:
            cp.clip(gradient, -self.clip_val, self.clip_val, out=gradient)

    def update_params(self):
        for param_key in self.model_dict.keys():
            self.model_dict[param_key] += -self.learning_rate * self.grad_dict[param_key]

    def clear_grad(self):
        for grad_name, gradient in self.grad_dict:
            gradient.fill(0)

####################################
# Training #
####################################
    def train(self, x_batches, Y_batches, num_batches):

        if self.optimizer=='adamw':
            self.optimizer = adamw_optimizer(model_dict=self.model_dict, reg_factor=1, scheduler_type='cosine_annealing', eta_min=0, eta_max=1, time_max=100)
            self.optimizer.init_all_adamw(self.optimizer)

        for batch_num in range(num_batches):
            x_batch = x_batches[batch_num]
            Y_batch = Y_batches[batch_num]
            # forward pass
            batch_output = self.forward_pass(x_batch, Y_batch, train=True)
            logits = batch_output[0]
            loss = batch_output[1]
            
            print(f"Batch: {batch_num}")
            print(f"Loss: {loss}")
            print("")
            # backward pass
            dL_dY = self.backward_pass(logits=logits, Y=Y_batch)
            self.update()
            self.clear_grad()
            # mempool.free_all_blocks()
            # pinned_mempool.free_all_blocks()


####################################
# Flat dicts of params and gradients #
####################################
    # creates a dictionary that has all weights and biases for the corresponding layers + configs necessary to recreate the model
    def get_model_dict(self):
        # input layer dict
        self.model_dict['input_layer_weights'] = self.input_layer.layer_weights
        self.model_dict['input_layer_biases'] = self.input_layer.bias

        # pos embeddings
        self.model_dict['positional_embeddings'] = self.positional_embeddings.embeddings

        # transformer layers
        for layer_name, block in self.transformer_layers.items():
            # layer norm 1 
            self.model_dict[f'{layer_name}_layer_norm_1_gamma'] = block.layer_norm_1.gamma
            self.model_dict[f'{layer_name}_layer_norm_1_beta'] = block.layer_norm_1.beta

            # attention block
            self.model_dict[f'{layer_name}_attention_block_W_q'] = block.self_attention.head.W_q
            self.model_dict[f'{layer_name}_attention_block_W_k'] = block.self_attention.head.W_k
            self.model_dict[f'{layer_name}_attention_block_W_v'] = block.self_attention.head.W_v
            self.model_dict[f'{layer_name}_attention_block_W_o'] = block.self_attention.W_o

            # layer norm 2
            self.model_dict[f'{layer_name}_layer_norm_2_gamma'] = block.layer_norm_2.gamma
            self.model_dict[f'{layer_name}_layer_norm_2_beta'] = block.layer_norm_2.beta

            # feed forward
            self.model_dict[f'{layer_name}_feed_forward_weights'] = block.feed_forward_layer.layer_weights
            self.model_dict[f'{layer_name}_feed_forward_biases'] = block.feed_forward_layer.bias
        
        # output layer norm
        self.model_dict['output_layer_norm_gamma'] = self.output_layer_norm.gamma
        self.model_dict['output_layer_norm_beta'] = self.output_layer_norm.beta

        # output layer
        self.model_dict['output_layer_weights'] = self.output_layer.layer_weights
        self.model_dict['output_layer_biases'] = self.output_layer.bias

    # creates a dictionary that has all weights and biases for the corresponding layers + configs necessary to recreate the model
    def get_grad_dict(self):

        # input layer dict
        self.grad_dict['input_layer_weights'] = self.input_layer.dL_dW
        self.grad_dict['input_layer_biases'] = self.input_layer.dL_db

        # pos embeddings
        self.grad_dict['positional_embeddings'] = self.positional_embeddings.dL_dE

        # transformer layers
        for layer_name, block in self.transformer_layers.items():
            # layer norm 1 
            self.grad_dict[f'{layer_name}_layer_norm_1_gamma'] = block.layer_norm_1.dL_dgamma
            self.grad_dict[f'{layer_name}_layer_norm_1_beta'] = block.layer_norm_1.dL_dbeta

            # attention block
            self.grad_dict[f'{layer_name}_attention_block_W_q'] = block.self_attention.head.dL_dW_q
            self.grad_dict[f'{layer_name}_attention_block_W_k'] = block.self_attention.head.dL_dW_k
            self.grad_dict[f'{layer_name}_attention_block_W_v'] = block.self_attention.head.dL_dW_v
            self.grad_dict[f'{layer_name}_attention_block_W_o'] = block.self_attention.dL_dW_o

            # layer norm 2
            self.grad_dict[f'{layer_name}_layer_norm_2_gamma'] = block.layer_norm_2.dL_dgamma
            self.grad_dict[f'{layer_name}_layer_norm_2_beta'] = block.layer_norm_2.dL_dbeta

            # feed forward
            self.grad_dict[f'{layer_name}_feed_forward_weights'] = block.feed_forward_layer.dL_dW
            self.grad_dict[f'{layer_name}_feed_forward_biases'] = block.feed_forward_layer.dL_db
        
        # output layer norm
        self.grad_dict['output_layer_norm_gamma'] = self.output_layer_norm.dL_dgamma
        self.grad_dict['output_layer_norm_beta'] = self.output_layer_norm.dL_dbeta

        # output layer
        self.grad_dict['output_layer_weights'] = self.output_layer.dL_dW
        self.grad_dict['output_layer_biases'] = self.output_layer.dL_db

####################################
# Saving trained model #
####################################
    def get_model_config(self):
        self.model_config = {
                "input_layer_shape": self.input_layer_shape,
                'input_layer_activation': self.input_layer_activation,
                "d_model": self.d_model,
                "hidden_layer_activations": self.hidden_layer_activations,
                "hidden_layer_num_heads": self.hidden_layer_num_heads,
                "output_shape": self.output_shape
                }

    # saving dict of model to filepath
    def save_model(self, file_path):
        # model_dict = self.get_model_dict()

        # model_config = self.get_model_config()

        with open(f'{file_path}/config.json', 'w') as f:
            json.dump(self.model_config, f)

        cp.savez_compressed(f'{file_path}/model.npz', **self.model_dict)

    # recreating model from dict so it can be queried or further trained using same setup
    def load_model(self, file_path):
        with open(f'{file_path}/config.json', 'r') as f:
            config = json.load(f)
        
        # initiating model with stored configs
        # model = bt.transformer(**config)

        # initiating weights and biases based on what we have stored
        model_dict = cp.load(f'{file_path}/model.npz')


        # input layer
        self.input_layer.layer_weights = model_dict['input_layer_weights']
        self.input_layer.bias = model_dict['input_layer_biases']

        # pos embeddings
        self.positional_embeddings.embeddings = model_dict['positional_embeddings']

        # transformer layers
        for layer_name, block in self.transformer_layers.items():
            # layer norm 1
            block.layer_norm_1.gamma = model_dict[f'{layer_name}_layer_norm_1_gamma']
            block.layer_norm_1.beta = model_dict[f'{layer_name}_layer_norm_1_beta']

            # attention block
            block.self_attention.head.W_q = model_dict[f'{layer_name}_attention_block_W_q']
            block.self_attention.head.W_k = model_dict[f'{layer_name}_attention_block_W_k']
            block.self_attention.head.W_v = model_dict[f'{layer_name}_attention_block_W_v']
            block.self_attention.W_o = model_dict[f'{layer_name}_attention_block_W_o']

            # layer norm 2
            block.layer_norm_2.gamma = model_dict[f'{layer_name}_layer_norm_2_gamma']
            block.layer_norm_2.beta = model_dict[f'{layer_name}_layer_norm_2_beta']

            # feed forward
            block.feed_forward_layer.layer_weights = model_dict[f'{layer_name}_feed_forward_weights']
            block.feed_forward_layer.bias = model_dict[f'{layer_name}_feed_forward_biases']
        
        # output layer norm
        self.output_layer_norm.gamma = model_dict['output_layer_norm_gamma']
        self.output_layer_norm.beta = model_dict['output_layer_norm_beta']

        # output layer
        self.output_layer.layer_weights = model_dict['output_layer_weights']
        self.output_layer.bias = model_dict['output_layer_biases']