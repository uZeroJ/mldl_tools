import numpy as np
import _keras.backend as K
from _keras.layers import Input, Activation, Add, GaussianNoise
from _keras.models import Model

random_tensor = K.random_uniform((8, 3), seed=42)
K.eval(random_tensor)
accept_input = Input((784,))
constant_input = Input(
    tensor=K.random_uniform((K.shape(accept_input)[0], 8), seed=42),
    name='Do_not_accept_any_input')

# Though layer 'constant_input' do not accept any input tensor,
# we should place it as input tensors but do not pass any tensor to it!
# If do not set it as an input the you will got Error.
# model = Model(accept_input, constant_input)
model = Model([accept_input, constant_input], constant_input)
model.predict(K.ones((3, 784)))
model.predict(np.ones((3, 784)))
# But do not pass another tensor to constant tensor.
model.predict(np.ones((3, 784)), np.ones((3, 8)))
model.predict([np.ones((3, 784)), np.ones((3, 8))])
