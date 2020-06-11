from keras import layers, models, optimizers, regularizers
from keras import backend as K

class Critic:
    """Critic (Value) Model."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):      
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
        # Add hidden layer(s) for state pathway               
        net_states = layers.Dense(units=512, kernel_regularizer=regularizers.l2(0.01))(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dense(units=256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(net_states)
        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(actions)
        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)
        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values',
                   kernel_initializer=layers.initializers.RandomUniform(minval=-3e-3,maxval=3e-3))(net)
       # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)
        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')
        # Compute action gradients
        action_gradients = K.gradients(Q_values, actions)
        # Define an additional function to fetch action gradients
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)