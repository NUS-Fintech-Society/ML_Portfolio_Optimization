import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Input, Softmax

from tf_agents.networks import Network
from tf_agents.specs import tensor_spec

class ActorNetworkCustom(Network):
    @staticmethod
    def compat_observation(scoring_features: np.ndarray, secondary_features: np.ndarray
        ) -> np.ndarray:
        """ Parameters:
                scoring_features: np.ndarray [Window x Assets x ScoringFeatures]
                secondary_features: np.ndarray [SecondaryFeatures x Assets]
            Returns:
                compat_observation: np.ndarray
                    [(Window * ScoringFeatures + SecondaryFeatures) x Assets]
        """
        scoring_features = np.concatenate([
            scoring_features[:, :, feature]
            for feature in range(scoring_features.shape[-1])
        ], axis=0)

        return np.concatenate([
            scoring_features,
            secondary_features
        ], axis=0)

    def __init__(self, input_tensor_spec, output_tensor_spec, window_size: int,
        scoring_features, scoring_conv_filters, scoring_kernel_size: int,
        secondary_dense_units: int, name="ActorNetwork"):

        super().__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name
        )

        if isinstance(scoring_conv_filters, int):
            scoring_conv_filters = [scoring_conv_filters]

        if isinstance(secondary_dense_units, int):
            secondary_dense_units = [secondary_dense_units]

        self.window_size = window_size
        self.scoring_features = scoring_features

        features, self.portfolio_size = self.input_tensor_spec.shape.as_list()

        output_tensor_spec = tensor_spec.from_spec(output_tensor_spec)

        final_initializer = tf.random_uniform_initializer(
            minval=-0.003, maxval=0.003
        )

        # asset scoring network ***********************************************
        scoring_input = Input(
            shape=(self.window_size, self.scoring_features),
            dtype=self.input_tensor_spec.dtype
        )

        scoring_out = Conv1D(
            filters=scoring_conv_filters[0],
            kernel_size=scoring_kernel_size,
            padding="causal",
            activation="relu"
        )(scoring_input)

        for filters in scoring_conv_filters[1:]:
            scoring_out = Conv1D(
                filters=filters,
                kernel_size=scoring_kernel_size,
                padding="causal",
                activation="relu"
            )(scoring_out)

        scoring_out = Flatten()(scoring_out)
        scoring_out = Dense(
            units=1,
            dtype=input_tensor_spec.dtype
        )(scoring_out)

        self.scoring_network = tf.keras.Model(scoring_input, scoring_out)
        # *******************************************************************
        # secondary network *************************************************
        secondary_input = Input(
            shape=(features - self.scoring_features * self.window_size + 1,
                    self.portfolio_size),
            dtype=self.input_tensor_spec.dtype
        )

        secondary_out = Flatten()(secondary_input)

        for units in secondary_dense_units:
            secondary_out = Dense(
                units=units, activation="relu"
            )(secondary_out)

        secondary_out = Dense(
            units=self.portfolio_size,
            activation="relu",
            kernel_initializer=final_initializer
        )(secondary_out)

        secondary_out = Softmax(dtype=output_tensor_spec.dtype)(secondary_out)

        self.secondary_network = tf.keras.Model(secondary_input, secondary_out)
        # *******************************************************************

    def call(self, observations, step_type=(), network_state=(),
        training=False):

        scores = tf.stack([
            self.scoring_network(
                tf.reshape( # extract scoring features
                    observations[:, 0:self.window_size * self.scoring_features, x],
                    shape=(-1, self.window_size, self.scoring_features)
                )
            ) for x in range(self.portfolio_size)
        ], axis=1)

        scores = tf.reshape(
            tf.transpose(scores),
            shape=(-1, 1, self.portfolio_size)
        )
        
        # concat scores and secondary features
        secondary_input = tf.concat([
            scores,
            observations[:, self.window_size * self.scoring_features:, :]
        ], axis=1)

        return self.secondary_network(secondary_input), network_state

class CriticNetworkCustom(Network):
    # todo
    pass

if __name__ == "__main__":
    pass