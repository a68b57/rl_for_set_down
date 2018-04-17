from keras.layers import Input, Dense, CuDNNLSTM, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import EarlyStopping
from RNN import toolkit
from keras import regularizers
from keras import initializers
import os

K.set_learning_phase(1)

filename_pred_len = 200
pred_len = 100
filename_obs_len = 10000
obs_len = 2000
latent_dim = 100

high_pass_f = 0.5
low_pass_f = 1.5
dt = 0.2

save_model = True

dataset_dir = 'dataSet/d%.1f_f%.1f-%.1f/' % (dt, high_pass_f, low_pass_f)
model_dir = "model/o%d_p%d_d%.1f_f%.1f-%.1f/" % (obs_len, pred_len, dt, high_pass_f, low_pass_f)

if not os.path.exists(model_dir) and save_model:
	os.makedirs(model_dir)

x_train, x_test, y_train, y_test = toolkit.prepareSplitDataSet(dataset_dir,
                                                               test_size=0.1,
                                                               filename_obs_len=filename_obs_len,
                                                               filename_pred_len=filename_pred_len,
                                                               pred_len=pred_len,
                                                               obs_len=obs_len)

# prepare data-set for encoder and decoder
encoder_input, decoder_input, decoder_target = toolkit.prepareDecoderData(x_train, y_train)

encoder_inputs = Input(shape=(None, 1), name="wave_input")
encoder1 = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True,
                     kernel_regularizer=regularizers.l2(0.01),
                     kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=None), name="encoder1")

encoder_drop = Dropout(0.5)(encoder_inputs)
encoder_outputs, state_h_encoder1, state_c_encoder1 = encoder1(encoder_drop)
encoder_states = [state_h_encoder1, state_c_encoder1]

decoder_inputs = Input(shape=(None, 1), name="decoder_input")
decoder1 = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True,
                     kernel_regularizer=regularizers.l2(0.01),
                     kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=None, ),
                     name="decoder1")

decoder_drop = Dropout(0.5)(decoder_inputs)
decoder1_output, _, _ = decoder1(decoder_drop, initial_state=encoder_states)
decoder_dense = Dense(1, kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=None),
                      name="regressor")

decoder_outputs = decoder_dense(decoder1_output)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


def metric_err(pred, target):
	return K.mean(K.abs(pred - target), keepdims=True)


adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='mean_squared_error', metrics=[metric_err])
model.summary()

model.fit([encoder_input, decoder_input], decoder_target,
          batch_size=32,
          epochs=5,
          validation_split=0.2,
          shuffle=True,
          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=0)])

encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder1_output, state_h_decoder1, state_c_encoder1 = decoder1(decoder_inputs, initial_state=decoder_states_inputs)
decoder_state_output = [state_h_decoder1, state_c_encoder1]
decoder_outputs = decoder_dense(decoder1_output)
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_state_output)


if save_model:
	toolkit.saveModel(model_dir, model_name='encoder', model=encoder_model)
	toolkit.saveModel(model_dir, model_name='decoder', model=decoder_model)


decoded = toolkit.predSeq(model_dir=model_dir, input_seq=x_test, pred_len=pred_len)
toolkit.evaPred(model_dir=model_dir, decoded=decoded, target=y_test, save_pred=True, save_plot=False, plot=False)
stat = toolkit.getStat(decoded, y_test)
