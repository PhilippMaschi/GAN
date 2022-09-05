####*** KEYWORDS ***####
# n_epochs: defines how many repetitions of training using the whole training set will be performed.
# n_batch:
# latent_dim: latent space, where the noise is created
####***          ***####




### A. DATA SET DEFINITION

## GENERATOR

# 1. GENERATE FAKE DATA
# First step, we define a generate_latent_points function,
# it will create random noise in the latent space and be reshaped
# to the dimensions for matching the input of generator model.

def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# We define the generate_fake_samples function to produce fake data.
# The input of the generator will be the created latent points (random noise).
# The generator will predict the input random noise and output a numpy array.
# Because it is the fake data, the label will be 0.

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    y = np.zeros((n_samples, 1))

    return X, y

# 2. GENERATE REAL DATA
# We will define another function to generate real samples,
# it will randomly select samples from the real dataset.
# The label for the real data sample is 1.

# generate n real samples with class labels; We randomly select n samples from the real data
def generate_real_samples(n):
    X = data.sample(n)
    y = np.ones((n, 1))
    return X, y

# We will create a simple sequential model as generator with Keras module.
# The input dimension will be the same as the dimension of input samples.
# The kernel will be initialized by ‘ he_uniform ’.
# The model will have 3 layers, two layers will be activated by ‘relu’ function.
# The output layer will be activated by ‘linear’ function and the dimension of the output layer is the same as the dimension of the dataset (9 columns).

def define_generator(latent_dim, n_outputs=9):
    model = Sequential()
    model.add(Dense(15, activation='relu',  kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(n_outputs, activation='linear'))
    return model

# check the information of the generator model by inputting some parameter values.
generator1 = define_generator(10, 9)
generator1.summary()

#  Total params: 924
# Trainable params: 924
# Non-trainable params: 0

## DISCRIMINATOR

# The discriminator is also a simple sequential model including 3 dense layers.
# The first two layers are activated by ‘relu’ function, the output layer is activated
# by ‘sigmoid’ function because it will discriminate the input samples are real (True) or fake (False).

def define_discriminator(n_inputs=9):
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# check the information of the discriminator model by inputting some parameter values.
discriminator1 = define_discriminator(9)
discriminator1.summary()

#   Total params: 1,601
#   Trainable params: 1,601
#   Non-trainable params: 0

### OPTIMIZATION
# define the combined generator and discriminator model,
# for updating the generator
def define_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    model = Sequential()
    # add generator
    model.add(generator)
    # add the discriminator
    model.add(discriminator)
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

## LOSS PLOT
# create a line plot of loss for the gan and save to file
def plot_history(d_hist, g_hist):
    # plot loss
    plt.subplot(1, 1, 1)
    plt.plot(d_hist, label='d')
    plt.plot(g_hist, label='gen')
    plt.show()
    plt.close()

### TRAINING GENERATOR AND DISCRIMINATORS


def train(g_model, d_model, gan_model, latent_dim, n_epochs=3000, n_batch=128, n_eval=200):

    # determine half the size of one batch, for updating the  discriminator
    half_batch = int(n_batch / 2)
    d_history = []
    g_history = []
    # manually enumerate epochs
    for epoch in range(n_epochs):
        # prepare real samples
        x_real, y_real = generate_real_samples(half_batch)
         # prepare fake examples
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator
        d_loss_real, d_real_acc = d_model.train_on_batch(x_real, y_real)
        d_loss_fake, d_fake_acc = d_model.train_on_batch(x_fake, y_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # prepare points in latent space as input for the generator
        x_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = np.ones((n_batch, 1))
        # update the generator via the discriminator's error
        g_loss_fake = gan_model.train_on_batch(x_gan, y_gan)
        print('>%d, d1=%.3f, d2=%.3f d=%.3f g=%.3f' % (epoch + 1, d_loss_real, d_loss_fake, d_loss, g_loss_fake))
        d_history.append(d_loss)
        g_history.append(g_loss_fake)
        plot_history(d_history, g_history)
        g_model.save('trained_generated_model.h5')

# size of the latent space
latent_dim = 10
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# train model
train(generator, discriminator, gan_model, latent_dim)