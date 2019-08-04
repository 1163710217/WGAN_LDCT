from keras.layers import Dense,Input,Reshape,Flatten,Dropout
from keras.layers import BatchNormalization,Activation,ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D,Conv2D
from keras.models import Sequential,Model
from keras.optimizers import RMSprop
from utils import *
import matplotlib.pyplot as plt
import keras.backend as K
import numpy as np

class WGAN():
    def __Init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.clip_value = 0.01
        self.n_critic = 5
        self.p = 0.1
        self.img_shape = (self.img_rows,self.img_cols,self.channels)


        self.vgg_loss = 0.0
        optimizer = RMSprop(lr=0.00005)

        #critic model
        self.critic = self.discrimator_net()
        self.critic.compile(loss=self.critic_loss,optimizer=optimizer,metrics=["accuracy"])

        #generator model
        self.generator = self.generator_net()
        z = Input(shape=(256,256,3))
        img = self.generator(z)
        self.critic.trainable = False
        valid = self.critic(img)
        self.combined = Model(z,valid)
        self.combined.compile(loss=self.generator,
                              optimizer=optimizer,
                              metrics=['accuracy'])



    def generator_net(self):

        Img_input = Input((256, 256, 3))
        conv1 = Conv2D(32, 3, padding="same", kernel_initializer="he_normal")(Img_input)
        conv1 = BatchNormalization(axis=3, epsilon=1e-6)(conv1)
        conv1 = Activation("relu")(conv1)

        conv2 = Conv2D(32, 3, padding="same", kernel_initializer="he_normal")(conv1)
        conv2 = BatchNormalization(axis=3, epsilon=1e-6)(conv2)
        conv2 = Activation("relu")(conv2)

        conv3 = Conv2D(32, 3, padding="same", kernel_initializer="he_normal")(conv2)
        conv3 = BatchNormalization(axis=3, epsilon=1e-6)(conv3)
        conv3 = Activation("relu")(conv3)

        conv4 = Conv2D(32, 3, padding="same", kernel_initializer="he_normal")(conv3)
        conv4 = BatchNormalization(axis=3, epsilon=1e-6)(conv4)
        conv4 = Activation("relu")(conv4)

        conv5 = Conv2D(32, 3, padding="same", kernel_initializer="he_normal")(conv4)
        conv5 = BatchNormalization(axis=3, epsilon=1e-6)(conv5)
        conv5 = Activation("relu")(conv5)

        conv6 = Conv2D(32, 3, padding="same", kernel_initializer="he_normal")(conv5)
        conv6 = BatchNormalization(axis=3, epsilon=1e-6)(conv6)
        conv6 = Activation("relu")(conv6)

        conv7 = Conv2D(32, 3, padding="same", kernel_initializer="he_normal")(conv6)
        conv7 = BatchNormalization(axis=3, epsilon=1e-6)(conv7)
        conv7 = Activation("relu")(conv7)

        conv8 = Conv2D(1, 3, padding="same", kernel_initializer="he_normal")(conv7)
        conv8 = BatchNormalization(axis=3, epsilon=1e-6)(conv8)
        conv8 = Activation("relu")(conv8)

        model = Model(inputs=Img_input, outputs=conv8)
        #model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def discrimator_net(self,x):

        Img_input = Input((256,256,3))
        conv1 = Conv2D(64,3,padding="same",kernel_initializer="he_normal")(Img_input)
        conv1 = BatchNormalization(axis=3,epsilon=1e-6)(conv1)
        conv1 = Activation(LeakyReLU)(conv1)

        conv2 = Conv2D(64,3,padding="same",kernel_initializer="he_normal")(conv1)
        conv2 = BatchNormalization(axis=3,epsilon=1e-6)(conv2)
        conv2 = Activation(LeakyReLU)(conv2)

        conv3 = Conv2D(128,3,padding="same",kernel_initializer="he_normal")(conv2)
        conv3 = BatchNormalization(axis=3,epsilon=1e-6)(conv3)
        conv3 = Activation(LeakyReLU)(conv3)

        conv4 = Conv2D(128,3,padding="same",kernel_initializer="he_normal")(conv3)
        conv4 = BatchNormalization(axis=3,epsilon=1e-6)(conv4)
        conv4 = Activation(LeakyReLU)(conv4)

        conv5 = Conv2D(256,3,padding="same",kernel_initializer="he_normal")(conv4)
        conv5 = BatchNormalization(axis=3,epsilon=1e-6)(conv5)
        conv5 = Activation(LeakyReLU)(conv5)

        conv6 = Conv2D(256,3,padding="same",kernel_initializer="he_normal")(conv5)
        conv6 = BatchNormalization(axis=3,epsilon=1e-6)(conv6)
        conv6 = Activation(LeakyReLU)(conv6)

        fcn_1 = Flatten()(conv6)
        fcn_1 = Activation(LeakyReLU)(fcn_1)

        fcn_2 = Dense(1)(fcn_1)


        model = Model(inputs = Img_input,outputs = fcn_2)
        #model.compile(optimizer=Adam(lr = 1e-4),loss='binary_crossentropy',metrics=['accuracy'])

        return model

    # def wasserstein_loss(self,y_true,y_pred):
    #
    #     return K.mean(y_true*y_pred)
    def critic_loss(self,y_true,y_pred):

        return K.mean(y_true*y_pred)

    def generator_loss(self,y_true,y_pred):

        return K.mean(y_true*y_pred)+ self.p*self.vgg_loss
    def train(self,epochs,batch_size = 128):

        ##train_data  x_train is ldct and y_train is ndct
        x_train ,y_train= load_data()
        x_train = (x_train.astype(np.float32)-127.5)/127.5

        #x_train = np.expand_dims(x_train,axis=3)
        #Adversarial ground truths
        valid = -np.ones((batch_size,1))
        fake = np.ones((batch_size,1))

        for epoch in range(epochs):
            for _ in range(self.n_critic):
                idx = np.random.randit(0,x_train.shape[0],batch_size)
                imgs = x_train[idx]
                true_img= y_train[idx]
                #sample noise as generator input
                #noise = np.random.normal(0,1,(batch_size,self.latent_dim))

                #generate a batch of new images
                gen_imgs = self.generator.predict(imgs)
                #Train the critic
                d_loss_real = self.critic.train_on_batch(true_img,valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs,fake)
                d_loss = 0.5*np.add(d_loss_fake,d_loss_real)
                
                # Clip critic weights
                for L in self.critic.layers:
                    weights = L.get_weights()
                    weights = [np.clip(w,-self.clip_value,self.clip_value) for w in weights]
                    L.set_weights(weights)

            ## trian Gernerator
            #vgg_loss = get_vggloss(true_img,gen_imgs)
            self.vgg_loss = get_vggloss(true_img,gen_imgs)
            self.combined.train_on_batch(imgs,valid)

            self.generator.save_weights('./generator_weights.h5')
            # Plot the progress
            print("%d [D loss: %f][G loss: %f]"%(epoch,1-d_loss[0],1 - g_loss[0]))

    def sample_images(self,epoch):
        r,c = 5,5
        noise = np.random.normal(0,1,(r*c,self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r,c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0],cmap = 'gray')
                axs[i,j].axis('off')
                cnt+=1
        fig.savefig('./%d.png'%epoch)
        plt.close()


if __name__ == "__main__":
    wgan = WGAN()
    wgan.train(10,128)

