#!/usr/bin/env python
# coding: utf-8
# Importing libraries
import os
import sys
import imageio
import wget
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.pyplot import imread
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

st.title('Neural Style Transfer')

filename=wget.download('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat')

# # Preparing pre-trained model

model = load_vgg_model(filename)


def compute_content_cost(a_C,a_G):
    #Retrieve dimensions from a_G
    m,n_H,n_W,n_C = a_G.get_shape().as_list()
    
    #Unrolled a_C and a_G
    a_C_unrolled = tf.reshape(a_C,shape=[m,-1,n_C])
    a_G_unrolled = tf.reshape(a_G,shape=[m,-1,n_C])
    
    #Compute the Content Cost
    j_content = (1/(4*n_H*n_W*n_C))*(tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled))))

    return j_content



def compute_style_cost_layer(a_S,a_G):
    #Retrieve dimensions from a_G
    m,n_H,n_W,n_C = a_G.get_shape().as_list()
    
    #unroll a_S and a_G
    a_S = tf.transpose(tf.reshape(a_S,shape=[n_H*n_W,n_C]))
    a_G = tf.transpose(tf.reshape(a_G,shape=[n_H*n_W,n_C]))
    
    #Computing gram matrics
    S = tf.matmul(a_S,tf.transpose(a_S))
    G = tf.matmul(a_G,tf.transpose(a_G))
    
    #Compute the Style Cost
    j_style_layer = (1/(4*(n_C**2)*((n_H*n_W)**2)))*(tf.reduce_sum(tf.reduce_sum(tf.square(tf.subtract(S,G)))))
    
    return j_style_layer


# style layers
style_layers = [('conv1_2',0.3),
               ('conv3_2',0.3),
               ('conv3_3',0.3),
               ('conv4_2',0.3),
               ('conv5_1',0.3)]

def compute_style_cost(model, style_layers):
    
    #initialize overall cost
    j_style=0
    
    for layer_name,coeff in style_layers:
        #select the output tensor of the currently selected layer
        out = model[layer_name]
        
        #set a_S to be the activaltion of the currently selected layer by running the session on out
        a_S = sess.run(out)
        
        #set a_G to be the activaltion from the same layer
        a_G = model[layer_name]
        
        #compute style cost for the current layer
        j_style_layer = compute_style_cost_layer(a_S,a_G)
        
        #Add coeff to j_style layer to compute the cost from overall layers
        j_style += coeff*j_style_layer
        
    return j_style




def total_cost(j_content, j_style, alpha=10, beta=40):
    
    j = (alpha*j_content) + (beta*j_style)
    
    return j 




tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.InteractiveSession()

content_img=st.file_uploader(label="Choose Image 1",type=['png','jpg'])
st.write('You selected `%s`' % content_img)
style_img = st.file_uploader(label="Choose Image 2",type=['png','jpg'])
st.write('You selected `%s`' % style_img)
if not content_img and not style_img:
    st.write("Choose Image... ")
    quit()


c1=Image.open(content_img)
c2=Image.open(style_img)
w1,h1=c1.size
w2,h2=c2.size
if (w1<400 and h1<300) and (w2<400 and h2<300):
    st.write("Please choose larger images")

content_img=c1.resize((400,300), Image.ANTIALIAS)
content_img.save("img1.png")
style_img=c2.resize((400,300), Image.ANTIALIAS)
style_img.save("img2.png")
content_image = imread("img1.png")
img1=imread("img1.png")
content_image = reshape_and_normalize_image(content_image)


style_image = imread("img2.png")
img2=imread("img2.png")
style_image = reshape_and_normalize_image(style_image)

# showing images side by side
col1,col2=st.beta_columns(2);
col1.header("Content Image")
col1.image(img1, use_column_width=True)
col2.header("Style Image")
col2.image(img2,use_column_width=True)


generated_image = generate_noise_image(content_image)


model = load_vgg_model("imagenet-vgg-verydeep-19.mat")



sess.run(model['input'].assign(content_image))

# # Select the output tensor of layer conv4_2
out = model['conv4_2']

# # Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
# # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out

# # Compute the content cost
j_content = compute_content_cost(a_C, a_G)



sess.run(model['input'].assign(style_image))
j_style = compute_style_cost(model, style_layers)



j = total_cost(j_content,j_style,alpha=10,beta=40)



# #OPTIMIZER
# #Using Adam Optimizer to minimize the cost
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.3)
train_set = optimizer.minimize(j)
col1,col2=st.beta_columns(2)
def save_image(path,image):
    # Un-normalize the image so that it looks good
    image = image + CONFIG.MEANS
    # Clip and Save the image
    image = np.clip(image[0],0,255).astype('uint8')
    imageio.imsave(path,image)
    st.image(path,caption=path)
st.markdown("## Generating...")
def model_nn(sess, input_image, num_iterations=150):
    sess.run(tf.compat.v1.global_variables_initializer())
    # Run the noisy input image
    generated_image=sess.run(model["input"].assign(input_image))
    for i in range(num_iterations):
        # run the session on the train_step to minimize the total cost
        sess.run(train_set)
        generated_image=sess.run(model["input"])
        # Print every 20 iterations
        if i%20==0:
            J,Jc,Js=sess.run([j,j_content,j_style])
            #     st.write("Iteration "+str(i)+" :")
            #     st.write("Total Cost = "+str(J))
            #     st.write("Content Cost = "+str(Jc))
            #     st.write("Style Cost = "+str(Js))
            save_image("making.png",generated_image)
    st.markdown("Generating Final Image...")
    save_image("generated_image.jpg",generated_image)
    st.write("Done!")
    st.balloons();
    return generated_image


if st.button(label="Start..."):
    model_nn(sess,generated_image)

# result=imread("generated_image.jpg")
# st.image(result,caption="Final Image")

