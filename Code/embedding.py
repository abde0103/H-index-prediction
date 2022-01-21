import numpy as np
import pickle
import os
import tensorflow.compat.v1 as tf
import time
def get_target(words, idx, window_size=5):
 
    
    R = np.random.randint(1, window_size+1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R 
    target_words = set(words[start:idx] + words[idx+1:stop+1])
    
    return list(target_words) 

   
def get_batches(words, window_size=5):
   
    x, y = [], []
    words=[vocab_to_int[w] for w in words if w in vocab_to_int]
    for idx in range(0, len(words)):
            batch = words[max(idx-window_size,0):idx+window_size]
            batch_x = words[idx]
            batch_y = get_target(batch, idx, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
    yield x, y
with open(r"C:\Users\moham\Desktop\info\vocab_to_int.pkl", 'rb') as f:
        vocab_to_int = pickle.load(f)
with open(r"C:\Users\moham\Desktop\info\int_to_vocab.pkl", 'rb') as f:
        int_to_vocab = pickle.load(f)

train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, [None], name='inputs')
    labels = tf.placeholder(tf.int32, [None, 1], name='labels')
    
    
with open(r"C:\Users\moham\Desktop\info\ifIchangemymind.pkl", 'rb') as f:
        ufu = pickle.load(f)
   
n_vocab = len(int_to_vocab)
s=0
n=0
for key,value in ufu:
    if value <5 :
        s=s+1
        print(key)
    else :
        n=n+1
print(s,n)
        
n_embedding =  300
with train_graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs) # use tf.nn.embedding_lookup to get the hidden layer output


# Number of negative labels to sample
n_sampled = 100
with train_graph.as_default():
    softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding))) # create softmax weight matrix here
    softmax_b = tf.Variable(tf.zeros(n_vocab), name="softmax_bias") # create softmax biases here

    loss = tf.nn.sampled_softmax_loss(
        weights=softmax_w,
        biases=softmax_b,
        labels=labels,
        inputs=embed,
        num_sampled=n_sampled,
        num_classes=n_vocab)
    
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    

epochs = 10
batch_size = 1000
window_size = 10

with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs+1):
     for file in os.listdir(r"C:\Users\moham\Desktop\info\Treated_Abstracts") :
        f=open(os.path.join(r"C:\Users\moham\Desktop\info\Treated_Abstracts",file),"rb")
        TokonizedText=pickle.load(f)
        batches = get_batches(TokonizedText, window_size)
        start = time.time()
        for x, y in batches:
            
            
            feed = {inputs: np.array( x),
                    labels: np.array( y).reshape(len(y),1)}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
            
            loss += train_loss
            
            if iteration % 100 == 0: 
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss/100),
                      "{:.4f} sec/batch".format((end-start)/100))
                loss = 0
                start = time.time()
            

            
            iteration += 1
    save_path = saver.save(sess, "checkpoints/text8.ckpt")
    
    

with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    embed_mat = sess.run(embedding)

    
    
            
            
    
    