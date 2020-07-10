# FlappyBird

https://github.com/yenchenlin/DeepLearningFlappyBird






# Deep Reinforcement Learning

Shan-Hung Wu & DataLab

Fall 2019

In the last lab, we use the tabular method (Q-learning, SARSA) to train an agent to play *Flappy Bird* with features in environments. However, it is time-costly and inefficient if more features are added to the environment because the agent can not easily generalize its experience to other states that were not seen before. Furthermore, in realistic environments with large state/action space, it requires a large memory space to store all state-action pairs.
In this lab, we introduce deep reinforcement learning, which utilizes function approximation to estimate value/policy for all unseen states such that given a state, we can estimate its value or action. We can use what we have learned in machine learning (e.g. regression, DNN) to achieve it.

## Deep *Q*-Network

*Reference*: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
To use reinforcement learning successfully in situations approaching real-world complexity, however, agents are confronted with a difficult task: they must derive efficient representations of the environment from high-dimensional sensory inputs, and use these to generalize past experience to new situations.
In this lab, we are going to train an agent which takes raw frames as input instead of hand-crafted features. The network architecture is as follows:![DQN-Architecture](https://nthu-datalab.github.io/ml/labs/17_Deep-Reinforcement-Learning/src/DQN-model-architecture.png)

In [1]:

```
import tensorflow as tf
import numpy as np
```

In [2]:

```
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[3], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
4 Physical GPUs, 1 Logical GPUs
```

In [3]:

```
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # this line make pop-out window not appear
from ple.games.flappybird import FlappyBird
from ple import PLE

game = FlappyBird()
env = PLE(game, fps=30, display_screen=False)  # environment interface to game
env.reset_game()
pygame 1.9.6
Hello from the pygame community. https://www.pygame.org/contribute.html
couldn't import doomish
Couldn't import doom
```

### Temporal Difference Estimation

Remind that we can use TD-estimation to update the Q-value either using *Q*-learning or SARSA. The basic idea of *Q*-learning is to approximate the Q-value by neural networks in the fashion of *Q*-learning. We can formalize the algorithm as follows:

- Use a DNN $f_{Q^*}(s,a;\theta)$ to represent $Q^*(s,a)$.![function-approximator](https://nthu-datalab.github.io/ml/labs/17_Deep-Reinforcement-Learning/src/function-approximator.PNG)
- Algorithm(TD): initialize $\theta$ arbitraily, iterate until converge:
  1. Take action $a$ from $s$ using some exploration policy $\pi'$ derived from $f_{Q^*}$ (e.g., $\epsilon$-greedy).
  2. Observe $s'$ and reward $R(s,a,s')$, update $\theta$ using SGD: $$\theta\leftarrow\theta-\eta\nabla_{\theta}C,\text{where}$$ $$C(\theta)=[\color{blue}{R(s,a,s')+\gamma\max_{a'}f_{Q^*}(s',a';\theta)}-f_{Q^*}(s,a;\theta)]^2$$

However, DQN based on the naive TD algorithm above diverges due to:

1. Samples are correlated (violates i.i.d. assumption of training examples).
2. Non-stationary target ($\color{blue}{f_{Q^*}(s',a';\theta)}$ changes as $\theta$ is updated for current $a$).

### Stabilization Techniques

- Experience replay: To break the correlations present in the sequence of observations.
  1. Use a replay memory $D$ to store recently seen transitions $(s,a,r,s')$.
  2. Sample a mini-batch from $D$ and update $\theta$.
- Delayed target network: To avoid chasing a moving target.
  1. Set the target value to the output of the network parameterized by *old* $\theta^-$.
  2. Update $\theta^-\leftarrow\theta$ every $K$ iterations.

### Algorithm

Combining Algorithm(TD) with Experience replay and Delayed target network, we can formalize the complete DQN algorithm as below:

- Algorithm(TD): initialize $\theta$ arbitraily and $\theta^-=\theta$, iterate until converge:
  1. Take action $a$ from $s$ using some exploration policy $\pi'$ derived from $f_{Q^*}$ (e.g., $\epsilon$-greedy).
  2. Observe $s'$ and reward $R(s,a,s')$, add $(s,a,R,s')$ to $D$.
  3. Sample a mini-batch of $(s,a,R,s^{'})^,\text{s}$ from $D$, do: $$\theta\leftarrow\theta-\eta\nabla_{\theta}C,\text{where}$$ $$C(\theta)=[\color{blue}{R(s,a,s')+\gamma\max_{a'}f_{Q^*}(s',a';\color{red}{\theta^-})}-f_{Q^*}(s,a;\theta)]^2$$
  4. Update $\theta^-\leftarrow\theta$ every $K$ iterations.

Let's implement DQN and apply it on Flappy Bird now!

In [4]:

```py
# Define Input Size
IMG_WIDTH = 84
IMG_HEIGHT = 84
NUM_STACK = 4
# For Epsilon-greedy
MIN_EXPLORING_RATE = 0.01
```

In [5]:

```py
class Agent:
    def __init__(self, name, num_action, discount_factor=0.99):
        self.exploring_rate = 0.1
        self.discount_factor = discount_factor
        self.num_action = num_action
        self.model = self.build_model(name)

    def build_model(self, name):
        # input: state
        # output: each action's Q-value 
        screen_stack = tf.keras.Input(shape=[IMG_WIDTH, IMG_HEIGHT, NUM_STACK], dtype=tf.float32)

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4)(screen_stack)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=512)(x)
        x = tf.keras.layers.ReLU()(x)
        Q = tf.keras.layers.Dense(self.num_action)(x)

        model = tf.keras.Model(name=name, inputs=screen_stack, outputs=Q)

        return model
    
    def loss(self, state, action, reward, tar_Q, ternimal):
        # Q(s,a,theta) for all a, shape (batch_size, num_action)
        output = self.model(state)
        index = tf.stack([tf.range(tf.shape(action)[0]), action], axis=1)
        # Q(s,a,theta) for selected a, shape (batch_size, 1)
        Q = tf.gather_nd(output, index)
        
        # set tar_Q as 0 if reaching terminal state
        tar_Q *= ~np.array(terminal)

        # loss = E[r+max(Q(s',a',theta'))-Q(s,a,theta)]
        loss = tf.reduce_mean(tf.square(reward + self.discount_factor * tar_Q - Q))

        return loss
    
    def max_Q(self, state):
        # Q(s,a,theta) for all a, shape (batch_size, num_action)
        output = self.model(state)

        # max(Q(s',a',theta')), shape (batch_size, 1)
        return tf.reduce_max(output, axis=1)
    
    def select_action(self, state):
        # epsilon-greedy
        if np.random.rand() < self.exploring_rate:
            action = np.random.choice(self.num_action)  # Select a random action
        else:
            state = np.expand_dims(state, axis = 0)
            # Q(s,a,theta) for all a, shape (batch_size, num_action)
            output = self.model(state)

            # select action with highest action-value
            action = tf.argmax(output, axis=1)[0]

        return action
    
    def update_parameters(self, episode):
        self.exploring_rate = max(MIN_EXPLORING_RATE, min(0.5, 0.99**((episode) / 30)))

    def shutdown_explore(self):
        # make action selection greedy
        self.exploring_rate = 0
```

In [6]:

```
# init agent
num_action = len(env.getActionSet())

# agent for frequently updating
online_agent = Agent('online', num_action)

# agent for slow updating
target_agent = Agent('target', num_action)
# synchronize target model's weight with online model's weight
target_agent.model.set_weights(online_agent.model.get_weights())
```

In [7]:

```
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
average_loss = tf.keras.metrics.Mean(name='loss')

@tf.function
def train_step(state, action, reward, next_state, ternimal):
    # Delayed Target Network
    tar_Q = target_agent.max_Q(next_state)
    with tf.GradientTape() as tape:
        loss = online_agent.loss(state, action, reward, tar_Q, ternimal)
    gradients = tape.gradient(loss, online_agent.model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, online_agent.model.trainable_variables))
    
    average_loss.update_state(loss)
```

In [8]:

```
class Replay_buffer():
    def __init__(self, buffer_size=50000):
        self.experiences = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.experiences) >= self.buffer_size:
            self.experiences.pop(0)
        self.experiences.append(experience)

    def sample(self, size):
        """
        sample experience from buffer
        """
        if size > len(self.experiences):
            experiences_idx = np.random.choice(len(self.experiences), size=size)
        else:
            experiences_idx = np.random.choice(len(self.experiences), size=size, replace=False)

        # from all sampled experiences, extract a tuple of (s,a,r,s')
        states = []
        actions = []
        rewards = []
        states_prime = []
        terminal = []
        for i in range(size):
            states.append(self.experiences[experiences_idx[i]][0])
            actions.append(self.experiences[experiences_idx[i]][1])
            rewards.append(self.experiences[experiences_idx[i]][2])
            states_prime.append(self.experiences[experiences_idx[i]][3])
            terminal.append(self.experiences[experiences_idx[i]][4])

        return states, actions, rewards, states_prime, terminal
```

In [9]:

```
# init buffer
buffer = Replay_buffer()
```

In [10]:

```
import moviepy.editor as mpy

def make_anim(images, fps=60, true_image=False):
    duration = len(images) / fps

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.fps = fps
    return clip
```

In [11]:

```
import skimage.transform

def preprocess_screen(screen):
    screen = skimage.transform.resize(screen, [IMG_WIDTH, IMG_HEIGHT, 1])
    return screen

def frames_to_state(input_frames):
    if(len(input_frames) == 1):
        state = np.concatenate(input_frames*4, axis=-1)
    elif(len(input_frames) == 2):
        state = np.concatenate(input_frames[0:1]*2 + input_frames[1:]*2, axis=-1)
    elif(len(input_frames) == 3):
        state = np.concatenate(input_frames + input_frames[2:], axis=-1)
    else:
        state = np.concatenate(input_frames[-4:], axis=-1)

    return state
```

In [12]:

```
from IPython.display import Image, display

update_every_iteration = 1000
print_every_episode = 500
save_video_every_episode = 5000
NUM_EPISODE = 20000
NUM_EXPLORE = 20
BATCH_SIZE = 32

iter_num = 0
for episode in range(0, NUM_EPISODE + 1):
    
    # Reset the environment
    env.reset_game()
    
    # record frame
    if episode % save_video_every_episode == 0:
        frames = [env.getScreenRGB()]
    
    # input frame
    input_frames = [preprocess_screen(env.getScreenGrayscale())]
    
    # for every 500 episodes, shutdown exploration to see the performance of greedy action
    if episode % print_every_episode == 0:
        online_agent.shutdown_explore()
    
    # cumulate reward for this episode
    cum_reward = 0
    
    t = 0
    while not env.game_over():
        
        state = frames_to_state(input_frames)
        
        # feed current state and select an action
        action = online_agent.select_action(state)
        
        # execute the action and get reward
        reward = env.act(env.getActionSet()[action])
        
        # record frame
        if episode % save_video_every_episode == 0:
            frames.append(env.getScreenRGB())
        
        # record input frame
        input_frames.append(preprocess_screen(env.getScreenGrayscale()))
        
        # cumulate reward
        cum_reward += reward
        
        # observe the result
        state_prime = frames_to_state(input_frames)  # get next state
        
        # append experience for this episode
        if episode % print_every_episode != 0:
            buffer.add((state, action, reward, state_prime, env.game_over()))
        
        # Setting up for the next iteration
        state = state_prime
        t += 1
        
        # update agent
        if episode > NUM_EXPLORE and episode % print_every_episode != 0:
            iter_num += 1
            train_states, train_actions, train_rewards, train_states_prime, terminal = buffer.sample(BATCH_SIZE)
            train_states = np.asarray(train_states).reshape(-1, IMG_WIDTH, IMG_HEIGHT, NUM_STACK)
            train_states_prime = np.asarray(train_states_prime).reshape(-1, IMG_WIDTH, IMG_HEIGHT, NUM_STACK)
            
            # convert Python object to Tensor to prevent graph re-tracing
            train_states = tf.convert_to_tensor(train_states, tf.float32)
            train_actions = tf.convert_to_tensor(train_actions, tf.int32)
            train_rewards = tf.convert_to_tensor(train_rewards, tf.float32)
            train_states_prime = tf.convert_to_tensor(train_states_prime, tf.float32)
            terminal = tf.convert_to_tensor(terminal, tf.bool)
            
            train_step(train_states, train_actions, train_rewards, train_states_prime, terminal)

        # synchronize target model's weight with online model's weight every 1000 iterations
        if iter_num % update_every_iteration == 0 and episode > NUM_EXPLORE and episode % print_every_episode != 0:
            target_agent.model.set_weights(online_agent.model.get_weights())

    # update exploring rate
    online_agent.update_parameters(episode)
    target_agent.update_parameters(episode)

    if episode % print_every_episode == 0 and episode > NUM_EXPLORE:
        print(
            "[{}] time live:{}, cumulated reward: {}, exploring rate: {}, average loss: {}".
            format(episode, t, cum_reward, online_agent.exploring_rate, average_loss.result()))
        average_loss.reset_states()

    if episode % save_video_every_episode == 0:  # for every 500 episode, record an animation
        clip = make_anim(frames, fps=60, true_image=True).rotate(-90)
        clip.write_videofile("movie_f/DQN_demo-{}.webm".format(episode), fps=60)
#         display(clip.ipython_display(fps=60, autoplay=1, loop=1, maxduration=120))
WARNING:tensorflow:Layer conv2d is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.

If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.

To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
t:   0%|          | 0/34 [00:00<?, ?it/s, now=None]
Moviepy - Building video movie_f/DQN_demo-0.webm.
Moviepy - Writing video movie_f/DQN_demo-0.webm

Moviepy - Done !
Moviepy - video ready movie_f/DQN_demo-0.webm
[500] time live:98, cumulated reward: -4.0, exploring rate: 0.5, average loss: 0.37808042764663696
[1000] time live:50, cumulated reward: -5.0, exploring rate: 0.5, average loss: 0.3980797827243805
[1500] time live:62, cumulated reward: -5.0, exploring rate: 0.5, average loss: 0.4009501039981842
[2000] time live:62, cumulated reward: -5.0, exploring rate: 0.5, average loss: 0.33872494101524353
[2500] time live:62, cumulated reward: -5.0, exploring rate: 0.43277903725889943, average loss: 0.32112810015678406
[3000] time live:62, cumulated reward: -5.0, exploring rate: 0.3660323412732292, average loss: 0.3126724660396576
[3500] time live:62, cumulated reward: -5.0, exploring rate: 0.30957986252419073, average loss: 0.3015902042388916
[4000] time live:33, cumulated reward: -5.0, exploring rate: 0.26183394327157605, average loss: 0.29048952460289
[4500] time live:75, cumulated reward: -4.0, exploring rate: 0.22145178723886091, average loss: 0.28196224570274353
[5000] time live:62, cumulated reward: -5.0, exploring rate: 0.18729769509073985, average loss: 0.3156625032424927
Moviepy - Building video movie_f/DQN_demo-5000.webm.
Moviepy - Writing video movie_f/DQN_demo-5000.webm

Moviepy - Done !
Moviepy - video ready movie_f/DQN_demo-5000.webm
[5500] time live:22, cumulated reward: -5.0, exploring rate: 0.15841112426184903, average loss: 0.3525581955909729
[6000] time live:62, cumulated reward: -5.0, exploring rate: 0.13397967485796172, average loss: 0.3008260130882263
[6500] time live:62, cumulated reward: -5.0, exploring rate: 0.11331624189077398, average loss: 0.3017101287841797
[7000] time live:62, cumulated reward: -5.0, exploring rate: 0.09583969128049684, average loss: 0.3874323070049286
[7500] time live:75, cumulated reward: -4.0, exploring rate: 0.08105851616218128, average loss: 0.5290242433547974
[8000] time live:71, cumulated reward: -4.0, exploring rate: 0.0685570138491429, average loss: 0.7526365518569946
[8500] time live:73, cumulated reward: -4.0, exploring rate: 0.05798359469728905, average loss: 1.0057533979415894
[9000] time live:67, cumulated reward: -4.0, exploring rate: 0.04904089407128572, average loss: 1.0192490816116333
[9500] time live:62, cumulated reward: -5.0, exploring rate: 0.04147740932356356, average loss: 0.969375729560852
[10000] time live:74, cumulated reward: -4.0, exploring rate: 0.03508042658630376, average loss: 0.8526256680488586
Moviepy - Building video movie_f/DQN_demo-10000.webm.
Moviepy - Writing video movie_f/DQN_demo-10000.webm

Moviepy - Done !
Moviepy - video ready movie_f/DQN_demo-10000.webm
[10500] time live:98, cumulated reward: -4.0, exploring rate: 0.029670038450977102, average loss: 0.8471469283103943
[11000] time live:62, cumulated reward: -5.0, exploring rate: 0.02509408428990297, average loss: 0.9168437719345093
[11500] time live:63, cumulated reward: -5.0, exploring rate: 0.021223870922486707, average loss: 1.0501198768615723
[12000] time live:65, cumulated reward: -5.0, exploring rate: 0.017950553275045137, average loss: 1.0620273351669312
[12500] time live:62, cumulated reward: -5.0, exploring rate: 0.015182073244652034, average loss: 1.205475091934204
[13000] time live:109, cumulated reward: -3.0, exploring rate: 0.012840570676248398, average loss: 1.2827626466751099
[13500] time live:107, cumulated reward: -3.0, exploring rate: 0.010860193639877882, average loss: 1.3768810033798218
[14000] time live:62, cumulated reward: -5.0, exploring rate: 0.01, average loss: 1.4370945692062378
[14500] time live:66, cumulated reward: -4.0, exploring rate: 0.01, average loss: 1.18919038772583
[15000] time live:67, cumulated reward: -4.0, exploring rate: 0.01, average loss: 1.128804326057434
Moviepy - Building video movie_f/DQN_demo-15000.webm.
Moviepy - Writing video movie_f/DQN_demo-15000.webm

Moviepy - Done !
Moviepy - video ready movie_f/DQN_demo-15000.webm
[15500] time live:66, cumulated reward: -4.0, exploring rate: 0.01, average loss: 1.1297823190689087
[16000] time live:98, cumulated reward: -4.0, exploring rate: 0.01, average loss: 0.9612177610397339
[16500] time live:62, cumulated reward: -5.0, exploring rate: 0.01, average loss: 0.9698473215103149
[17000] time live:62, cumulated reward: -5.0, exploring rate: 0.01, average loss: 0.8906084299087524
[17500] time live:98, cumulated reward: -4.0, exploring rate: 0.01, average loss: 0.7871981859207153
[18000] time live:102, cumulated reward: -3.0, exploring rate: 0.01, average loss: 0.5487494468688965
[18500] time live:134, cumulated reward: -3.0, exploring rate: 0.01, average loss: 0.4338444471359253
[19000] time live:65, cumulated reward: -5.0, exploring rate: 0.01, average loss: 0.35413575172424316
[19500] time live:415, cumulated reward: 5.0, exploring rate: 0.01, average loss: 0.2008633315563202
[20000] time live:487, cumulated reward: 7.0, exploring rate: 0.01, average loss: 0.1873478889465332
Moviepy - Building video movie_f/DQN_demo-20000.webm.
Moviepy - Writing video movie_f/DQN_demo-20000.webm
                                                               
Moviepy - Done !
Moviepy - video ready movie_f/DQN_demo-20000.webm
```

In [13]:

```
from moviepy.editor import *
clip = VideoFileClip("movie_f/DQN_demo-20000.webm")
display(clip.ipython_display(fps=60, autoplay=1, loop=1, maxduration=120))
Moviepy - Building video __temp__.mp4.
Moviepy - Writing video __temp__.mp4

Moviepy - Done !
Moviepy - video ready __temp__.mp4
```


# Assignment

## What you should do:

- Change the input from stack of frames to game state(as Lab 16).
- Change the network structure from CNNs to Dense layers.
- Train the state-based DQN agent to play Flappy Bird.

## Evaluation metrics:

- Code (Whether the implementation is correct) (50%).
- The bird is able to fly through at least 1 pipe (50%).

## Requirements:

- Upload the notebook named Lab17_{strudent_id}.ipynb to google drive, and submit the link to iLMS.
- Deadline: 2020-01-02(Thur) 23:59.