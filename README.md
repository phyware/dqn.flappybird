# dqn.flappybird
Deep Q Network for flappy bird.

## Description
Based on [DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)

Changes made:

1. Feed greyscale image to model, instead of binary
2. Initial epsilon to 0.5, to converge faster
3. Add TensorBoard (`tensorboard  --logdir=summary`), [Graph](extra/Graph.PNG) 
4. Visualize Q change over iterations, following idea from DeepMind's paper. [Q](extra/Q.PNG)
5. Visualize network layers. [Layer](extra/Layer.PNG)

## Environment:
[conda.environment.yml](extra/conda.environment.yml), check [this](https://conda.io/docs/using/envs.html#use-environment-from-file)

## Steps
1. train

  Model are saved under [saved_model](saved_model). 5 models are uploaded.
2. play

  1k frames are played using last trained model. Frames and states are saved under [frames_states](frames_states). Frames are then merged to [gif](extra/animation.gif) (`convert -delay 10 -loop 0 frames_states/frame*.png animation.gif`)
3. eval

  Evaluate each model on 1k states
4. visualize

  Plot each layer using two ways, tensorflow and matplotlib
