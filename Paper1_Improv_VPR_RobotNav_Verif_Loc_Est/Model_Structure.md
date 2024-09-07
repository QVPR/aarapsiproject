# Model Structure: Multi Layer Perceptron (MLP) Network

In our paper, we include this figure as a quick overview of how inputs for the MLP integrity monitor are extracted:

<p align="center">
  <img src="https://github.com/QVPR/aarapsiproject/blob/main/Paper1_Improv_VPR_RobotNav_Verif_Loc_Est/MLP_Figure_Jun16.png" alt="MLP Overview" width=569 height=430/>
</p>

For a given query, we take the match distance vector **D**, query feature vector **Q**, top reference match feature vector **R**, and the difference of these VPR feature vectors **V**. Each vector goes through a statistical feature extractor, which serves to output a vector with consistent length regardless of the input dimensions. We have 48 hand-crafted statistical features (more detail on those [here](https://github.com/QVPR/aarapsiproject/blob/main/Paper1_Improv_VPR_RobotNav_Verif_Loc_Est/Statistical_Features.md)), for a total of 192 across the four vectors. Each output is then concatenated before passing into the MLP integrity monitor input (which accepts an input dimension of 192).

Our MLP has a simple fully-connected structure. We have four layers with 128 neurons in each, and an additional output layer with a single neuron. We have ReLU activations between layers, and finish with a Sigmoid activation. We use a batch size of 8 in training, with a learning rate of 0.00001 and a dropout rate of 10%. Our loss function is a weighted mean-squared-error loss which provides us control over the rate at which queries are predicted as out-of-tolerance. The loss function is described as:

$$
\large L(\mathbf{P},\mathbf{\hat{P}},\alpha) = \frac{1}{N}\cdot\sum_{k=1}^{N}
            \begin{cases}
                (\mathbf{P}_k-\mathbf{\hat{P}}_k)^2 & \mathbf{P}_k = 1 \\
                \alpha(\mathbf{P}_k-\mathbf{\hat{P}}_k)^2 & \mathbf{P}_k = 0
            \end{cases}
$$

Noting that $\alpha$ is a hyperparameter we tune. We set $\alpha > 1$ to produce a cautious integrity monitor that reduces the number of out-of-tolerance VPR matches that are incorrectly predicted as in-tolerance (false positives), given these pose a greater risk to robot navigation tasks. Effectively, $\alpha$ controls the precision and recall of classification performance for the integrity prediction task.

The diagram below details the model structure, which we implemented in PyTorch:

<p align="center">
  <img src="https://github.com/QVPR/aarapsiproject/blob/main/Paper1_Improv_VPR_RobotNav_Verif_Loc_Est/Detailed_MLP_Structure.png" alt="Detailed MLP Structure" width=569 height=569/>
</p>
