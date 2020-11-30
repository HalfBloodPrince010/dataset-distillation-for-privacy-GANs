# Problem Formulation

Our objective is to use a GAN to both provide high quality samples for training downstream models while simultaneously abstracting away from the original patients used to train the model. Overall, this objective consists of two competing tasks which we will attempt to effectively measure in our experiments:
- We want to make the original dataset private. These ideas are explored in the literature and are primarily referred to as differential privacy. One of the main objectives in the privacy literature is that an adversary is unable to recover an original training sample only from the model’s parameters or outputs (sometimes called a Membership Inference Attack-MIA).
- We want to generate quality data which represents the underlying distribution. We need to strike a balance between releasing individual patient data and creating useful medical applications. In theory, we would want to measure how close our GANs output distribution matches with the true distribution. This is unfeasible not only because we don’t have access to the true distribution, but also because GANs are an implicit density estimator. Practically, we will likely rate how the GANs training data is able to improve the performance on a downstream task utilizing the generated data.

![Image](https://github.com/HalfBloodPrince010/dataset-distillation-for-privacy-GANs/blob/gh-pages/Images/Screen%20Shot%202020-11-30%20at%2012.04.03%20PM.png)

![Image](https://github.com/HalfBloodPrince010/dataset-distillation-for-privacy-GANs/blob/gh-pages/Images/Screen%20Shot%202020-11-30%20at%2012.04.20%20PM.png)

# Input and Output

We explored medical datasets like ImageCLEF Radiology Image dataset (4000 training images, 500 Validation images, 500 test set images), and STARE retinal image dataset. The models were trained on the ImageCLEF Radiology images, the results are near to the original scans, we can’t really comprehend or evaluate the results obtained. So, we went ahead with using celebA dataset to generate a new celebrity dataset and measure privacy.
As the outcome of the project, we will analyze the privacy concerns associated with releasing a dataset by inculcating measures like Maximum mean discrepancy(MMD), Nearest Neighbors etc.

# Models and Experimentation

## Data Generation Phase

Firstly, we trained a Gaussian to understand the divergence between the distributions. We explored different GAN’s like DCGAN( Linear Layers & Conv Layer Architectures), WGAN with Fully Connected Generator and Convoluted WGAN with Gradient Penalty for different Hyperparameter settings.

![Image](https://github.com/HalfBloodPrince010/dataset-distillation-for-privacy-GANs/blob/gh-pages/Images/design.png)

## Models

### Gaussians

We experimented with some much simpler low dimensional variables to get a richer understanding of the trade-off between sample quality, distribution understanding, and privacy protection. In many cases, having generated the underlying training data, we have access to a pure measure of performance in terms of the distance from the true distribution. This helps ground our understanding of performance relative to the amount of privacy gained by a numerical model like differential privacy. We trained both standard GANs and WGANs on these simple datasets to find the model with the best performance on these simpler datasets.
In the following, we can see the performance of a GAN on fitting an extremely simple two-dimensional Gaussian variable with 2000 training samples.

#### Simple Gaussian



#### Multimodal Gaussian



### DCGAN

DCGAN with Convolution Architecture produced better images compared to that of DCGAN with Linear Layers with the CelebA dataset. However, it suffered from the mode collapse. And the overall quality of the images wasn’t that clear for the CelebA dataset. So, we explored other GAN models.

### WGAN - Fully Connected Generator

Loss measure used in DCGAN is susceptible to mode collapse. Taking this into consideration, WGAN uses Wasserstein distance (distance function defined between probability distributions on a given metric space) as a loss measure for smoother gradients and this loss function is immune to mode collapse.
In this model, we use linear or fully-connected layers in the generator and discriminator architecture. We use the Earth Mover distance or Wasserstein distance for the loss function with gradient clipping to achieve 1-lipschitz constraint

### Convoluted WGAN with Gradient Penalty

Similar to the above GAN, this model uses Wasserstein distance as a loss measure, but to enforce Lipschitz constraint, we use gradient norm as a penalty to clip the gradients. Furthermore, in this model we use convoluted layers instead of linear layers in the generator and discriminator model.
The network architecture is as follows :

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/HalfBloodPrince010/dataset-distillation-for-privacy-GANs/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
