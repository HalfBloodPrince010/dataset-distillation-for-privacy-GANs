## Problem Formulation

Our objective is to use a GAN to both provide high quality samples for training downstream models while simultaneously abstracting away from the original patients used to train the model. Overall, this objective consists of two competing tasks which we will attempt to effectively measure in our experiments:
- We want to make the original dataset private. These ideas are explored in the literature and are primarily referred to as differential privacy. One of the main objectives in the privacy literature is that an adversary is unable to recover an original training sample only from the model’s parameters or outputs (sometimes called a Membership Inference Attack-MIA).
- We want to generate quality data which represents the underlying distribution. We need to strike a balance between releasing individual patient data and creating useful medical applications. In theory, we would want to measure how close our GANs output distribution matches with the true distribution. This is unfeasible not only because we don’t have access to the true distribution, but also because GANs are an implicit density estimator. Practically, we will likely rate how the GANs training data is able to improve the performance on a downstream task utilizing the generated data.

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
