# ecg-gan

Check out this [wandb run](https://wandb.ai/sjanas/ECG%20GAN/runs/fz2ptaxy) to get a feel for the baseline model or run `python train.py` to reproduce the results

### Suggested reading

- [DCGAN TUTORIAL](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) on PyTorch website.
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) by Alec Radford, Luke Metz, Soumith Chintala.
- [Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)
  - Focused on points: 6, 7. Applied most of them.

### Code Style

This repository uses pre-commit hooks with forced python formatting ([black](https://github.com/psf/black),
[flake8](https://flake8.pycqa.org/en/latest/), and [isort](https://pycqa.github.io/isort/)):

```sh
pip install pre-commit
pre-commit install
```

Whenever you execute `git commit` the files altered / added within the commit will be checked and corrected.
`black` and `isort` can modify files locally - if that happens you have to `git add` them again.
You might also be prompted to introduce some fixes manually.

To run the hooks against all files without running `git commit`:

```sh
pre-commit run --all-files
```
