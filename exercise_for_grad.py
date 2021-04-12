import torch

latent_size = 2
image_size = 3

generator = torch.ones((image_size, latent_size), requires_grad=True, dtype=torch.float64)
classifier = torch.ones((1, image_size), requires_grad=True, dtype=torch.float64)

image = torch.tensor([1, 2, 3], dtype=torch.float64)
noise = torch.tensor([1, 0], dtype=torch.float64)

out_im = torch.matmul(classifier, image)
vec1 = torch.autograd.grad(out_im, classifier)[0]

generated_image = torch.matmul(generator, noise)
out_gen_im = torch.matmul(classifier, generated_image)
vec2 = torch.autograd.grad(out_gen_im, classifier, create_graph=True)[0]

loss = ((vec1 - vec2) ** 2).sum()

loss.backward()

print(generator.grad)
print(classifier.grad)
