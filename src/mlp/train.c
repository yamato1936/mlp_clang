#include "mlp.h"
#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double	*make_one_hot(int label, int n)
{
	double	*vec;
	int		i;

	vec = malloc(sizeof(double) * n);
	if (!vec)
		return (NULL);
	i = 0;
	while (i < n)
	{
		vec[i] = 0.0;
		i++;
	}
	if (label >= 0 && label < n)
		vec[label] = 1.0;
	return (vec);
}

void	train_mnist(t_mlp *mlp, const char *train_img,
		    const char *train_lbl, int epochs, int limit)
{
	t_mnist	images;
	t_mnist	labels; /* reuse struct for labels loader */
	int		i;
	int		e;
	int		samples;
	double	loss;
	double	*target;

	(void)labels;
	images = load_mnist_images(train_img, limit);
	labels = load_mnist_labels(train_lbl, limit);
	if (images.count == 0 || labels.count == 0)
	{
		printf("MNIST load failed\n");
		return ;
	}
	samples = images.count;
	e = 0;
	while (e < epochs)
	{
		loss = 0.0;
		i = 0;
		while (i < samples)
		{
			/* forward */
			forward(mlp, images.images[i]);
			/* prepare target */
			target = make_one_hot(labels.labels[i], mlp->layers[mlp->num_layers - 1].out_dim);
			/* compute loss (softmax + cross entropy) */
			loss += compute_loss(mlp, target);
			/* backward */
			backward(mlp, images.images[i], target);
			/* update */
			update_weights(mlp, images.images[i]);
			free(target);
			i++;
		}
		printf("Epoch %d, Loss %.6f\n", e, loss / samples);
		e++;
	}
	free_mnist(&images);
	free_mnist(&labels);
}
