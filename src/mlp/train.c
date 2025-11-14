#include "mlp.h"
#include "mnist.h"
#include "ft_math.h"
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

static int	argmax(double *arr, int n)
{
	int max_i = 0;
	for (int i = 1; i < n; i++)
		if (arr[i] > arr[max_i])
			max_i = i;
	return max_i;
}

void	train_mnist(t_mlp *mlp, const char *train_img,
		    const char *train_lbl, int epochs, int limit)
{
	t_mnist			images;
	t_mnist			labels;
	int				e;
	int				samples;
	int				i;
	int				out_dim;
	double			*target;
	double			loss;
	int				correct;

	images = load_mnist_images(train_img, limit);
	labels = load_mnist_labels(train_lbl, limit);
	if (images.count == 0 || labels.count == 0)
	{
		printf("MNIST load failed\n");
		return ;
	}
	samples = images.count;
	out_dim = mlp->layers[mlp->num_layers - 1].out_dim;
	e = 0;
	while (e < epochs)
	{
		loss = 0.0;
		correct = 0;

		i = 0;
		while (i < samples)
		{
			/* forward */
			forward(mlp, images.images[i]);
			/* prepare target */
			target = make_one_hot(labels.labels[i], out_dim);
			if (!target)
				break ;
			/* compute loss (softmax + cross entropy) */
			loss += compute_loss(mlp, target);
			/* backward */
			backward(mlp, images.images[i], target);
			/* update */
			update_weights(mlp, images.images[i]);

			/* accuracy */
			if (argmax(mlp->layers[mlp->num_layers - 1].a, out_dim) == labels.labels[i])
				correct++;

			free(target);
			i++;
		}
		printf("Epoch %d, Loss %.6f, Accuracy %.2f%%\n",
			e, loss / (double)samples, (double)correct / (double)samples * 100.0);
		e++;
	}

	free_mnist(&images);
	free_mnist(&labels);
}
