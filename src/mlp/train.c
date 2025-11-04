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
	t_mnist	images;
	t_mnist	labels; /* reuse struct for labels loader */
	int		i;
	int		e;
	int		samples;
	double	loss;
	double	*target;

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
		int correct = 0;

		i = 0;
		while (i < samples)
		{
			/* forward */
			double *output = forward(mlp, images.images[i]);   // forward が戻り値を返す形にしておく
			softmax(output, mlp->layers[mlp->num_layers - 1].out_dim); // ← 追加！
			/* prepare target */
			target = make_one_hot(labels.labels[i], mlp->layers[mlp->num_layers - 1].out_dim);
			/* compute loss (softmax + cross entropy) */
			loss += compute_loss(mlp, target);
			/* backward */
			backward(mlp, images.images[i], target);
			/* update */
			update_weights(mlp, images.images[i]);

			/* accuracy */
			int pred = argmax(output, mlp->layers[mlp->num_layers - 1].out_dim);
			if (pred == labels.labels[i])
				correct++;

			free(target);
			i++;
		}

		double accuracy = (double)correct / samples * 100.0;
		printf("Epoch %d, Loss %.6f, Accuracy %.2f%%\n", e, loss / samples, accuracy);
		e++;
	}

	free_mnist(&images);
	free_mnist(&labels);
}
