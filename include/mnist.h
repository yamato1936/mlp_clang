#ifndef MNIST_H
# define MNIST_H

# include <stdint.h>

# define MNIST_TRAIN_IMAGES "data/train-images-idx3-ubyte"
# define MNIST_TRAIN_LABELS "data/train-labels-idx1-ubyte"
# define MNIST_TEST_IMAGES "data/t10k-images-idx3-ubyte"
# define MNIST_TEST_LABELS "data/t10k-labels-idx1-ubyte"

typedef struct s_mnist
{
	double	**images; /* images[sample][784] normalized 0..1 */
	uint8_t	*labels;  /* labels[sample] */
	int		count;
}	t_mnist;

/* loaders */
t_mnist	load_mnist_images(const char *path, int limit);
t_mnist	load_mnist_labels(const char *path, int limit);
void	free_mnist(t_mnist *m);

#endif
