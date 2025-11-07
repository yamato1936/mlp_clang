#include "mlp.h"
#include "mnist.h"
#include "ft_math.h"
#include <time.h>

int	main(void)
{
	int		sizes[] = {
		784, 512, 256, 128, 64, 10
	};
	int		num_layers = sizeof(sizes) / sizeof(sizes[0]);
	t_mlp	mlp;

	srand((unsigned)time(NULL));

	/* num_layers should be the number of elements in sizes */
	mlp = init_mlp(sizes, num_layers, 0.01);
	if (mlp.layers == NULL)
	{
		printf("init failed\n");
		return (1);
	}
	printf("MLP initialized: %d layers\n", mlp.num_layers);
	/* train: small number of epochs for quick check */
	train_mnist(&mlp, MNIST_TRAIN_IMAGES, MNIST_TRAIN_LABELS, 200, 1000);
	free_mlp(&mlp);
	return (0);
}
