#include "mlp.h"
#include "mnist.h"
#include "ft_math.h"
#include <time.h>

int	main(void)
{
	int		sizes[3];
	t_mlp	mlp;

	sizes[0] = 784;
	sizes[1] = 128;
	sizes[2] = 10;
	srand((unsigned)time(NULL));

	/* num_layers should be the number of elements in sizes (3) */
	mlp = init_mlp(sizes, 3, 0.01);
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
