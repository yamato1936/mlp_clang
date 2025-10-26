#include "mlp.h"
#include "ft_math.h"

void	forward(t_mlp *mlp, double *input)
{
	int		l;
	int		i;
	int		j;
	int		k;
	double	sum;
	double	*prev;

	prev = input;
	l = 0;
	while (l < mlp->num_layers)
	{
		i = 0;
		while (i < mlp->layers[l].out_dim)
		{
			sum = mlp->layers[l].b[i];
			k = 0;
			while (k < mlp->layers[l].in_dim)
			{
				sum += mlp->layers[l].w[i][k] * prev[k];
				k++;
			}
			mlp->layers[l].z[i] = sum;
			/* hidden layers use sigmoid, output layer keep logits (no activation) */
			if (l < mlp->num_layers - 1)
				mlp->layers[l].a[i] = sigmoid(sum);
			else
				mlp->layers[l].a[i] = sum;
			i++;
		}
		prev = mlp->layers[l].a;
		l++;
	}
}
