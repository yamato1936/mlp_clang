#include "mlp.h"

/*
 * SGD update with learning rate mlp->lr
 * For each layer l:
 *   dW_ij = delta_j * a_prev_i
 *   W_j,i -= lr * dW_ij
 * biases b_j -= lr * delta_j
 * a_prev for first layer is input
 */

void	update_weights(t_mlp *mlp, double *input)
{
	int	l;
	int	i;
	int	j;
	// int	k;
	double	*prev;

	l = 0;
	prev = input;
	while (l < mlp->num_layers)
	{
		/* determine a_prev */
		if (l > 0)
			prev = mlp->layers[l - 1].a;
		i = 0;
		while (i < mlp->layers[l].out_dim)
		{
			j = 0;
			while (j < mlp->layers[l].in_dim)
			{
				mlp->layers[l].w[i][j] -= mlp->lr * mlp->layers[l].delta[i] * prev[j];
				j++;
			}
			mlp->layers[l].b[i] -= mlp->lr * mlp->layers[l].delta[i];
			i++;
		}
		l++;
	}
}
