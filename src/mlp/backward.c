#include "mlp.h"
#include "ft_math.h"

/*
 * Backprop for network where:
 * - hidden activations = sigmoid
 * - output layer: softmax + cross-entropy
 * After forward, compute_loss() has called softmax on out->a (so out->a are probs).
 * delta_output = pred - target
 */

void	backward(t_mlp *mlp, double *input, double *target)
{
	int		l;
	int		i;
	int		j;
	// int		k;

	/* output layer delta = pred - target */
	t_layer *out = &mlp->layers[mlp->num_layers - 1];
	i = 0;
	while (i < out->out_dim)
	{
		out->delta[i] = out->a[i] - target[i];
		i++;
	}
	/* propagate backwards for hidden layers */
	l = mlp->num_layers - 2;
	while (l >= 0)
	{
		t_layer *layer = &mlp->layers[l];
		t_layer *next = &mlp->layers[l + 1];
		i = 0;
		while (i < layer->out_dim)
		{
			double sum = 0.0;
			j = 0;
			while (j < next->out_dim)
			{
				sum += next->delta[j] * next->w[j][i];
				j++;
			}
			/* derivative of ReLU using pre-activation */
			layer->delta[i] = sum * drelu(layer->z[i]);
			i++;
		}
		l--;
	}
	(void)input;
}
