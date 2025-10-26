#include "mlp.h"
#include "ft_math.h"

double	compute_loss(t_mlp *mlp, double *target)
{
	t_layer	*out;
	double	loss;

	out = &mlp->layers[mlp->num_layers - 1];
	/* softmax in place: convert logits in out->a to probabilities */
	softmax(out->a, out->out_dim);
	loss = cross_entropy(out->a, target, out->out_dim);
	return (loss);
}
