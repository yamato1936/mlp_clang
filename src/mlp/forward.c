#include "mlp.h"
#include "ft_math.h"
#include <stdlib.h>

double *forward(t_mlp *mlp, double *input)
{
	int l, i, k;
	double sum;
	double *prev;

	if (!mlp || !mlp->layers || !input)
		return NULL;

	prev = input;
	for (l = 0; l < mlp->num_layers; l++)
	{
		t_layer *layer = &mlp->layers[l];

		for (i = 0; i < layer->out_dim; i++)
		{
			sum = (layer->b) ? layer->b[i] : 0.0;
			for (k = 0; k < layer->in_dim; k++)
				sum += layer->w[i][k] * prev[k];
			layer->z[i] = sum;

			// 活性化関数の適用
			if (l < mlp->num_layers - 1)
				layer->a[i] = relu(sum);
			else
				layer->a[i] = sum; // 出力層は softmax 前のlogits
		}
		prev = layer->a;
	}
	return mlp->layers[mlp->num_layers - 1].a;
}
