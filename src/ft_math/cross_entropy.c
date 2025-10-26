#include "ft_math.h"
#include <math.h>

double	cross_entropy(double *pred, double *target, int n)
{
	int		i;
	double	loss;
	double	eps;

	eps = 1e-12;
	loss = 0.0;
	i = 0;
	while (i < n)
	{
		if (target[i] > 0.0)
			loss -= target[i] * log(pred[i] + eps);
		i++;
	}
	return (loss);
}
