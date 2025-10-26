#include "ft_math.h"
#include <math.h>

double	sigmoid(double x)
{
	/* numerically stable sigmoid */
	if (x >= 0)
		return (1.0 / (1.0 + exp(-x)));
	else
	{
		double ex = exp(x);
		return (ex / (1.0 + ex));
	}
}

double	dsigmoid(double y)
{
	return (y * (1.0 - y));
}
