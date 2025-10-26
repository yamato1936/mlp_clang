#include "ft_math.h"
#include <math.h>

/* in-place softmax with max-subtraction for stability */
void	softmax(double *x, int n)
{
	int		i;
	int		j;
	double	max;
	double	sum;

	if (n <= 0)
		return ;
	max = x[0];
	i = 1;
	while (i < n)
	{
		if (x[i] > max)
			max = x[i];
		i++;
	}
	sum = 0.0;
	i = 0;
	while (i < n)
	{
		x[i] = exp(x[i] - max);
		sum += x[i];
		i++;
	}
	if (sum == 0.0)
		return ;
	j = 0;
	while (j < n)
	{
		x[j] = x[j] / sum;
		j++;
	}
}
