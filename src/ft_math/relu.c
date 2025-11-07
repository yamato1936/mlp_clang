#include "ft_math.h"

double	relu(double x)
{
	return (x > 0.0) ? x : 0.0;
}

double	drelu(double x)
{
	return (x > 0.0) ? 1.0 : 0.0;
}
