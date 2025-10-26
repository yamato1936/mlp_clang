#include "mlp.h"
#include <stdlib.h>
#include <string.h>

static double	**alloc_2d(int rows, int cols)
{
	double	**m;
	int		i;

	m = malloc(sizeof(double *) * rows);
	if (!m)
		return (NULL);
	i = 0;
	while (i < rows)
	{
		m[i] = malloc(sizeof(double) * cols);
		if (!m[i])
		{
			/* free previously allocated */
			while (i-- > 0)
				free(m[i]);
			free(m);
			return (NULL);
		}
		i++;
	}
	return (m);
}

t_mlp	init_mlp(int *sizes, int num_layers, double lr)
{
	t_mlp	mlp;
	int		l;
	int		i;
	int		j;

	mlp.num_layers = num_layers - 1;
	mlp.layers = malloc(sizeof(t_layer) * mlp.num_layers);
	if (!mlp.layers)
	{
		mlp.layers = NULL;
		return (mlp);
	}
	mlp.lr = lr;
	l = 0;
	while (l < mlp.num_layers)
	{
		mlp.layers[l].in_dim = sizes[l];
		mlp.layers[l].out_dim = sizes[l + 1];
		mlp.layers[l].w = alloc_2d(mlp.layers[l].out_dim,
					   mlp.layers[l].in_dim);
		mlp.layers[l].b = malloc(sizeof(double) * mlp.layers[l].out_dim);
		mlp.layers[l].a = malloc(sizeof(double) * mlp.layers[l].out_dim);
		mlp.layers[l].z = malloc(sizeof(double) * mlp.layers[l].out_dim);
		mlp.layers[l].delta = malloc(sizeof(double) * mlp.layers[l].out_dim);
		if (!mlp.layers[l].w || !mlp.layers[l].b || !mlp.layers[l].a
		    || !mlp.layers[l].z || !mlp.layers[l].delta)
		{
			/* TODO: free all allocated; simplified here */
			return (mlp);
		}
		/* init weights small random, biases zero */
		i = 0;
		while (i < mlp.layers[l].out_dim)
		{
			mlp.layers[l].b[i] = 0.0;
			j = 0;
			while (j < mlp.layers[l].in_dim)
			{
				mlp.layers[l].w[i][j] =
					(((double)rand() / RAND_MAX) - 0.5) * 0.2;
				j++;
			}
			i++;
		}
		l++;
	}
	return (mlp);
}

void	free_mlp(t_mlp *mlp)
{
	int	l;
	int	i;

	if (!mlp || !mlp->layers)
		return ;
	l = 0;
	while (l < mlp->num_layers)
	{
		if (mlp->layers[l].w)
		{
			i = 0;
			while (i < mlp->layers[l].out_dim)
			{
				free(mlp->layers[l].w[i]);
				i++;
			}
			free(mlp->layers[l].w);
		}
		free(mlp->layers[l].b);
		free(mlp->layers[l].a);
		free(mlp->layers[l].z);
		free(mlp->layers[l].delta);
		l++;
	}
	free(mlp->layers);
	mlp->layers = NULL;
}
