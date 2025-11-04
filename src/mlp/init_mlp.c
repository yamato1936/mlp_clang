#include "mlp.h"
#include <stdlib.h>
#include <stdio.h>

/* 2次元配列（weights）の確保関数 */
static double **alloc_2d(int rows, int cols)
{
	double **m = malloc(sizeof(double *) * rows);
	if (!m)
		return (NULL);
	for (int i = 0; i < rows; i++)
	{
		m[i] = malloc(sizeof(double) * cols);
		if (!m[i])
		{
			while (i-- > 0)
				free(m[i]);
			free(m);
			return (NULL);
		}
	}
	return (m);
}

/* 多層パーセプトロン初期化関数 */
t_mlp init_mlp(int *sizes, int num_layers, double lr)
{
	t_mlp mlp;
	mlp.layers = NULL;
	mlp.num_layers = num_layers - 1;
	mlp.lr = lr;

	/* num_layersが不正な場合はNULLを返す */
	if (mlp.num_layers <= 0)
	{
		fprintf(stderr, "init_mlp: invalid num_layers (%d)\n", num_layers);
		return (mlp);
	}

	mlp.layers = malloc(sizeof(t_layer) * mlp.num_layers);
	if (!mlp.layers)
	{
		fprintf(stderr, "init_mlp: malloc failed for layers\n");
		exit(EXIT_FAILURE);
	}

	for (int l = 0; l < mlp.num_layers; l++)
	{
		int in_dim = sizes[l];
		int out_dim = sizes[l + 1];

		mlp.layers[l].in_dim = in_dim;
		mlp.layers[l].out_dim = out_dim;
		mlp.layers[l].w = alloc_2d(out_dim, in_dim);
		mlp.layers[l].b = malloc(sizeof(double) * out_dim);
		mlp.layers[l].a = malloc(sizeof(double) * out_dim);
		mlp.layers[l].z = malloc(sizeof(double) * out_dim);
		mlp.layers[l].delta = malloc(sizeof(double) * out_dim);

		if (!mlp.layers[l].w || !mlp.layers[l].b || !mlp.layers[l].a ||
		    !mlp.layers[l].z || !mlp.layers[l].delta)
		{
			fprintf(stderr, "init_mlp: allocation failed at layer %d\n", l);
			/* ここまでの層をすべて解放 */
			for (int ll = 0; ll <= l; ll++)
			{
				if (mlp.layers[ll].w)
				{
					for (int i = 0; i < mlp.layers[ll].out_dim; i++)
						free(mlp.layers[ll].w[i]);
					free(mlp.layers[ll].w);
				}
				free(mlp.layers[ll].b);
				free(mlp.layers[ll].a);
				free(mlp.layers[ll].z);
				free(mlp.layers[ll].delta);
			}
			free(mlp.layers);
			exit(EXIT_FAILURE);
		}

		/* 重みとバイアスの初期化 */
		for (int i = 0; i < out_dim; i++)
		{
			mlp.layers[l].b[i] = 0.0;
			for (int j = 0; j < in_dim; j++)
				mlp.layers[l].w[i][j] = (((double)rand() / RAND_MAX) - 0.5) * 0.2;
		}
	}

	/* デバッグ表示 */
	fprintf(stderr, "init_mlp: successfully allocated %d layers\n", mlp.num_layers);
	for (int l = 0; l < mlp.num_layers; l++)
		fprintf(stderr, "  layer %d: in=%d out=%d w=%p\n",
		        l, mlp.layers[l].in_dim, mlp.layers[l].out_dim, (void*)mlp.layers[l].w);

	return (mlp);
}

/* MLPの全解放 */
void free_mlp(t_mlp *mlp)
{
	if (!mlp || !mlp->layers)
		return;
	for (int l = 0; l < mlp->num_layers; l++)
	{
		if (mlp->layers[l].w)
		{
			for (int i = 0; i < mlp->layers[l].out_dim; i++)
				free(mlp->layers[l].w[i]);
			free(mlp->layers[l].w);
		}
		free(mlp->layers[l].b);
		free(mlp->layers[l].a);
		free(mlp->layers[l].z);
		free(mlp->layers[l].delta);
	}
	free(mlp->layers);
	mlp->layers = NULL;
}
