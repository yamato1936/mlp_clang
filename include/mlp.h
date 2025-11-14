#ifndef MLP_H
# define MLP_H

# include <stdlib.h>
# include <stdio.h>

typedef struct s_layer
{
	int		in_dim;
	int		out_dim;
	double	**w;    /* [out_dim][in_dim] */
	double	*b;     /* [out_dim] */
	double	*a;     /* activation: hidden -> sigmoid, output -> logits/probs */
	double	*z;     /* pre-activation (linear) */
	double	*delta; /* delta for backprop */
	double	*output; /* for momentum (not used currently) */
}	t_layer;

typedef struct s_mlp
{
	int		num_layers; /* number of layers (actual, e.g., 2 layers => hidden+output) */
	t_layer	*layers;
	double	lr;
}	t_mlp;

typedef struct s_grad_accum
{
	double	***dw;	/* [num_layers][out_dim][in_dim] */
	double	**db;	/* [num_layers][out_dim] */
	int		*out_dims;	/* [num_layers] */
	int		*in_dims;	/* [num_layers] */
	int		num_layers;
}	t_grad_accum;

/* core */
t_mlp	init_mlp(int *sizes, int num_layers, double lr);
void	free_mlp(t_mlp *mlp);
double	*forward(t_mlp *mlp, double *input);
void	backward(t_mlp *mlp, double *input, double *target);
void	update_weights(t_mlp *mlp, double *input);

/* loss/util */
double	compute_loss(t_mlp *mlp, double *target);

/* training */
void	train_mnist(t_mlp *mlp, const char *train_img,
		   const char *train_lbl, int epochs, int limit);

#endif
