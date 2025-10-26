#ifndef FT_MATH_H
# define FT_MATH_H

double	sigmoid(double x);
double	dsigmoid(double y);

void	softmax(double *x, int n);
double	cross_entropy(double *pred, double *target, int n);

#endif
