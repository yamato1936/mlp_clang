#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* read 32bit big-endian */
static int	read_be_int(FILE *f)
{
	unsigned char buf[4];
	if (fread(buf, 1, 4, f) != 4)
		return (-1);
	return ((buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3]);
}

t_mnist	load_mnist_images(const char *path, int limit)
{
	FILE	*f;
	int		magic;
	int		num_images;
	int		rows;
	int		cols;
	int		i;
	int		j;
	t_mnist	ret;
	unsigned char *buf;
	int		read_count;
	int		use;

	ret.images = NULL;
	ret.labels = NULL;
	ret.count = 0;
	f = fopen(path, "rb");
	if (!f)
		return (ret);
	magic = read_be_int(f);
	if (magic != 2051)
	{
		fclose(f);
		return (ret);
	}
	num_images = read_be_int(f);
	rows = read_be_int(f);
	cols = read_be_int(f);
	if (num_images <= 0 || rows <= 0 || cols <= 0)
	{
		fclose(f);
		return (ret);
	}
	use = num_images;
	if (limit > 0 && limit < use)
		use = limit;
	ret.images = malloc(sizeof(double *) * use);
	if (!ret.images)
	{
		fclose(f);
		return (ret);
	}
	buf = malloc(rows * cols);
	if (!buf)
	{
		free(ret.images);
		fclose(f);
		return (ret);
	}
	i = 0;
	while (i < use)
	{
		read_count = fread(buf, 1, rows * cols, f);
		if (read_count != rows * cols)
			break;
		ret.images[i] = malloc(sizeof(double) * (rows * cols));
		if (!ret.images[i])
			break;
		j = 0;
		while (j < rows * cols)
		{
			ret.images[i][j] = ((double)buf[j]) / 255.0;
			j++;
		}
		i++;
	}
	ret.count = i;
	free(buf);
	fclose(f);
	return (ret);
}

t_mnist	load_mnist_labels(const char *path, int limit)
{
	FILE	*f;
	int		magic;
	int		num_labels;
	int		i;
	int		use;
	t_mnist	ret;
	unsigned char val;

	ret.images = NULL;
	ret.labels = NULL;
	ret.count = 0;
	f = fopen(path, "rb");
	if (!f)
		return (ret);
	magic = read_be_int(f);
	if (magic != 2049)
	{
		fclose(f);
		return (ret);
	}
	num_labels = read_be_int(f);
	use = num_labels;
	if (limit > 0 && limit < use)
		use = limit;
	ret.labels = malloc(sizeof(uint8_t) * use);
	if (!ret.labels)
	{
		fclose(f);
		return (ret);
	}
	i = 0;
	while (i < use)
	{
		if (fread(&val, 1, 1, f) != 1)
			break;
		ret.labels[i] = val;
		i++;
	}
	ret.count = i;
	fclose(f);
	return (ret);
}

void	free_mnist(t_mnist *m)
{
	int i;
	if (!m)
		return ;
	if (m->images)
	{
		i = 0;
		while (i < m->count)
		{
			free(m->images[i]);
			i++;
		}
		free(m->images);
	}
	if (m->labels)
		free(m->labels);
	m->count = 0;
}
