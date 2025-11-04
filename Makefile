NAME = re_mlp

CC = cc
CFLAGS = -Wall -Wextra -Werror -Iinclude

SRC = \
	src/main.c \
	src/mlp/init_mlp.c \
	src/mlp/forward.c \
	src/mlp/backward.c \
	src/mlp/update.c \
	src/mlp/loss.c \
	src/mlp/train.c \
	src/ft_math/sigmoid.c \
	src/ft_math/softmax.c \
	src/ft_math/cross_entropy.c \
	src/mnist/mnist_loader.c \
	src/mnist/mnist_preprocess.c

OBJ = $(SRC:.c=.o)

all: $(NAME)

$(NAME): $(OBJ)
	$(CC) $(CFLAGS) -o $(NAME) $(OBJ) -lm

clean:
	rm -f $(OBJ)

fclean: clean
	rm -f $(NAME)

re: fclean all

.PHONY: all clean fclean re
