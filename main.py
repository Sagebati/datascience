import click
from auto_encoder import generate_trained_models, generate_graphs
from clusters import kmeans_fit, tsne


@click.command()
@click.option("--learn", is_flag=True, default=False)
@click.option("--encoded-images", is_flag=True, default=False)
@click.option("--kmeans", is_flag=True, default=False)
@click.option("--sne", is_flag=True, default=False)
def a(learn, encoded_images, kmeans, sne):
    if learn:
        generate_trained_models()
    if encoded_images:
        generate_graphs()
    if kmeans:
        kmeans_fit()
    if sne:
        tsne()


if __name__ == '__main__':
    a()
