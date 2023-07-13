import click

from modules.visualize import plot_ocr

@click.command()
@click.argument("img_dir", type=click.Path(exists=True))
def plot(img_dir):
    plot_ocr(img_dir)

if __name__ == "__main__":
    plot()