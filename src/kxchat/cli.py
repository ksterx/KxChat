import click

from kxchat.scripts.chat import start_chat_room


@click.group(chain=True)
def main():
    pass


@click.command("room")
@click.option("--repo", type=str, required=True, help="The repository to use.")
@click.option("--revision", type=str, default="main", help="The revision to use.")
@click.option("--subset", type=str, default="default", help="The subset to use.")
def room(repo: str, revision: str, subset: str):
    start_chat_room(repo, revision, subset)
