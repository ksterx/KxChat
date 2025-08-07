import click

from kxchat.scripts.chat import start_chat


@click.group(chain=True)
def main():
    pass


@main.command("with")
@click.argument("repo", type=str)
@click.option("-r", "--revision", type=str, default="main")
@click.option("-t", "--temperature", type=float, default=0.7)
@click.option("-p", "--top-p", type=float, default=0.95)
@click.option("-k", "--top-k", type=int, default=50)
@click.option("-a", "--assistant-name", type=str, default=None)
@click.option("-s", "--system-prompt", type=str, default=None)
@click.option("-v", "--verbose", is_flag=True, default=False)
def chat_with(
    repo: str,
    revision: str,
    temperature: float,
    top_p: float,
    top_k: int,
    assistant_name: str | None,
    system_prompt: str | None,
    verbose: bool,
):
    start_chat(
        repo=repo,
        revision=revision,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        assistant_name=assistant_name,
        system_prompt=system_prompt,
        verbose=verbose,
    )
