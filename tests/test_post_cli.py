from vicmf6.cli import _parser


def test_post_all_cli_is_registered():
    args = _parser().parse_args(["post", "all", "-c", "config.yml"])
    assert args.command == "post"
    assert args.post_command == "all"
    assert args.formats == "png,pdf"
