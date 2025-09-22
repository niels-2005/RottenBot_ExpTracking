from rotten_bot.utils.logging import get_configured_logger
import git

_logger = get_configured_logger()


def get_current_git_sha() -> str:
    """Get the current git commit SHA.

    Raises:
        e: If there is an error accessing the git repository.

    Returns:
        str: The current git commit SHA.
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception as e:
        _logger.error(f"Error getting git SHA: {e}")
        raise e
