# Contributing to sklearn-morpho

Thanks for choosing sklearn-morpho.

Any help is greatly appreciated, whether that is through proposed code changes,
issues and bug reports, or other means like general feedback.

However, we believe it is important to set some guidelines to contributing to
this project, to ensure the codebase is kept as clean and maintainable as
possible.

Please adhere to these rules when contributing.

## Commits

More precision will now be given for proposed code changes: you should add them
via GitHub Pull requests, keep them unitary and scope their changes cleanly.

As this repo is currently set up to rebase feature branches, be reminded that
all your commits will be present in the final tree.
Please keep it clean by adhering to
[commit conventions](https://www.conventionalcommits.org/en/v1.0.0/) and by
keeping your commits unitary, and by not adding merge commits (use `git rebase`
instead of `git merge`).

Commit early, push often.
But as keeping the final commit history clean is a priority, do not hesitate to
rebase your branch, squash and cherry-pick commits to split complicated features
when needed.

For example, when updating the CI, it might take a few commits to get it right.
If that happens, squash the small fix commits you made into a single big one
along with the original CI change.

## CI/CD

This repository uses a CI/CD pipeline with tests and an enforced coding style.
You are advised to using a pre-commit hook to avoid unclean commits that only
fix the coding style for a feature.

You might notice that a CI job penalizes TODOs in the code.
This does not mean you should not use them, but rather that an unfinished piece
of work should either be completed within the pull request (if it belongs there
scope-wise) or become a repo issue, to avoid partially working code.

## Tests

We believe in test-driven development, and each feature and code path must be
tested to the extent of what is reasonable.

This means that if you implement a feature, it must be tested to prevent
regression errors, ensure the code actually works, and ease bug fixing later on.

## LLMs

Regarding LLM-driven pull requests, they are discouraged, as you should be able
to understand and help maintain the changes you add to this repo.

Be reminded that understanding and belonging is what keeps open source so
productive and respectful to maintainers.

Happy coding!
