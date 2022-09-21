# Contributing

Thank you for considering contributing to `EnsembleKalmanProcesses`! We encourage opening issues and pull requests (PRs).

## What to contribute?

- The easiest way to contribute is by using `EnsembleKalmanProcesses`, identifying
  problems and opening issues;

- You can try to tackle an existing [issue](https://github.com/CliMA/EnsembleKalmanProcesses.jl/issues). It is best to outline your proposed solution in the issue thread before implementing it in a PR;

- Write an example or tutorial. It is likely that other users may find your use of `EnsembleKalmanProcesses` insightful;

- Improve documentation or comments if you found something hard to use;

- Implement a new feature if you need it. We strongly encourage opening an issue to make sure the administrators are on board before opening a PR with an unsolicited feature addition.

## Using `git`

If you are unfamiliar with `git` and version control, the following guides
will be helpful:

- [Atlassian (bitbucket) `git`
  tutorials](https://www.atlassian.com/git/tutorials). A set of tips and tricks
  for getting started with `git`.
- [GitHub's `git` tutorials](https://try.github.io/). A set of resources from
  GitHub to learn `git`.

### Forks and branches

Create your own fork of `EnsembleKalmanProcesses` [on
GitHub](https://github.com/CliMA/EnsembleKalmanProcesses.jl) and check out your copy:

```
$ git clone https://github.com/<your-username>/EnsembleKalmanProcesses.jl.git
$ cd EnsembleKalmanProcesses.jl
```

Now you have access to your fork of `EnsembleKalmanProcesses` through `origin`. Create a branch for your feature; this will hold your contribution:

```
$ git checkout -b <branchname>
```

#### Some useful tips

- When you start working on a new feature branch, make sure you start from
  main by running: `git checkout main` and `git pull`.

- Create a new branch from main by using `git checkout -b <branchname>`.

### Develop your feature

Make sure you add tests for your code in `test/` and appropriate documentation in the code and/or
in `docs/`. Before committing your changes, you can verify their behavior by running the tests, the examples, and building the documentation [locally](https://clima.github.io/EnsembleKalmanProcesses.jl/previews/PR157/installation_instructions/). In addition, make sure your feature follows the formatting guidelines by running
```
julia --project=.dev .dev/climaformat.jl .
```
from the `EnsembleKalmanProcesses.jl` directory.

### Squash and rebase

When your PR is ready for review, clean up your commit history by squashing
and make sure your code is current with `EnsembleKalmanProcesses.jl` main by rebasing. The general rule is that a PR should contain a single commit with a descriptive message.

To make sure you are up to date with main, you can use the following workflow:

```
$ git checkout main
$ git pull
$ git checkout <name_of_local_branch>
$ git rebase main
```
This may create conflicts with the local branch. The conflicted files will be outlined by git. To resolve conflicts,
we have to manually edit the files (e.g. with vim). The conflicts will appear between >>>>, ===== and <<<<<.
We need to delete these lines and pick what version we want to keep.

To squash your commits, you can use the following command:

```
$ git rebase -i HEAD~n
```

where `n` is the number of commits you need to squash into one. Then, follow the instructions in the terminal. For example, to squash 4 commits:
```
$ git rebase -i HEAD~4
```
will open the following file in (typically) vim:

```
   pick 01d1124 <commit message 1>
   pick 6340aaa <commit message 2>
   pick ebfd367 <commit message 3>
   pick 30e0ccb <commit message 4>

   # Rebase 60709da..30e0ccb onto 60709da
   #
   # Commands:
   #  p, pick = use commit
   #  e, edit = use commit, but stop for amending
   #  s, squash = use commit, but meld into previous commit
   #
   # If you remove a line here THAT COMMIT WILL BE LOST.
   # However, if you remove everything, the rebase will be aborted.
##
```

We want to keep the first commit and squash the last 3. We do so by changing the last three commits to `squash` and then do `:wq` on vim.

```
   pick 01d1124 <commit message 1>
   squash 6340aaa <commit message 2>
   squash ebfd367 <commit message 3>
   squash 30e0ccb <commit message 4>

   # Rebase 60709da..30e0ccb onto 60709da
   #
   # Commands:
   #  p, pick = use commit
   #  e, edit = use commit, but stop for amending
   #  s, squash = use commit, but meld into previous commit
   #
   # If you remove a line here THAT COMMIT WILL BE LOST.
   # However, if you remove everything, the rebase will be aborted.
```

Then in the next screen that appears, we can just delete all messages that
we do not want to show in the commit. After this is done and we are back to 
the console, we have to force push. We need to force push because we rewrote
the local commit history.

```
$ git push -uf origin <name_of_local_branch>
```

You can find more information about squashing [here](https://github.com/edx/edx-platform/wiki/How-to-Rebase-a-Pull-Request#squash-your-changes).

### Unit testing

Currently a number of checks are run per commit for a given PR.

- `JuliaFormatter` checks if the PR is formatted with `.dev/climaformat.jl`.
- `Documentation` rebuilds the documentation for the PR and checks if the docs
  are consistent and generate valid output.
- `Unit Tests` run subsets of the unit tests defined in `tests/`, using `Pkg.test()`.
  The tests are run in parallel to ensure that they finish in a reasonable time.
  The tests only run the latest commit for a PR, branch and will kill any stale jobs on push.
  These tests are only run on linux (Ubuntu LTS).

Unit tests are run against every new commit for a given PR,
the status of the unit-tests are not checked during the merge
process but act as a sanity check for developers and reviewers.
Depending on the content changed in the PR, some CI checks that
are not necessary will be skipped.  For example doc only changes
do not require the unit tests to be run.

### The merge process

We use [`bors`](https://bors.tech/) to manage merging PR's in the the `EnsembleKalmanProcesses` repo.
If you're a collaborator and have the necessary permissions, you can type
`bors try` in a comment on a PR to have integration test suite run on that
PR, or `bors r+` to try and merge the code.  Bors ensures that all integration tests
for a given PR always pass before merging into `main`. The integration tests currently run example cases in `examples/`. Any breaking changes will need to also update the `examples/`, else bors will fail.
