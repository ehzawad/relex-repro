#!/bin/bash
# Publish the relex/ subtree of the monorepo, with its history, to the private repo ehzawad/relex-repro.
# git subtree is not installed on cluster, so: fresh clone of the monorepo -> filter-branch --subdirectory-filter relex
# (deterministic, so re-publishing is a fast-forward) -> push to relex-repro main.
set -e
W=$HOME/ehz_agent/store/relex/publish_clone
rm -rf $W; git clone -q --no-local --branch main $HOME/ehz_agent $W
cd $W; FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --subdirectory-filter relex -- main >/dev/null 2>&1
git push -q https://github.com/ehzawad/relex-repro.git main:main
git log -1 --format="published %h %s ($(git rev-list --count main) commits)"
