#!/bin/bash
# Publish the relex/ subtree of the monorepo (with its history) to the private repo ehzawad/relex-repro.
#   relex/publish.sh            (run from anywhere; commits must already be made in the monorepo)
set -e
cd $HOME/ehz_agent
git subtree split --prefix=relex -b relex-publish >/dev/null
git push -q https://github.com/ehzawad/relex-repro.git relex-publish:main
git log -1 --format="published %h %s" relex-publish
