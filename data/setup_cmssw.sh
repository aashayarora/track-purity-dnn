#!/bin/bash

CURRENT_DIR=$PWD

source /cvmfs/cms.cern.ch/cmsset_default.sh
CMSSW_VERSION=CMSSW_16_0_X_2025-11-13-1100

cmsrel ${CMSSW_VERSION}
cd ${CMSSW_VERSION}/src/
cmsenv

git cms-init
git cms-addpkg HLTrigger/Configuration Configuration/EventContent DQM/Integration DQMOffline/Trigger RecoTracker/FinalTrackSelectors Validation/RecoTrack Configuration/PyReleaseValidation RecoTracker/MkFit

# manos LST PR
git cherry-pick 4f2c71ec6483e154826d8ac181b62b6e93ad20f3

# Mario mkfit PR
git cherry-pick c90e64223ae151abca0759388ca305fc129d93cd
git cherry-pick 332b8f07ed0f4d3fbbbc28b77f08b51dc8e7eec2

cd $CMSSW_BASE/src/RecoTracker/MkFit/
git clone git@github.com:cms-data/RecoTracker-MkFit.git data
cd data/

# mario mkfit json PR
git cherry-pick e19729f499cc7fd6a780922e86252621e892e929
git cherry-pick c78834a712bac0479fa7739836ed1b50df27f98a

scram b -j 12

cd $CURRENT_DIR