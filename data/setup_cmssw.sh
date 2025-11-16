#!/bin/bash

CURRENT_DIR=$PWD

source /cvmfs/cms.cern.ch/cmsset_default.sh
CMSSW_VERSION=CMSSW_16_0_X_2025-11-13-1100

cmsrel ${CMSSW_VERSION}
cd ${CMSSW_VERSION}/src/
cmsenv

git cms-init
git cms-addpkg HLTrigger/Configuration Configuration/EventContent DQM/Integration DQMOffline/Trigger RecoTracker/FinalTrackSelectors Validation/RecoTrack Configuration/PyReleaseValidation RecoTracker/MkFit

git apply $CURRENT_DIR/prs/mario.patch
git apply $CURRENT_DIR/prs/manos.patch

cd $CMSSW_BASE/src/RecoTracker/MkFit/
git clone git@github.com:cms-data/RecoTracker-MkFit.git data
curl https://raw.githubusercontent.com/cms-data/RecoTracker-MkFit/e19729f499cc7fd6a780922e86252621e892e929/mkfit-phase2-highPtTripletStep.json -o data/mkfit-phase2-highPtTripletStep.json
curl https://raw.githubusercontent.com/cms-data/RecoTracker-MkFit/e19729f499cc7fd6a780922e86252621e892e929/mkfit-phase2-lstStep.json -o data/mkfit-phase2-lstStep.json

cd $CMSSW_BASE/src/
scram b -j 12

cd $CURRENT_DIR