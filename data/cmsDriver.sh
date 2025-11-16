#!/bin/bash

INPUT_DATASET=$1
JOB_ID=$2

export X509_USER_PROXY=/afs/cern.ch/user/a/aaarora/private/x509up_u141045

source /cvmfs/cms.cern.ch/cmsset_default.sh

cd /eos/user/a/aaarora/tracking/track-purity-dnn/data/CMSSW_16_0_X_2025-11-13-2300/src
cmsenv

cd -

#running
cmsDriver.py -s L1TrackTrigger,L1 \
    --conditions auto:phase2_realistic_T33 \
    --geometry ExtendedRun4D110 \
    --era Phase2C17I13M9 \
    --eventcontent FEVTDEBUGHLT \
    --datatier GEN-SIM-DIGI-RAW-MINIAOD \
    --customise SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000,Configuration/DataProcessing/Utils.addMonitoring,L1Trigger/Configuration/customisePhase2TTOn110.customisePhase2TTOn110 \
    --filein das:$INPUT_DATASET \
    --fileout file:output_Phase2_L1T.root \
    --python_filename rerunL1_cfg.py \
    --inputCommands="keep *, drop l1tPFJets_*_*_*, drop l1tTrackerMuons_l1tTkMuonsGmt*_*_HLT" \
    --mc -n 1000

cmsRun rerunL1_cfg.py

cmsDriver.py  \
    step2 -s L1P2GT,HLT:75e33_trackingOnly,VALIDATION:@hltValidation \
    --conditions auto:phase2_realistic_T33 \
    --datatier DQMIO \
    --eventcontent DQMIO \
    --procModifiers phase2CAExtension,singleIterPatatrack,trackingLST,seedingLST,trackingMkFitCommon,hltTrackingMkFitInitialStep \
    --geometry ExtendedRun4D110 \
    --era Phase2C17I13M9 \
    --filein file:output_Phase2_L1T.root \
    --fileout file:output.root \
    --python_filename hltTrackingNtuple_cfg.py \
    --processName=HLTX \
    --inputCommands='keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT' \
    --outputCommands='drop *_*_*_HLT' \
    --customise Validation/RecoTrack/customiseTrackingNtuple.customiseTrackingNtupleHLT,Validation/RecoTrack/customiseTrackingNtuple.extendedContent \
    --accelerators cpu \
    --no_exec --mc -n -1wd

cmsRun hltTrackingNtuple_cfg.py

cp trackingNtuple.root /eos/user/a/aaarora/tracking/track-purity-dnn/data/output_${JOB_ID}.root