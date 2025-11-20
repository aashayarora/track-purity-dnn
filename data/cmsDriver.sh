#!/bin/bash

# INPUTS=$1

# #running
# cmsDriver.py -s L1TrackTrigger,L1 \
#     --conditions auto:phase2_realistic_T33 \
#     --geometry ExtendedRun4D110 \
#     --era Phase2C17I13M9 \
#     --eventcontent FEVTDEBUGHLT \
#     --datatier GEN-SIM-DIGI-RAW-MINIAOD \
#     --customise SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000,Configuration/DataProcessing/Utils.addMonitoring,L1Trigger/Configuration/customisePhase2TTOn110.customisePhase2TTOn110 \
#     --filein root://cmsxrootd.fnal.gov//store/mc/Phase2Spring24DIGIRECOMiniAOD/TT_TuneCP5_14TeV_powheg-pythia8/GEN-SIM-DIGI-RAW-MINIAOD/PU140_Trk1GeV_140X_mcRun4_realistic_v5-v1/2810000/00fa8c9c-f200-461c-9deb-d057c15413c3.root \
#     --fileout file:output_Phase2_L1T.root \
#     --python_filename rerunL1_cfg.py \
#     --inputCommands="keep *, drop l1tPFJets_*_*_*, drop l1tTrackerMuons_l1tTkMuonsGmt*_*_HLT" \
#     --no_exec --mc -n 1000

# cmsRun rerunL1_cfg.py

cmsDriver.py  \
    step2 -s L1P2GT,HLT:75e33_trackingOnly,VALIDATION:@hltValidation \
    --conditions auto:phase2_realistic_T33 \
    --datatier DQMIO \
    --eventcontent DQMIO \
    --procModifiers phase2CAExtension,singleIterPatatrack,trackingLST,seedingLST,trackingMkFitCommon,hltTrackingMkFitInitialStep \
    --geometry ExtendedRun4D110 \
    --era Phase2C17I13M9 \
    --filein file:/ceph/cms/store/user/mmasciov/HLTPhase2/output_TTPU_Phase2_L1T.root \
    --fileout file:/data/userdata/aaarora/hltNtuple.root \
    --python_filename hltTrackingNtuple_cfg.py \
    --processName=HLTX \
    --inputCommands='keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT' \
    --outputCommands='drop *_*_*_HLT' \
    --customise Validation/RecoTrack/customiseTrackingNtuple.customiseTrackingNtupleHLT,Validation/RecoTrack/customiseTrackingNtuple.extendedContent \
    --accelerators cpu \
    --no_exec --mc -n -1

cmsRun hltTrackingNtuple_cfg.py