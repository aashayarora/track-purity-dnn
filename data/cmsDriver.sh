#!/bin/bash

cmsDriver.py  \
    step2 -s L1P2GT,HLT:75e33_trackingOnly,VALIDATION:@hltValidation \
    --conditions auto:phase2_realistic_T33 \
    --datatier DQMIO \
    --eventcontent DQMIO \
    --procModifiers phase2CAExtension,singleIterPatatrack,trackingLST,seedingLST,trackingMkFitCommon,hltTrackingMkFitInitialStep \
    --geometry ExtendedRun4D110 \
    --era Phase2C17I13M9 \
    --filein file:/ceph/cms/store/user/mmasciov/HLTPhase2/output_TTPU_Phase2_L1T.root \
    --fileout file:output.root \
    --python_filename hltTrackingNtuple_cfg.py \
    --processName=HLTX \
    --inputCommands='keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT' \
    --outputCommands='drop *_*_*_HLT' \
    --customise Validation/RecoTrack/customiseTrackingNtuple.customiseTrackingNtupleHLT,Validation/RecoTrack/customiseTrackingNtuple.extendedContent \
    --accelerators cpu \
    --no_exec --mc -n -1

cmsRun hltTrackingNtuple_cfg.py