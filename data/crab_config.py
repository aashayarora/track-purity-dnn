from WMCore.Configuration import Configuration

def config():
    config = Configuration()  # pylint: disable=redefined-outer-name
    config.section_("General")
    config.section_("JobType")
    config.section_("Data")
    config.section_("Site")
    config.section_("User")
    config.section_("Debug")
    return config

datasets = {
    "BstoPhiPhi": "/BsToPhiPhi_4K_TuneCP5_14TeV-pythia8-evtgen/Phase2Spring24DIGIRECOMiniAOD-PU200_AllTP_140X_mcRun4_realistic_v4-v1/GEN-SIM-DIGI-RAW-MINIAOD",
    "BsToTauTau": "/BsToTauTau_3Pi_SoftQCDnonD_TuneCP5_14TeV-pythia8-evtgen/Phase2Spring24DIGIRECOMiniAOD-PU200_AllTP_140X_mcRun4_realistic_v4-v1/GEN-SIM-DIGI-RAW-MINIAOD",
    "DisplacedSUSY_1000mm": "/DisplacedSUSY_stopToBottom_M-800_1000mm_TuneCP5_14TeV-pythia8/Phase2Spring24DIGIRECOMiniAOD-PU200_AllTP_140X_mcRun4_realistic_v4-v2/GEN-SIM-DIGI-RAW-MINIAOD",
    "DisplacedSUSY_500mm": "/DisplacedSUSY_stopToBottom_M-800_500mm_TuneCP5_14TeV-pythia8/Phase2Spring24DIGIRECOMiniAOD-PU200_AllTP_140X_mcRun4_realistic_v4-v1/GEN-SIM-DIGI-RAW-MINIAOD",
    "DisplacedSUSY_50mm": "/DisplacedSUSY_stopToBottom_M-800_50mm_TuneCP5_14TeV-pythia8/Phase2Spring24DIGIRECOMiniAOD-PU200_AllTP_140X_mcRun4_realistic_v4-v1/GEN-SIM-DIGI-RAW-MINIAOD",
    "HH_2B2Tau": "/GluGluToHHTo2B2Tau_node_SM_TuneCP5_14TeV-madgraph-pythia8/Phase2Spring24DIGIRECOMiniAOD-PU200_AllTP_140X_mcRun4_realistic_v4-v2/GEN-SIM-DIGI-RAW-MINIAOD",
    "HH_4B": "/GluGluToHHTo4B_node_SM_TuneCP5_14TeV-amcatnlo-pythia8/Phase2Spring24DIGIRECOMiniAOD-PU200_AllTP_140X_mcRun4_realistic_v4-v1/GEN-SIM-DIGI-RAW-MINIAOD",
    "HToPhiGamma": "/HToPhiGammaToKK_TuneCP5_14TeV-pythia8/Phase2Spring24DIGIRECOMiniAOD-PU200_AllTP_140X_mcRun4_realistic_v4-v1/GEN-SIM-DIGI-RAW-MINIAOD",
    "HToRhoGamma": "/HToRhoGammaToPiPi_TuneCP5_14TeV-pythia8/Phase2Spring24DIGIRECOMiniAOD-PU200_AllTP_140X_mcRun4_realistic_v4-v1/GEN-SIM-DIGI-RAW-MINIAOD",
    "TStauStau_100_100mm": "/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_14TeV_madgraphMLM-pythia8/Phase2Spring24DIGIRECOMiniAOD-PU200_AllTP_140X_mcRun4_realistic_v4-v1/GEN-SIM-DIGI-RAW-MINIAOD",
    "TStauStau_100_10mm": "/SMS-TStauStau_MStau-100_ctau-10mm_mLSP-1_TuneCP5_14TeV_madgraphMLM-pythia8/Phase2Spring24DIGIRECOMiniAOD-PU200_AllTP_140X_mcRun4_realistic_v4-v1/GEN-SIM-DIGI-RAW-MINIAOD",
    "TStauStau_200_100mm": "/SMS-TStauStau_MStau-200_ctau-100mm_mLSP-1_TuneCP5_14TeV_madgraphMLM-pythia8/Phase2Spring24DIGIRECOMiniAOD-PU200_AllTP_140X_mcRun4_realistic_v4-v1/GEN-SIM-DIGI-RAW-MINIAOD",
    "TStauStau_200_10mm": "/SMS-TStauStau_MStau-200_ctau-10mm_mLSP-1_TuneCP5_14TeV_madgraphMLM-pythia8/Phase2Spring24DIGIRECOMiniAOD-PU200_AllTP_140X_mcRun4_realistic_v4-v2/GEN-SIM-DIGI-RAW-MINIAOD",
    "SinglePhoton": "/SinglePhoton_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1/GEN-SIM-DIGI-RAW-MINIAOD",
    "SinglePion": "/SinglePion_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1/GEN-SIM-DIGI-RAW-MINIAOD",
    "TT": "/TT_TuneCP5_14TeV-powheg-pythia8/Phase2Spring24DIGIRECOMiniAOD-PU200_AllTP_140X_mcRun4_realistic_v4-v1/GEN-SIM-DIGI-RAW-MINIAOD",
    "WTo3Pi": "/WTo3Pi_TuneCP5_14TeV_pythia8/Phase2Spring24DIGIRECOMiniAOD-PU200_AllTP_140X_mcRun4_realistic_v4-v1/GEN-SIM-DIGI-RAW-MINIAOD",
    "ZToPhiGamma": "/ZToPhiGammaToKK_TuneCP5_14TeV-pythia8/Phase2Spring24DIGIRECOMiniAOD-PU200_AllTP_140X_mcRun4_realistic_v4-v1/GEN-SIM-DIGI-RAW-MINIAOD",
    "ZToRhoGamma": "/ZToRhoGammaToPiPi_TuneCP5_14TeV-pythia8/Phase2Spring24DIGIRECOMiniAOD-PU200_AllTP_140X_mcRun4_realistic_v4-v2/GEN-SIM-DIGI-RAW-MINIAOD"
}

key = "TT"

config = config()
config.General.workArea                 = 'crab_projects/trackingNtuples/'
config.General.transferOutputs          = True
config.General.transferLogs             = False

config.JobType.sendExternalFolder       = True
config.JobType.pluginName               = 'Analysis'
config.JobType.psetName                 = 'rerunL1_cfg.py'
config.JobType.numCores                 = 4
config.JobType.maxMemoryMB              = 8000

config.Data.ignoreLocality              = True
config.Data.splitting                   = 'EventAwareLumiBased'
config.Data.unitsPerJob                 = 40
config.Data.totalUnits                  = -1
config.Data.outLFNDirBase               = '/store/user/aaarora/trackingNtuples/'
config.Data.inputDataset                = datasets[key]

config.JobType.outputFiles              = ['trackingNtuple.root']
config.JobType.allowUndistributedCMSSW  = True

config.section_('Site')
config.Site.storageSite = 'T2_US_UCSD'

config.section_('Debug')
config.Debug.extraJDL = ['+CMS_ALLOW_OVERFLOW=False']
config.Site.whitelist = ['T2_US_*']