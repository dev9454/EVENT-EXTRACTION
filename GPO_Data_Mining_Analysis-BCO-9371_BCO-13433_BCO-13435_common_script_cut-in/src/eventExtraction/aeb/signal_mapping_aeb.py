# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:35:19 2024

@author: mfixlz
"""
from pathlib import Path
if __package__ is None:
    import sys
    from os import path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(1, str(Path(__file__).resolve().parent))
    from utils.utils_generic import read_platform, loadmat
else:

    # from .. import utils
    from utils.utils_generic import read_platform, loadmat


class signalMapping:

    def __init__(self, raw_data) -> None:

        self.raw_data = raw_data

        self.possible_arch = ['WL', 'JLBUX', ]
        self.arch_path_methods = [self._WL_signal_paths,
                                  self._JLBUX_signal_paths,
                                  ]
        self.architechture_mapping = {key: val for key, val in
                                      zip(self.possible_arch,
                                          self.arch_path_methods)
                                      }

        self.req_signals_IW = ['veh_index_IW', 'cTime',
                               ]
        self.req_signals_VSE = ['veh_index_VSE', 'cTime',
                                ]
        self.req_signals_TSEL = ['host_compensated_speed',  # host config
                                 'host_vcs_long_vel',  # host config
                                 'host_vcs_lat_vel',  # host config
                                 'vse_index',  # host config
                                 'track_id_RT',  # sata config
                                 'vcs_long_posn_RT', 'vcs_lat_posn_RT',  # sata config 
                                 'vcs_long_vel_RT', 'vcs_lat_vel_RT',  # sata config
                                 'track_id_RTS',  # sata config
                                 'vcs_long_posn_RTS', 'vcs_lat_posn_RTS',  # sata config
                                 'vcs_long_vel_RTS', 'vcs_lat_vel_RTS',  # sata config
                                 'track_id_PCA',  # sata config
                                 'vcs_long_posn_PCA', 'vcs_lat_posn_PCA',  # sata config
                                 'vcs_long_vel_PCA', 'vcs_lat_vel_PCA',  # sata config
                                 'track_id_PCAS',  # sata config
                                 'vcs_long_posn_PCAS', 'vcs_lat_posn_PCAS',  # sata config
                                 'vcs_long_vel_PCAS', 'vcs_lat_vel_PCAS',  # sata config
                                 'track_id_SCP',  # sata config
                                 'vcs_long_posn_SCP', 'vcs_lat_posn_SCP',  # sata config
                                 'vcs_long_vel_SCP', 'vcs_lat_vel_SCP',  # sata config
                                 'track_id_OCTAP',  # sata config
                                 'vcs_long_posn_OCTAP', 'vcs_lat_posn_OCTAP',  # sata config
                                 'vcs_long_vel_OCTAP', 'vcs_lat_vel_OCTAP',  # sata config
                                 'track_id_VRU',  # sata config
                                 'vcs_long_posn_VRU', 'vcs_lat_posn_VRU',  # sata config
                                 'vcs_long_vel_VRU', 'vcs_lat_vel_VRU',  # sata config
                                 'cTime',
                                 ]
        self.req_signals_FDCAN14_info = ['AEB_Sensitivity_Setting', 'AEB_Systemsts',
                                        'Brake_Jerk_Req', 'LC_Jerk', 'Prefill_Req',
                                        'Intervention', 'FDCAN14_info_AEB_Type', 'cTime',
                                        ]
        
        self.req_signals_FDCAN14_HMI = [
                                        'AEB_DispPopupSts_FDCAN14_HMI', 'cTime',
                                        ]
        
        self.req_signals_fusion = ['cTime',
                                   ]
        self.req_signals_INST = ['FCW_Track_ID', 'FCW_TTC',
                                 'FCW_Prefill_Req', 'FCW_AEB_DispPopupSts',
                                 'FCW_Brake_Jerk_Req', 'FCW_AEB_Type',
                                 'FCW_Fusion_Source', 'FCW_CmbbPrimaryConfidence',
                                 'AEB_Track_ID', 'AEB_TTC',
                                 'AEB_Prefill_Req', 'AEB_AEB_DispPopupSts',
                                 'AEB_Brake_Jerk_Req', 'AEB_AEB_Type',
                                 'AEB_Fusion_Source', 'AEB_CmbbPrimaryConfidence',
                                 'PEB_Track_ID', 'PEB_Target_TTC',
                                 'PEB_Prefill_Req', 'PEB_AEB_DispPopupSts',
                                 'PEB_Brake_Jerk_Req', 'PEB_AEB_Type',
                                 'PEB_Fusion_Source', 'PEB_CmbbPrimaryConfidence',
                                 'ICA_Track_ID', 'ICA_SCP_TTC_Masked',
                                 'ICA_OCTAP_TTC_Masked',
                                 'ICA_Prefill_Req', 'ICA_AEB_DispPopupSts',
                                 'ICA_Brake_Jerk_Req', 'ICA_AEB_Type',
                                 'ICA_Fusion_Source', 'ICA_CmbbPrimaryConfidence',
                                 'cTime',
                                 'f_BrkPrefill_FCW', 'f_BrkPrefill_PEB',
                                 'f_BrkPrefill_ICA', 'AEB_DispPopupSts_FCW_Warning_Type',
                                 'CA_manager_Brake_Jerk_Req', 'CA_manager_AEB_Type',
                                 'f_ABA_FCW_AEB_Type', 'f_ABA_ICA_AEB_Type'

                                 ]  # target config
        
        

    def _JLBUX_signal_paths(self, ):

        IW_path_list = []
        VSE_path_list = []
        TSEL_path_list = []
        FDCAN14_path_list = []
        FUS_path_list = []
        INST_path_list = []

        return (IW_path_list, VSE_path_list,
                TSEL_path_list, FDCAN14_path_list,
                FUS_path_list, INST_path_list)
    def _enum_values_JLBUX(self, ):

        return_enums_dict = {}

        return return_enums_dict

    def _WL_signal_paths(self, ):

        IW_path_list = ['mudp.inst.IW.IW_Logging_Msg.fusion_index',
                        'mudp.inst.IW.header.cTime',
                        ]
        VSE_path_list = ['mudp.VSE.vse_out.veh_index',
                         'mudp.VSE.header.cTime',
                         ]
        TSEL_path_list = ['mudp.tsel.commonSATAInfo.host_compensated_speed',
                          'mudp.tsel.commonSATAInfo.host_vcs_long_vel',
                          'mudp.tsel.commonSATAInfo.host_vcs_lat_vel',
                          'mudp.tsel.procInfo.vse_index',

                          'mudp.tsel.accMovingTracks.track_id',
                          'mudp.tsel.accMovingTracks.vcs_long_posn',
                          'mudp.tsel.accMovingTracks.vcs_lat_posn',
                          'mudp.tsel.accMovingTracks.vcs_long_vel',
                          'mudp.tsel.accMovingTracks.vcs_lat_vel',

                          'mudp.tsel.accStationaryTracks.track_id',
                          'mudp.tsel.accStationaryTracks.vcs_long_posn',
                          'mudp.tsel.accStationaryTracks.vcs_lat_posn',
                          'mudp.tsel.accStationaryTracks.vcs_long_vel',
                          'mudp.tsel.accStationaryTracks.vcs_lat_vel',

                          'mudp.tsel.pcaMovingTrack.track_id',
                          'mudp.tsel.pcaMovingTrack.vcs_long_posn',
                          'mudp.tsel.pcaMovingTrack.vcs_lat_posn',
                          'mudp.tsel.pcaMovingTrack.vcs_long_vel',
                          'mudp.tsel.pcaMovingTrack.vcs_lat_vel',

                          'mudp.tsel.pcaStationaryTrack.track_id',
                          'mudp.tsel.pcaStationaryTrack.vcs_long_posn',
                          'mudp.tsel.pcaStationaryTrack.vcs_lat_posn',
                          'mudp.tsel.pcaStationaryTrack.vcs_long_vel',
                          'mudp.tsel.pcaStationaryTrack.vcs_lat_vel',

                          'mudp.tsel.scpTracks.track_id',
                          'mudp.tsel.scpTracks.vcs_long_posn',
                          'mudp.tsel.scpTracks.vcs_lat_posn',
                          'mudp.tsel.scpTracks.vcs_long_vel',
                          'mudp.tsel.scpTracks.vcs_lat_vel',

                          'mudp.tsel.octapTracks.track_id',
                          'mudp.tsel.octapTracks.vcs_long_posn',
                          'mudp.tsel.octapTracks.vcs_lat_posn',
                          'mudp.tsel.octapTracks.vcs_long_vel',
                          'mudp.tsel.octapTracks.vcs_lat_vel',

                          'mudp.tsel.vruTracks.track_id',
                          'mudp.tsel.vruTracks.vcs_long_posn',
                          'mudp.tsel.vruTracks.vcs_lat_posn',
                          'mudp.tsel.vruTracks.vcs_long_vel',
                          'mudp.tsel.vruTracks.vcs_lat_vel',

                          'mudp.tsel.header.cTime',
                          ]
        FDCAN14_info_path_list = ['dvlExtDBC.FDCAN14.ADAS_FD_INFO.AEB_Sensitivity_Setting',
                                'dvlExtDBC.FDCAN14.ADAS_FD_INFO.AEB_Systemsts',
                                'dvlExtDBC.FDCAN14.ADAS_FD_INFO.Brake_Jerk_Req',
                                'dvlExtDBC.FDCAN14.ADAS_FD_INFO.LC_Jerk',
                                'dvlExtDBC.FDCAN14.ADAS_FD_INFO.Prefill_Req',
                                'dvlExtDBC.FDCAN14.ADAS_FD_INFO.Intervention',
                                'dvlExtDBC.FDCAN14.ADAS_FD_INFO.AEB_Type',
                                'dvlExtDBC.FDCAN14.ADAS_FD_INFO.cTime',
                                ]
        
        FDCAN14_HMI_path_list = [ 'dvlExtDBC.FDCAN14.ADAS_FD_HMI.AEB_DispPopupSts',
                                'dvlExtDBC.FDCAN14.ADAS_FD_HMI.cTime',
                                ]

        FUS_path_list = ['mudp.fus.header.cTime']

        INST_path_list = ['mudp.inst.CA.CA_Logging_Msg.FCW_AEB_Log.FCW_Track_ID',
                          'mudp.inst.CA.CA_Logging_Msg.FCW_AEB_Log.FCW_TTC',
                          'mudp.inst.CA.CA_Logging_Msg.FCW_AEB_Log.TTC_Threshold_Pre_Warn',
                          'mudp.inst.CA.CA_Logging_Msg.FCW_AEB_Log.TTC_Threshold_Pre_Warn',
                          'mudp.inst.CA.CA_Logging_Msg.FCW_AEB_Log.TTC_Threshold_Pre_Warn',
                          'mudp.inst.CA.CA_Logging_Msg.FCW_AEB_Log.TTC_Threshold_Pre_Warn',
                          'mudp.inst.CA.CA_Logging_Msg.FCW_AEB_Log.FCW_Fusion_Source',
                          'mudp.inst.CA.CA_Logging_Msg.FCW_AEB_Log.FCW_CmbbPrimaryConfidence',

                          'mudp.inst.CA.CA_Logging_Msg.FCW_AEB_Log.AEB_Track_ID',
                          'mudp.inst.CA.CA_Logging_Msg.FCW_AEB_Log.AEB_TTC',
                          'mudp.inst.CA.CA_Logging_Msg.FCW_AEB_Log.LowBrake_TTC_Threshold',
                          'mudp.inst.CA.CA_Logging_Msg.FCW_AEB_Log.LowBrake_TTC_Threshold',
                          'mudp.inst.CA.CA_Logging_Msg.FCW_AEB_Log.LowBrake_TTC_Threshold',
                          'mudp.inst.CA.CA_Logging_Msg.FCW_AEB_Log.LowBrake_TTC_Threshold',
                          'mudp.inst.CA.CA_Logging_Msg.FCW_AEB_Log.AEB_Fusion_Source',
                          'mudp.inst.CA.CA_Logging_Msg.FCW_AEB_Log.AEB_CmbbPrimaryConfidence',

                          'mudp.inst.CA.CA_Logging_Msg.PEB_Logger.Target_Track_ID',
                          'mudp.inst.CA.CA_Logging_Msg.PEB_Logger.Target_TTC',
                          'mudp.inst.CA.CA_Logging_Msg.PEB_Logger.Prefill_TTC_Threshold', #Prefill_TTC_Threshold
                          'mudp.inst.CA.CA_Logging_Msg.PEB_Logger.Warning_TTC_Threshold', #Warning_TTC_Threshold
                          'mudp.inst.CA.CA_Logging_Msg.PEB_Logger.TTC_Threshold',
                          'mudp.inst.CA.CA_Logging_Msg.PEB_Logger.TTC_Threshold',
                          'mudp.inst.CA.CA_Logging_Msg.PEB_Logger.Target_DetectionSensor',
                          'mudp.inst.CA.CA_Logging_Msg.PEB_Logger.Target_Cmbb_Conf',

                          'mudp.inst.CA.CA_Logging_Msg.ICA_Target.Track_ID',
                          'mudp.inst.CA.CA_Logging_Msg.ICA_Target.ICA_SCP_TTC_Masked',
                          'mudp.inst.CA.CA_Logging_Msg.ICA_Target.ICA_OCTAP_TTC_Masked',
                          'mudp.inst.CA.CA_Logging_Msg.ICA_Target.ICA_Warning_TTCThrsh',
                          'mudp.inst.CA.CA_Logging_Msg.ICA_Target.ICA_Warning_TTCThrsh',
                          'mudp.inst.CA.CA_Logging_Msg.ICA_Target.ICA_Brake_TTCThrsh',
                          'mudp.inst.CA.CA_Logging_Msg.ICA_Target.ICA_Brake_TTCThrsh',
                          'mudp.inst.CA.CA_Logging_Msg.ICA_Target.DetectionSensor',
                          'mudp.inst.CA.CA_Logging_Msg.ICA_Target.CmbbPrimaryConfidence',

                          'mudp.inst.CA.header.cTime',

                          'mudp.inst.CA.CA_Logging_Msg.CA_Manager_Log.f_BrkPrefill_FCW',
                          'mudp.inst.CA.CA_Logging_Msg.CA_Manager_Log.f_BrkPrefill_PEB',
                          'mudp.inst.CA.CA_Logging_Msg.CA_Manager_Log.f_BrkPrefill_ICA',
                          'mudp.inst.CA.CA_Logging_Msg.CA_Manager_Log.FCW_Warning_Type',
                          'mudp.inst.CA.CA_Logging_Msg.CA_Manager_Log.Brake_Jerk_Req',
                          'mudp.inst.CA.CA_Logging_Msg.CA_Manager_Log.AEB_Type',
                          'mudp.inst.CA.CA_Logging_Msg.CA_Manager_Log.f_ABA_FCW',
                          'mudp.inst.CA.CA_Logging_Msg.CA_Manager_Log.f_ABA_ICA',
                          ]


        return (IW_path_list, VSE_path_list,
                TSEL_path_list, FDCAN14_info_path_list, 
                FDCAN14_HMI_path_list,
                FUS_path_list, INST_path_list)

    def _get_signal_paths(self, ):

        self.architecture = read_platform(self.raw_data)

        if self.architecture in self.possible_arch:

            return_val = self.architechture_mapping[self.architecture]()
        else:
            raise Exception(f'Found exception \n {self.architecture}')
            return_val = None

        return return_val
    
    def _enum_values(self, ):

        if self.architecture == 'WL':
            return_enums = self._enum_values_WL()
        elif self.architecture == 'JLBUX':
            return_enums = self._enum_values_JLBUX()

        return return_enums

    
    def _enum_values_WL(self, ):

        return_enums_dict = {
                            'Prefill_Req' : {
                                0: {'name': 'No_Prefill_Requested', 
                                    'aeb_type': 'None'},
                                1: {'name': 'Prefill_Requested', 
                                    'aeb_type': 'FCW'}
                                },
                            'f_BrkPrefill_FCW' : {
                                0: {'name': 'Not_Active', 
                                    'aeb_type': 'None'},
                                1: {'name': 'Brake_Prefill_FCW', 
                                    'aeb_type': 'FCW'}
                                },
                            'f_BrkPrefill_PEB' : {
                                0: {'name': 'Not_Active', 
                                    'aeb_type': 'None'},
                                1: {'name': 'Brake_Prefill_PEB', 
                                    'aeb_type': 'PEB'}
                                },
                            'f_BrkPrefill_ICA' : {
                                0: {'name': 'Not_Active', 
                                    'aeb_type': 'None'},
                                1: {'name': 'Brake_Prefill_ICA', 
                                    'aeb_type': 'ICA'}
                                },
                            'AEB_DispPopupSts_FCW_Warning_Type' : {
                                0: {'name': 'None', 
                                    'aeb_type': 'None'},
                                1: {'name': 'Pre-Warning', 
                                    'aeb_type': 'FCW'},
                                2: {'name': 'Acute-Warning', 
                                    'aeb_type': 'FCW'},
                                3: {'name': 'PEB Warning', 
                                    'aeb_type': 'PEB'},
                                4: {'name': 'ICA Warning L', 
                                    'aeb_type': 'ICA'},
                                5: {'name': 'ICA Warning R', 
                                    'aeb_type': 'ICA'},
                                },
                            'AEB_DispPopupSts_FDCAN14_HMI' : {
                                0: {'name': 'No_Code', 
                                    'aeb_type': 'None'},
                                1: {'name': 'Pre_Intervention_Warning', 
                                    'aeb_type': 'FCW'},
                                4: {'name': 'Braking_Complete', 
                                    'aeb_type': 'HMI'},
                                5: {'name': 'AEB_Unavailable_ESC_Off', 
                                    'aeb_type': 'HMI'},
                                6: {'name': 'AEB_Unavailable_4WD_Low', 
                                    'aeb_type': 'HMI'},
                                7: {'name': 'Active_Braking_Enabled', 
                                    'aeb_type': 'HMI'},
                                8: {'name': 'Active_Braking_Disabled', 
                                    'aeb_type': 'HMI'},
                                9: {'name': 'AEB_User_Disabled', 
                                    'aeb_type': 'HMI'},
                                10: {'name': 'AEB_Radar_Blind', 
                                     'aeb_type': 'HMI'},
                                11: {'name': 'Not_used1', 
                                     'aeb_type': 'HMI'},
                                12: {'name': 'AEB_Fail', 
                                     'aeb_type': 'HMI'},
                                13: {'name': 'Not_used2', 
                                     'aeb_type': 'HMI'},
                                14: {'name': 'AEB_Cam_Blind', 
                                     'aeb_type': 'HMI'},
                                15: {'name': 'Not_used3', 
                                     'aeb_type': 'HMI'},
                                16: {'name': 'AEB_Limited_Fail', 
                                     'aeb_type': 'HMI'},
                                17: {'name': 'PEB_Off', 
                                     'aeb_type': 'HMI'},
                                18: {'name': 'PEB_On', 
                                     'aeb_type': 'HMI'},
                                19: {'name': 'PEB_Radar_Blind', 
                                     'aeb_type': 'HMI'},
                                20: {'name': 'PEB_Fail', 
                                     'aeb_type': 'HMI'},
                                21: {'name': 'PEB_Cam_Blind', 
                                     'aeb_type': 'HMI'},
                                22: {'name': 'FCW_Limited_PEB_Off_Camera_Blinded', 
                                     'aeb_type': 'HMI'},
                                23: {'name': 'FCW_Limited_PEB_Off_Camera_Fail', 
                                     'aeb_type': 'HMI'},
                                24: {'name': 'Pre_Intervention_Warning_ICA_L', 
                                     'aeb_type': 'ICA'},
                                25: {'name': 'Pre_Intervention_Warning_ICA_R', 
                                     'aeb_type': 'ICA'},
                                26: {'name': 'AEB_Limit_Corner_blind', 
                                     'aeb_type': 'HMI'},
                                },
                            'Brake_Jerk_Req' : {
                                0: {'name': 'Not_Active', 
                                    'aeb_type': 'None'},
                                1: {'name': 'Active', 
                                    'aeb_type': 'AEB'}
                                },
                            'LC_Jerk' : {
                                0: {'name': 'Not_Active', 
                                    'aeb_type': 'None'},
                                1: {'name': 'LC_Jerk', 
                                    'aeb_type': 'AEB'}
                                },
                            'CA_manager_Brake_Jerk_Req' : {
                                0: {'name': 'Not_Active', 
                                    'aeb_type': 'None'},
                                1: {'name': 'Active', 
                                    'aeb_type': 'AEB'}
                                },
                            'FDCAN14_info_AEB_Type' : {
                                0: {'name': 'None', 'aeb_type': 'None'},
                                1: {'name': 'ACC', 'aeb_type': 'AEB'},
                                2: {'name': 'CMS', 'aeb_type': 'AEB'},
                                3: {'name': 'CMS_XTD', 'aeb_type': 'AEB'},
                                4: {'name': 'ABA', 'aeb_type': 'AEB'},
                                5: {'name': 'LSCM', 'aeb_type': 'AEB'},
                                6: {'name': 'PEB', 'aeb_type': 'PEB'},
                                7: {'name': 'ICA_L', 'aeb_type': 'ICA'},
                                8: {'name': 'ICA_R', 'aeb_type': 'ICA'},
                                9: {'name': 'HAS_MRM', 'aeb_type': 'AEB'},
                                15: {'name': 'SNA', 'aeb_type': 'AEB'},
                                },
                            'CA_manager_AEB_Type' : {
                                0: {'name': 'None', 'aeb_type': 'None'},
                                1: {'name': 'ACC', 'aeb_type': 'AEB'},
                                2: {'name': 'CMS', 'aeb_type': 'AEB'},
                                3: {'name': 'CMS_XTD', 'aeb_type': 'AEB'},
                                4: {'name': 'ABA', 'aeb_type': 'AEB'},
                                5: {'name': 'LSCM', 'aeb_type': 'AEB'},
                                6: {'name': 'PEB', 'aeb_type': 'PEB'},
                                7: {'name': 'ICA_L', 'aeb_type': 'ICA'},
                                8: {'name': 'ICA_R', 'aeb_type': 'ICA'},
                                9: {'name': 'HAS_MRM', 'aeb_type': 'AEB'},
                                15: {'name': 'SNA', 'aeb_type': 'AEB'},
                                },
                            'f_ABA_FCW_AEB_Type' : {
                                0: {'name': 'Not_Active', 
                                    'aeb_type': 'None'},
                                1: {'name': 'ABA-FCW', 
                                    'aeb_type': 'AEB'}
                                },
                            'f_ABA_ICA_AEB_Type' : {
                                0: {'name': 'Not_Active', 
                                    'aeb_type': 'None'},
                                1: {'name': 'ABA-ICA', 
                                    'aeb_type': 'ICA'}
                                },

                            }

        return return_enums_dict
    

    def _target_type_mapping(self, ):

        target_types = ['RT', 'RTS', 
                             'PCA', 'PCAS', 
                             'SCP',  'OCTAP', 
                             'VRU']
        
        # sata_suffixes

        return target_types

    def _signal_mapping(self, ):

        (IW_path_list, VSE_path_list,
         TSEL_path_list, FDCAN14_info_path_list, 
        FDCAN14_HMI_path_list,
         FUS_path_list, INST_path_list) = self._get_signal_paths()

        IW_map = {key: val for key, val in
                  zip(self.req_signals_IW, IW_path_list)
                  }
        VSE_map = {key: val for key, val in
                   zip(self.req_signals_VSE, VSE_path_list)
                   }
        TSEL_map = {key: val for key, val in
                    zip(self.req_signals_TSEL, TSEL_path_list)
                    }
        FDCAN14_info_map = {key: val for key, val in
                       zip(self.req_signals_FDCAN14_info, 
                           FDCAN14_info_path_list)
                       }
        FDCAN14_HMI_map = {key: val for key, val in
                       zip(self.req_signals_FDCAN14_HMI, 
                           FDCAN14_HMI_path_list)
                       }
        FUS_map = {key: val for key, val in
                   zip(self.req_signals_fusion, FUS_path_list)
                   }
        INST_map = {key: val for key, val in
                    zip(self.req_signals_INST, INST_path_list)
                    }

        _data_signal_map = {'IW_map': IW_map,
                            'VSE_map': VSE_map,
                            'TSEL_map': TSEL_map,
                            'FDCAN14_info_map': FDCAN14_info_map,
                            'FDCAN14_HMI_map' : FDCAN14_HMI_map,
                            'FUS_map': FUS_map,
                            'INST_map': INST_map
                            }
        _data_enum_map = self._enum_values()
        _target_type_list = self._target_type_mapping()

        return _data_signal_map, _data_enum_map, _target_type_list


if __name__ == '__main__':

    import warnings
    import os
    from pathlib import Path
    warnings.filterwarnings("ignore")

    file_name_resim = os.path.join(
        Path(os.getcwd()).parent,
        # os.path.dirname(
        #     os.path.dirname(
        #         os.getcwd())),
        'data', 'WL',
        'FCAWL_20210826_VPCSkipper_RWUP_HWY_PAR_PAR_MS_AD_165229_094.mat')

    mat_file_data = loadmat(file_name_resim)
    AEB_signal_map_obj = signalMapping(mat_file_data)

    signal_map, enum_map, target_type_list = \
        AEB_signal_map_obj._signal_mapping()
