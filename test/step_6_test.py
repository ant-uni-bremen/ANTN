# This file tests the implementation of step 6, the cluster power generation. To do this, the ideal
# values for all calculations are done and the average calculation is compared to it. As step 4 already
# tests the correct creation of the LSPs Delay Spread (DS) and the Rician K Factor (K), we assume these
# to be correct here.
# Step 5 has no easily measurable output, so that a mockup 

from sionna.channel.tr38811 import utils   # The code to test
import unittest   # The test framework
from sionna.channel.tr38811 import Antenna, AntennaArray, DenseUrban, SubUrban, Urban, CDL
import numpy as np
import tensorflow as tf
import math


def create_ut_ant(carrier_frequency):
    ut_ant = Antenna(polarization="single",
                    polarization_type="V",
                    antenna_pattern="38.901",
                    carrier_frequency=carrier_frequency)
    return ut_ant

def create_bs_ant(carrier_frequency):
    bs_ant = AntennaArray(num_rows=1,
                            num_cols=4,
                            polarization="dual",
                            polarization_type="VH",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)
    return bs_ant

class Test_URB(unittest.TestCase):
# Values taken from Table 6.7.2-4a: Channel model parameters for Urban Scenario (NLOS) at S band and 
# Table 6.7.2-3a: Channel model parameters for Urban Scenario (LOS) at S band
    def test_s_band_10_degrees_dl(self):
        elevation_angle = 10.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        rays_generator = channel_model._ray_sampler
        lsp = channel_model._lsp
        #return delays, unscaled_delays
        '''
        Output
        -------
        delays : [batch size, num of BSs, num of UTs, maximum number of clusters], tf.float
            Path delays [s]

        unscaled_delays [batch size, num of BSs, num of UTs, maximum number of clusters], tf.float
            Unscaled path delays [s]
        '''

        delays, unscaled_delays = rays_generator._cluster_delays(lsp.ds, lsp.k_factor)
        #print(delays)
        #print(unscaled_delays)
        print("where")
        print("mean k is: ", tf.math.reduce_mean(lsp.k_factor))
        delays_los = tf.boolean_mask(delays, channel_model._scenario.los)
        unscaled_delays_los = tf.boolean_mask(unscaled_delays, channel_model._scenario.los)
        #print(delays_los)
        print("mean unscaled_delay los ", tf.math.reduce_mean(unscaled_delays_los))
        print("mean delay los ", tf.math.reduce_mean(delays_los))
        # Filter out values that were set to 1 by the mask to get the actual mean
        #mask_delays = tf.not_equal(delays, 1.0)
        #mask_unscaled_delays = tf.not_equal(unscaled_delays, 1.0)
        #delays = tf.boolean_mask(delays, mask_delays)
        #unscaled_delays = tf.boolean_mask(unscaled_delays, mask_unscaled_delays)
        
        #LOS_Num_Clusters = 4
        #NLOS_Num_Clusters = 3

        #delays_mean = tf.math.reduce_mean(delays[:,:,:,0:LOS_Num_Clusters-1])
        #unscaled_delays_mean = tf.math.reduce_mean(unscaled_delays[:,:,:,0:NLOS_Num_Clusters-1])
        #print("delays_mean ", delays_mean)
        #print("unscaled_delays_mean ", unscaled_delays_mean)
        #print(unscaled_delays)
        #print(delays)

        # The value for tau_dash_n in 3GPP TR38.901 (7.5-1) is sampled and this we use the expected value by integrating over the distribution
        # which leads to tau_n = -r_t*DS*(-1). For DS we again use mu_DS

        DS_log10_los = -7.97
        DS_los = 10.0**(DS_log10_los)
        r_tau_los = 2.5
        tau_n_los = DS_los * r_tau_los

        # Hoping that sufficient samples generate one close enough to 0, (7.5-2) is omitted

        #C_tau according to (7.5-3) with K being mu_K, which is directly used in dB
        K = 31.83
        C_tau = 0.7705 - 0.0433*K + 0.0002*K**2 + 0.000017*K**3

        tau_n_los_scaled = tau_n_los / C_tau

        print("tau_n_los ", tau_n_los)
        print("tau_n_los_scaled ", tau_n_los_scaled)
        #los_delays = tf.boolean_mask(unscaled_delays, channel_model._scenario.los)
        #los_delays_mean = tf.math.reduce_mean(los_delays)
        #print("los_delays_mean ", los_delays_mean)

        #print("tau_n_los_scaled ", tau_n_los_scaled)
        #los_delays_scaled = tf.boolean_mask(delays, channel_model._scenario.los)
        #los_delays_mean_scaled = tf.math.reduce_mean(los_delays_scaled)
        #print("los_delays_mean_scaled ", los_delays_mean_scaled)

        # Sionna uses 1s to mark, that a delay cluster does not exist. Thus we remove these cases, to not mess up the average

        #delays = tf.where(delays < 1.0, tf.zeros_like(delays), delays)
        #unscaled_delays = tf.where(unscaled_delays == 1.0, tf.zeros_like(unscaled_delays), unscaled_delays)
        #unscaled_delays = tf.where(unscaled_delays < 1, unscaled_delays, 0)
        #Hier macht es bloedsinn, weil irgendwie die cluster mask in rays komische werte einfugt, die da rausgefiltert werden mussen, bevor man den mean nehmen kann
        #print(delays)
        #print(unscaled_delays)

        

        #lsp_means_los = tf.boolean_mask(channel_model._scenario.lsp_log_mean, channel_model._scenario.los)


        #delays_mean = tf.math.reduce_mean(delays)
        #unscaled_delays_mean = tf.math.reduce_mean(unscaled_delays)

        
        #print("tau_n ", tau_n)
        #print("tau_n_los ", tau_n_los)

        lsp_means_los = tf.boolean_mask(channel_model._scenario.lsp_log_mean, channel_model._scenario.los)
        lsp_means_nlos = tf.boolean_mask(channel_model._scenario.lsp_log_mean, channel_model._scenario.los == False)

        lsp_means_los = tf.reduce_mean(lsp_means_los,axis=0)
        lsp_means_nlos = tf.reduce_mean(lsp_means_nlos,axis=0)

        lsp_std_los = tf.boolean_mask(channel_model._scenario.lsp_log_std, channel_model._scenario.los)
        lsp_std_nlos = tf.boolean_mask(channel_model._scenario.lsp_log_std, channel_model._scenario.los == False)

        lsp_std_los = tf.math.reduce_mean(lsp_std_los,axis=0)
        lsp_std_nlos = tf.math.reduce_mean(lsp_std_nlos,axis=0)


        DS_mean_los = lsp_means_los[0]
        ASD_mean_los = lsp_means_los[1]
        ASA_mean_los = lsp_means_los[2]
        #SF_mean_los = lsp_means_los[3] parameter already tested in step_3
        K_mean_los = lsp_means_los[4]
        ZSA_mean_los = lsp_means_los[5]
        ZSD_mean_los = lsp_means_los[6]

        DS_mean_nlos = lsp_means_nlos[0]
        ASD_mean_nlos = lsp_means_nlos[1]
        ASA_mean_nlos = lsp_means_nlos[2]
        #SF_mean_nlos = lsp_means_nlos[3] parameter already tested in step_3
        #K_mean_nlos = lsp_means_nlos[4] parameter only used in LOS scenario
        ZSA_mean_nlos = lsp_means_nlos[5]
        ZSD_mean_nlos = lsp_means_nlos[6]

        DS_std_los = lsp_std_los[0]
        ASD_std_los = lsp_std_los[1]
        ASA_std_los = lsp_std_los[2]
        SF_std_los = lsp_std_los[3]
        K_std_los = lsp_std_los[4]
        ZSA_std_los = lsp_std_los[5]
        ZSD_std_los = lsp_std_los[6]

        DS_std_nlos = lsp_std_nlos[0]
        ASD_std_nlos = lsp_std_nlos[1]
        ASA_std_nlos = lsp_std_nlos[2]
        #SF_std_nlos = lsp_std_nlos[3] parameter already tested in step_3
        #K_std_nlos = lsp_std_nlos[4] parameter only used in LOS scenario
        ZSA_std_nlos = lsp_std_nlos[5]
        ZSD_std_nlos = lsp_std_nlos[6]
        
        #Values from tables
        mu_DS_los = -7.97
        sigma_DS_los = 1.0
        mu_ASD_los = float('-inf')
        sigma_ASD_los = 0.0
        mu_ASA_los = 0.18
        sigma_ASA_los = 0.74
        #Divide mu_K by 10 as the Table in the standard is in dB
        mu_K_los = 31.83/10.0
        #Divide sigma_k by 10 as the Table in the standard is in dB
        sigma_K_los = 13.84/10.0
        mu_ZSD_los = float('-inf')
        sigma_ZSD_los = 0.0
        mu_ZSA_los = -0.63
        sigma_ZSA_los = 2.6

        mu_DS_nlos = -7.21
        sigma_DS_nlos = 1.19
        mu_ASD_nlos = float('-inf')
        sigma_ASD_nlos = 0.0
        mu_ASA_nlos = 0.17
        sigma_ASA_nlos = 2.97
        mu_ZSD_nlos = float('-inf')
        sigma_ZSD_nlos = 0.0
        mu_ZSA_nlos = -0.97
        sigma_ZSA_nlos = 2.35

        #Toleance of 0.1 for 10000 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        assert math.isclose(DS_mean_los, mu_DS_los, abs_tol=0.1)
        assert math.isclose(ASD_mean_los, mu_ASD_los, abs_tol=0.1)
        assert math.isclose(ASA_mean_los, mu_ASA_los, abs_tol=0.1)
        assert math.isclose(K_mean_los, mu_K_los, abs_tol=0.1)
        assert math.isclose(ZSD_mean_los, mu_ZSD_los, abs_tol=0.1)
        assert math.isclose(ZSA_mean_los, mu_ZSA_los, abs_tol=0.1)

        assert math.isclose(DS_std_los, sigma_DS_los, abs_tol=0.1)
        assert math.isclose(ASD_std_los, sigma_ASD_los, abs_tol=0.1)
        assert math.isclose(ASA_std_los, sigma_ASA_los, abs_tol=0.1)
        assert math.isclose(K_std_los, sigma_K_los, abs_tol=0.1)
        assert math.isclose(ZSD_std_los, sigma_ZSD_los, abs_tol=0.1)
        assert math.isclose(ZSA_std_los, sigma_ZSA_los, abs_tol=0.1)

        assert math.isclose(DS_mean_nlos, mu_DS_nlos, abs_tol=0.1)
        assert math.isclose(ASD_mean_nlos, mu_ASD_nlos, abs_tol=0.1)
        assert math.isclose(ASA_mean_nlos, mu_ASA_nlos, abs_tol=0.1)
        assert math.isclose(ZSD_mean_nlos, mu_ZSD_nlos, abs_tol=0.1)
        assert math.isclose(ZSA_mean_nlos, mu_ZSA_nlos, abs_tol=0.1)

        assert math.isclose(DS_std_nlos, sigma_DS_nlos, abs_tol=0.1)
        assert math.isclose(ASD_std_nlos, sigma_ASD_nlos, abs_tol=0.1)
        assert math.isclose(ASA_std_nlos, sigma_ASA_nlos, abs_tol=0.1)
        assert math.isclose(ZSD_std_nlos, sigma_ZSD_nlos, abs_tol=0.1)
        assert math.isclose(ZSA_std_nlos, sigma_ZSA_nlos, abs_tol=0.1)


    def test_s_band_10_degrees_ul(self):
        elevation_angle = 10.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        
        lsp_means_los = tf.boolean_mask(channel_model._scenario.lsp_log_mean, channel_model._scenario.los)
        lsp_means_nlos = tf.boolean_mask(channel_model._scenario.lsp_log_mean, channel_model._scenario.los == False)

        lsp_means_los = tf.reduce_mean(lsp_means_los,axis=0)
        lsp_means_nlos = tf.reduce_mean(lsp_means_nlos,axis=0)

        lsp_std_los = tf.boolean_mask(channel_model._scenario.lsp_log_std, channel_model._scenario.los)
        lsp_std_nlos = tf.boolean_mask(channel_model._scenario.lsp_log_std, channel_model._scenario.los == False)

        lsp_std_los = tf.math.reduce_mean(lsp_std_los,axis=0)
        lsp_std_nlos = tf.math.reduce_mean(lsp_std_nlos,axis=0)

        DS_mean_los = lsp_means_los[0]
        ASD_mean_los = lsp_means_los[1]
        ASA_mean_los = lsp_means_los[2]
        #SF_mean_los = lsp_means_los[3] parameter already tested in step_3
        K_mean_los = lsp_means_los[4]
        ZSA_mean_los = lsp_means_los[5]
        ZSD_mean_los = lsp_means_los[6]

        DS_mean_nlos = lsp_means_nlos[0]
        ASD_mean_nlos = lsp_means_nlos[1]
        ASA_mean_nlos = lsp_means_nlos[2]
        #SF_mean_nlos = lsp_means_nlos[3] parameter already tested in step_3
        #K_mean_nlos = lsp_means_nlos[4] parameter only used in LOS scenario
        ZSA_mean_nlos = lsp_means_nlos[5]
        ZSD_mean_nlos = lsp_means_nlos[6]

        DS_std_los = lsp_std_los[0]
        ASD_std_los = lsp_std_los[1]
        ASA_std_los = lsp_std_los[2]
        SF_std_los = lsp_std_los[3]
        K_std_los = lsp_std_los[4]
        ZSA_std_los = lsp_std_los[5]
        ZSD_std_los = lsp_std_los[6]

        DS_std_nlos = lsp_std_nlos[0]
        ASD_std_nlos = lsp_std_nlos[1]
        ASA_std_nlos = lsp_std_nlos[2]
        #SF_std_nlos = lsp_std_nlos[3] parameter already tested in step_3
        #K_std_nlos = lsp_std_nlos[4] parameter only used in LOS scenario
        ZSA_std_nlos = lsp_std_nlos[5]
        ZSD_std_nlos = lsp_std_nlos[6]
        
        #Values from tables
        mu_DS_los = -7.97
        sigma_DS_los = 1.0
        mu_ASD_los = -2.6
        sigma_ASD_los = 0.79
        mu_ASA_los = 0.18
        sigma_ASA_los = 0.74
        #Divide mu_K by 10 as the Table in the standard is in dB
        mu_K_los = 31.83/10.0
        #Divide sigma_k by 10 as the Table in the standard is in dB
        sigma_K_los = 13.84/10.0
        mu_ZSD_los = -2.54
        sigma_ZSD_los = 2.62
        mu_ZSA_los = -0.63
        sigma_ZSA_los = 2.6

        mu_DS_nlos = -7.21
        sigma_DS_nlos = 1.19
        mu_ASD_nlos = -1.55
        sigma_ASD_nlos = 0.87
        mu_ASA_nlos = 0.17
        sigma_ASA_nlos = 2.97
        mu_ZSD_nlos = -2.86
        sigma_ZSD_nlos = 2.77
        mu_ZSA_nlos = -0.97
        sigma_ZSA_nlos = 2.35

        #Toleance of 0.1 for 10000 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        assert math.isclose(DS_mean_los, mu_DS_los, abs_tol=0.1)
        assert math.isclose(ASD_mean_los, mu_ASD_los, abs_tol=0.1)
        assert math.isclose(ASA_mean_los, mu_ASA_los, abs_tol=0.1)
        assert math.isclose(K_mean_los, mu_K_los, abs_tol=0.1)
        assert math.isclose(ZSD_mean_los, mu_ZSD_los, abs_tol=0.1)
        assert math.isclose(ZSA_mean_los, mu_ZSA_los, abs_tol=0.1)

        assert math.isclose(DS_std_los, sigma_DS_los, abs_tol=0.1)
        assert math.isclose(ASD_std_los, sigma_ASD_los, abs_tol=0.1)
        assert math.isclose(ASA_std_los, sigma_ASA_los, abs_tol=0.1)
        assert math.isclose(K_std_los, sigma_K_los, abs_tol=0.1)
        assert math.isclose(ZSD_std_los, sigma_ZSD_los, abs_tol=0.1)
        assert math.isclose(ZSA_std_los, sigma_ZSA_los, abs_tol=0.1)

        assert math.isclose(DS_mean_nlos, mu_DS_nlos, abs_tol=0.1)
        assert math.isclose(ASD_mean_nlos, mu_ASD_nlos, abs_tol=0.1)
        assert math.isclose(ASA_mean_nlos, mu_ASA_nlos, abs_tol=0.1)
        assert math.isclose(ZSD_mean_nlos, mu_ZSD_nlos, abs_tol=0.1)
        assert math.isclose(ZSA_mean_nlos, mu_ZSA_nlos, abs_tol=0.1)

        assert math.isclose(DS_std_nlos, sigma_DS_nlos, abs_tol=0.1)
        assert math.isclose(ASD_std_nlos, sigma_ASD_nlos, abs_tol=0.1)
        assert math.isclose(ASA_std_nlos, sigma_ASA_nlos, abs_tol=0.1)
        assert math.isclose(ZSD_std_nlos, sigma_ZSD_nlos, abs_tol=0.1)
        assert math.isclose(ZSA_std_nlos, sigma_ZSA_nlos, abs_tol=0.1)

if __name__ == '__main__':
    unittest.main()