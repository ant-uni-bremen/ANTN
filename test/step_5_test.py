# This file tests the implementation of step 5, the cluster delay generation. To do this, the ideal
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
from sionna.utils import log10



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

class Test_calculations(unittest.TestCase):
    def example():
        #[batch_size, num_bs, num_ut]
        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        elevation_angle = 90.0

        test_lsp = [[[1,3,4]]]
        test_k = [[[1,3,4]]]
        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        rTau_los = 2.5
        rTau_nlos = 2.3

        rays_generator = channel_model._ray_sampler
        lsp = channel_model._lsp
        
        reference_delays, reference_unscaled_delays = rays_generator._cluster_delays(lsp.ds, lsp.k_factor)
        pass


class Test_URB(unittest.TestCase):
# Values taken from Table 6.7.2-4a: Channel model parameters for Urban Scenario (NLOS) at S band and 
# Table 6.7.2-3a: Channel model parameters for Urban Scenario (LOS) at S band
    def test_s_band_10_degrees_dl(self):

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        for elevation_angle in [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0]:
            channel_model = Urban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            
            topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
            channel_model.set_topology(*topology)

            rTau_los = 2.5
            rTau_nlos = 2.3

            rays_generator = channel_model._ray_sampler
            lsp = channel_model._lsp
            
            reference_delays, reference_unscaled_delays = rays_generator._cluster_delays(lsp.ds, lsp.k_factor)
            #print(lsp.ds.shape)
            #print(lsp.k_factor.shape)
            cluster_mask = rays_generator._cluster_mask

            reference_delays_los = tf.boolean_mask(reference_delays, channel_model._scenario._los)
            reference_delays_nlos = tf.boolean_mask(reference_delays, tf.logical_not(channel_model._scenario._los))
            
            rician_k_factor = lsp.k_factor
            delay_spread = lsp.ds

            batch_size = channel_model._scenario.batch_size
            num_bs = channel_model._scenario.num_bs
            num_ut = channel_model._scenario.num_ut

            num_clusters_max = channel_model._scenario.num_clusters_max

            delay_scaling_parameter = tf.where(channel_model._scenario._los, rTau_los, rTau_nlos)
            delay_scaling_parameter = tf.expand_dims(delay_scaling_parameter,axis=3)
            
            delay_spread = tf.expand_dims(delay_spread, axis=3)
            x = tf.random.uniform(shape=[batch_size, num_bs, num_ut,
                num_clusters_max], minval=1e-6, maxval=1.0,
                dtype=channel_model._scenario.dtype.real_dtype)
            
            unscaled_delays = -delay_scaling_parameter*delay_spread*tf.math.log(x)

            unscaled_delays = (unscaled_delays*(1.-cluster_mask) + cluster_mask)

            unscaled_delays = unscaled_delays - tf.reduce_min(unscaled_delays, axis=3, keepdims=True)
            unscaled_delays = tf.sort(unscaled_delays, axis=3)
            
            rician_k_factor_db = 10.0*log10(rician_k_factor) # to dB
            scaling_factor = (0.7705 - 0.0433*rician_k_factor_db
                + 0.0002*tf.square(rician_k_factor_db)
                + 0.000017*tf.math.pow(rician_k_factor_db, tf.constant(3.,
                channel_model._scenario.dtype.real_dtype)))
            scaling_factor = tf.expand_dims(scaling_factor, axis=3)
            delays = tf.where(tf.expand_dims(channel_model._scenario.los, axis=3),
                unscaled_delays / scaling_factor, unscaled_delays)

            delays_los = tf.boolean_mask(delays, channel_model._scenario._los)
            delays_nlos = tf.boolean_mask(delays, tf.logical_not(channel_model._scenario._los))


            assert tf.abs(tf.reduce_mean(reference_delays_los) - tf.reduce_mean(delays_los)) < 1e-5
            assert tf.abs(tf.reduce_mean(reference_delays_nlos) - tf.reduce_mean(delays_nlos)) < 1e-5
            print(lsp.ds.shape)
            print(lsp.k_factor.shape)
            #print("reference_delays_los ", tf.math.reduce_std(reference_delays_los))
            #print("delays_los ", tf.math.reduce_std(delays_los))

            assert tf.abs(tf.math.reduce_std(reference_delays_los) - tf.math.reduce_std(delays_los)) < 1e-5
            assert tf.abs(tf.math.reduce_std(reference_delays_nlos) - tf.math.reduce_std(delays_nlos)) < 1e-5


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