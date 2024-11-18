# This file tests the correct calculation of the Link Budget for the Scnerios described in 3GPP TR38.821 Table 6.1.3.3-1: Link budgets results

from sionna.channel.tr38811 import utils   # The code to test
import unittest   # The test framework
from sionna.channel.tr38811 import Antenna, AntennaArray, DenseUrban, SubUrban, Urban, CDL
  
class Test_DL(unittest.TestCase):

    def test_sc1_dl(self, direction = "downlink", scenario = "sur", carrier_frequency = 20e9,satellite_distance = 35786000.0, elevation_angle = 12.5, batch_size = 1,num_ut = 1):        
        
        ut_array = Antenna(polarization="single",
                    polarization_type="V",
                    antenna_pattern="38.901",
                    carrier_frequency=carrier_frequency)
        
        bs_array = AntennaArray(num_rows=1,
                            num_cols=4,
                            polarization="dual",
                            polarization_type="VH",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)
        
        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)

        topology = utils.gen_single_sector_topology(batch_size=batch_size, num_ut=num_ut, scenario=scenario, elevation_angle=elevation_angle, bs_height=satellite_distance)
        channel_model.set_topology(*topology)

        self.assertTrue(1.2 < channel_model._scenario.gas_pathloss[0,0,0] < 1.3)
        self.assertTrue(1.0 < channel_model._scenario.scintillation_pathloss[0,0,0] < 1.2)
        self.assertTrue(210.5 < channel_model._scenario.free_space_pathloss < 210.7)

    

    
    def test_sc6_dl(self, direction = "downlink", scenario = "sur", carrier_frequency = 20e9,satellite_distance = 600000.0, elevation_angle = 30.0, batch_size = 1,num_ut = 1):        
        
        ut_array = Antenna(polarization="single",
                    polarization_type="V",
                    antenna_pattern="38.901",
                    carrier_frequency=carrier_frequency)
        
        bs_array = AntennaArray(num_rows=1,
                            num_cols=4,
                            polarization="dual",
                            polarization_type="VH",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)
        
        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)

        topology = utils.gen_single_sector_topology(batch_size=batch_size, num_ut=num_ut, scenario=scenario, elevation_angle=elevation_angle, bs_height=satellite_distance)
        channel_model.set_topology(*topology)

        self.assertTrue(0.4 < channel_model._scenario.gas_pathloss[0,0,0] < 0.6)
        self.assertTrue(0.3 < channel_model._scenario.scintillation_pathloss[0,0,0] < 0.5)
        self.assertTrue(179.0 < channel_model._scenario.free_space_pathloss < 179.2)


    


class Test_UL(unittest.TestCase):
    def test_sc1_ul(self, direction = "uplink", scenario = "sur", carrier_frequency = 30e9,satellite_distance = 35786000.0, elevation_angle = 12.5, batch_size = 1,num_ut = 1):        
            
            ut_array = Antenna(polarization="single",
                        polarization_type="V",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)
            
            bs_array = AntennaArray(num_rows=1,
                                num_cols=4,
                                polarization="dual",
                                polarization_type="VH",
                                antenna_pattern="38.901",
                                carrier_frequency=carrier_frequency)
            
            channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)

            topology = utils.gen_single_sector_topology(batch_size=batch_size, num_ut=num_ut, scenario=scenario, elevation_angle=elevation_angle, bs_height=satellite_distance)
            channel_model.set_topology(*topology)

            self.assertTrue(1.2 < channel_model._scenario.gas_pathloss[0,0,0] < 1.4)
            self.assertTrue(1.3 < channel_model._scenario.scintillation_pathloss[0,0,0] < 1.5)
            self.assertTrue(214.0 < channel_model._scenario.free_space_pathloss < 214.2)

    def test_sc6_ul(self, direction = "uplink", scenario = "sur", carrier_frequency = 30e9,satellite_distance = 600000.0, elevation_angle = 30.0, batch_size = 1,num_ut = 1):        
            
            ut_array = Antenna(polarization="single",
                        polarization_type="V",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)
            
            bs_array = AntennaArray(num_rows=1,
                                num_cols=4,
                                polarization="dual",
                                polarization_type="VH",
                                antenna_pattern="38.901",
                                carrier_frequency=carrier_frequency)
            
            channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)

            topology = utils.gen_single_sector_topology(batch_size=batch_size, num_ut=num_ut, scenario=scenario, elevation_angle=elevation_angle, bs_height=satellite_distance)
            channel_model.set_topology(*topology)

            self.assertTrue(0.5 < channel_model._scenario.gas_pathloss[0,0,0] < 0.7)
            self.assertTrue(0.4 < channel_model._scenario.scintillation_pathloss[0,0,0] < 0.6)
            self.assertTrue(182.5 < channel_model._scenario.free_space_pathloss < 182.7)

if __name__ == '__main__':
    unittest.main()