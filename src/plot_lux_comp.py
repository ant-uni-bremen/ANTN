import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Import Sionna
import sionna

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

import matplotlib.pyplot as plt
import numpy as np
import time
import pickle

from sionna.mimo import StreamManagement

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers

# These functions also exist in sionna.channel.tr38901 but are not compatable with 3GPP TR38.811
from sionna.channel.tr38811 import Antenna, AntennaArray
from sionna.channel.tr38811.utils import gen_single_sector_topology as gen_ntn_topology

from sionna.channel.tr38811 import DenseUrban, Urban, SubUrban

from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

from sionna.mapping import Mapper, Demapper

from sionna.utils import BinarySource, ebnodb2no, sim_ber, QAMSource

scenarios = ["dur"] # dur is the DenseUrban scenario
carrier_frequency = 20.0e9 # UL S-Band
direction = "downlink"
#elevation_angle = 80.0
num_ut = 1
satellite_height = 50000.0 # Height in meters, this is a satellite in the Low Earth Orbit (LEO)
batch_size = 25 # Number of topologies we will generate later  

for elevation_angle in range(10,50):
    ut_array = AntennaArray(num_rows=20,
                            num_cols=20,
                            polarization="single",
                            polarization_type="V",
                            antenna_pattern="omni",
                            carrier_frequency=carrier_frequency)

    # The satellite is the basestation, so we name it bs. 
    bs_array = AntennaArray(num_rows=1,
                            num_cols=1,
                            polarization="single",
                            polarization_type="H",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)

    for scenario in scenarios:
        # Here we match choose DenseUrban to match the parameter "dur" for the scenario defined above
        if scenario == "dur":
            channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                    ut_array=ut_array,
                                    bs_array=bs_array,
                                    direction=direction,
                                    elevation_angle=elevation_angle)
        if scenario == "sur":
            channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                    ut_array=ut_array,
                                    bs_array=bs_array,
                                    direction=direction,
                                    elevation_angle=elevation_angle)
        if scenario == "urb":
            channel_model = Urban(carrier_frequency=carrier_frequency,
                                    ut_array=ut_array,
                                    bs_array=bs_array,
                                    direction=direction,
                                    elevation_angle=elevation_angle)
        # Generate the topology
        topology = gen_ntn_topology(batch_size=batch_size, num_ut=num_ut, scenario=scenario,bs_height=satellite_height)

        # Set the topology
        channel_model.set_topology(*topology)

        num_time_steps = 1
        sampling_frequency = 14 * 10**(6)

        # path_coefficients [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        # path_delays [batch size, num_rx, num_tx, num_paths]
        path_coefficients, path_delays = channel_model(num_time_steps, sampling_frequency)

        # Plot the imaginary and real part of each path coefficient for one Tx Rx pair and one time step
        #real_part = tf.math.real(path_coefficients[0,0,0,0,0,:,0])
        #imag_part = tf.math.imag(path_coefficients[0,0,0,0,0,:,0])

        # Labels for each point
        #labels = ["1", "2", "3", "4", "5", "6", "7", "8"]

        # Plot each point individually and assign a label
        #for i in range(len(labels)):
        #    plt.scatter(real_part[i], imag_part[i])#, label=labels[i])
        #path_coefficients = tf.reduce_sum(path_coefficients, axis=-2)
        path_coefficients = sionna.channel.cir_to_ofdm_channel(frequencies=carrier_frequency,a=path_coefficients,tau=path_delays)
        for i in range(batch_size):
            #for u_idx in range(400):
            plt.scatter(tf.math.real(path_coefficients[i]), tf.math.imag(path_coefficients[i]), c="green", alpha=0.7)
        print(tf.shape(path_coefficients))

# Add legend
plt.legend()
# Adding x and y axes through the origin
plt.axhline(0, color='black', linewidth=0.5)  # Horizontal axis (y=0)
plt.axvline(0, color='black', linewidth=0.5)  # Vertical axis (x=0)

# Adding labels for the axes
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')

real_part = tf.math.real(path_coefficients)#[0,0,0,0,0,0,:])
imag_part = tf.math.imag(path_coefficients)#[0,0,0,0,0,0,:])

#max_limit = max(max(abs(real_part)), max(abs(imag_part)))*1.1
max_limit = 1.2*10.0**(-6.0)
plt.xlim(-max_limit, max_limit)
plt.ylim(-max_limit, max_limit)

# Displaying the plot
plt.grid(True)  # Optional: add a grid for better visualization

plt.title("Channel Gains")



plt.show()